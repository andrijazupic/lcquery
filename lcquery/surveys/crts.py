import io
import re
import time as _time
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import requests
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

# Catalina Schmidt (CSS) station, Mt. Bigelow, AZ -- the telescope behind the
# photcat data. Topocentric term is sub-ms, so the exact site barely matters,
# but CSS is the correct one for the returned photometry.
CATALINA = EarthLocation.from_geodetic(
    lon=-110.7317 * u.deg, lat=32.4417 * u.deg, height=2510.0 * u.m)

# CRTS is unfiltered, transformed to approximate Johnson V (Vega), NOT AB.
# Convert via the V-band Vega zero-point flux (~3631 Jy); numerically ~the AB
# 23.9, but it is the V (Vega) zero point.
V_F0_JY = 3631.0
V_ZP_UJY = 2.5 * np.log10(V_F0_JY * 1e6)             # ~23.90; flux = 10**((V_ZP_UJY - mag)/2.5)

_CGI = "http://nunuku.caltech.edu/cgi-bin/getcssconedb_release_img.cgi"


def _fetch_table(ra, dec, radius_arcmin, database, timeout, retries):
    """Run the cone query and return the raw photometry DataFrame, or None.

    Confirmed from a real browser request (DevTools): POST with
    Content-Type multipart/form-data, fields RADec, Rad, IMG, DB, .submit,
    OUT, SHORT, PLOT.

    CRITICAL: use files= (multipart). Do NOT switch to data=<dict> -- that
    sends application/x-www-form-urlencoded, which this CGI ignores (it
    returns the bare form page). The (None, value) tuples force multipart
    text fields with no filename.
    """
    fields = {
        "RADec":   (None, f"{float(ra):.6f} {float(dec):.6f}"),
        "Rad":     (None, str(radius_arcmin)),
        "IMG":     (None, "nun"),
        "DB":      (None, database),
        ".submit": (None, "Submit"),
        "OUT":     (None, "csv"),
        "SHORT":   (None, "short"),
        "PLOT":    (None, "plot"),
    }
    reason = "exception"
    for attempt in range(retries):
        try:
            r = requests.post(_CGI, files=fields, timeout=timeout)
            r.raise_for_status()
            text = r.text
            if text.lstrip().startswith("MasterID"):
                return pd.read_csv(io.StringIO(text)), "ok"
            m = re.search(r'([^\s"\'<>]*result_web_file\w+\.csv)', text)
            if not m:
                m = re.search(r'href=["\']?([^\s"\'<>]+\.csv)', text, re.I)
            if not m:
                return None, "no_link"          # got a page, no CSV -> no data / no coverage
            csv_url = urljoin(r.url, m.group(1))
            rr = requests.get(csv_url, timeout=timeout)
            rr.raise_for_status()
            return pd.read_csv(io.StringIO(rr.text)), "ok"
        except Exception:
            reason = "exception"
            if attempt == retries - 1:
                return None, reason
            _time.sleep(3 * (attempt + 1))
    return None, reason


def fetch_crts_lc(source_id, ra, dec, radius_arcmin=0.1,
                  database="photcat", timeout=90, retries=3, clean=True):
    """
    CRTS / CSDR3 light curve for one source, via the single-position cone search.

    time   : BJD_TDB (full JD). Service gives MJD (UTC, topocentric); we apply
             the same barycentric correction as BlackGEM/ZTF.
    flux   : approximate-V (Vega) flux density in micro-Jy via the V zero point
             (~3631 Jy). Same UNIT as the others, but a clear/V-ish band
             (~between Sloan g and r) and noisy -- watch bright outliers
             (blend/saturation artifacts).
    filter : "crts-clear" (CSS observes unfiltered; mags are approx-V calibrated).
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "CRTS", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "CRTS", "source_id": sid,
                     "observations": len(df), "status": "success"})

    raw, reason = _fetch_table(ra, dec, radius_arcmin, database, timeout, retries)
    if raw is None:
        return result("no_data" if reason == "no_link" else "search_failed")
    raw = raw.rename(columns=lambda c: str(c).strip())

    required = {"MasterID", "Mag", "Magerr", "RA", "Dec", "MJD", "Blend"}
    if not required.issubset(raw.columns):
        return result("missing_columns")

    for col in ["Mag", "Magerr", "RA", "Dec", "MJD", "Blend"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    raw = raw.dropna(subset=["Mag", "Magerr", "MJD"])
    if clean:
        raw = raw[(raw["Blend"] == 0) & (raw["Magerr"] > 0)]
    else:
        raw = raw[raw["Magerr"] > 0]
    if len(raw) == 0:
        return result("filtered_out")

    # If the cone caught >1 object, keep the nearest to the input position.
    if raw["MasterID"].nunique() > 1:
        sep2 = ((raw["RA"] - float(ra)) * np.cos(np.radians(float(dec)))) ** 2 \
               + (raw["Dec"] - float(dec)) ** 2
        raw = raw[raw["MasterID"] == raw.loc[sep2.idxmin(), "MasterID"]]
    raw = raw.reset_index(drop=True)

    # time: MJD (UTC, topocentric) -> BJD_TDB at the source position
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    t = Time(raw["MJD"].to_numpy(float), format="mjd", scale="utc", location=CATALINA)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    # flux: approximate-V (Vega) mag -> uJy
    mag = raw["Mag"].to_numpy(float)
    magerr = raw["Magerr"].to_numpy(float)
    flux = 10 ** ((V_ZP_UJY - mag) / 2.5)
    flux_err = flux * magerr / 1.0857                               # 1.0857 = 2.5/ln(10)

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          "crts-clear",
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)