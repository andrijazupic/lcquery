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

CATALINA = EarthLocation.from_geodetic(
    lon=-110.7317 * u.deg, lat=32.4417 * u.deg, height=2510.0 * u.m)

V_F0_JY = 3636.0
V_ZP_UJY = 2.5 * np.log10(V_F0_JY * 1e6)           

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
                return None, "no_link"        
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


def fetch_crts_lc(source_id, ra, dec, radius_arcsec=6.0,
                  database="photcat", timeout=90, retries=3, clean=True):
    """
    --------------------------------------------------------------------------------
    CRTS / CSDR  (Catalina Real-Time Transient Survey / Catalina Surveys Data Release)
    --------------------------------------------------------------------------------
    Access. Single-position cone search against the Catalina Surveys photometry
    server (getcssconedb_release_img.cgi, database photcat), submitted as
    multipart/form-data (the CGI ignores urlencoded POSTs). Returns one CSV with
    columns MasterID, Mag, Magerr, RA, Dec, MJD, Blend. If more than one MasterID
    falls in the cone, only the object nearest the input position is kept. Filter
    label: crts-clear (CSS observes unfiltered). Detection (not forced) SExtractor
    aperture photometry from the 0.7 m Catalina Schmidt; 30 s exposures, four per
    field separated by ~10 min.

    Time -> BJD_TDB. The server returns the topocentric (Earth-based, uncorrected)
    observed time as MJD on the UTC scale (confirmed by the standard literature
    practice of barycentre-correcting CRTS times before folding). The code treats it
    as topocentric UTC at the Catalina Schmidt site, scale-converts to TDB, and adds
    the barycentric light-travel time to the source:

        t       = Time(MJD, format="mjd", scale="utc", location=CATALINA)
        BJD_TDB = (t.tdb + t.light_travel_time(source_coord, kind="barycentric")).jd

    The observatory choice affects only the sub-millisecond topocentric term, so it
    is valid for all photcat sub-surveys. A <= 15 s start-vs-midpoint ambiguity on
    the 30 s exposures is undocumented but negligible for hour-to-day periods.

    Flux -> uJy. CRTS is unfiltered, with the instrumental magnitude transformed to
    an approximate Johnson/Cousins V (Vega-based) using standard stars; the single
    Mag column is this V_CSS magnitude. It is converted to flux density via
    flux[uJy] = 10^((ZP - Mag)/2.5) with ZP = 2.5*log10(3636e6) = 23.9015
    (zero-point flux 3636 Jy, the Johnson V Vega value), and
    flux_err = flux * Magerr / 1.0857. System "Vega", F0 = 3636 Jy. The 3636 Jy
    Johnson-V Vega constant is used for internal consistency: V_CSS is a Vega V band,
    so it matches both the "Vega" label and the ogle-v override, which is the same
    kind of band. (The earlier AB value 3631 Jy differed by only ~0.0015 mag -- about
    50x below CRTS's own calibration scatter -- so the change is cosmetic for the
    science.) The result is an approximate-V (Vega) flux density in uJy, suitable for
    relative variability, not precise absolute or cross-band photometry.

    Cleaning. Always: drop NaN in Mag, Magerr, MJD. With clean=True: keep Blend == 0
    (the CSDR blending flag marking sources unresolved from a neighbour within
    ~2-3 arcsec) and Magerr > 0; with clean=False, only Magerr > 0. Multi-object
    cones are reduced to the nearest MasterID. Not applied (optional hardening): a
    magnitude cut (~11.5 < V < 20) to drop saturation artifacts and faint-end noise
    where CSDR photometry is least reliable.
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "CRTS", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "CRTS", "source_id": sid,
                     "observations": len(df), "status": "success"})

    raw, reason = _fetch_table(ra, dec, radius_arcsec / 60.0, database, timeout, retries)
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

    if raw["MasterID"].nunique() > 1:
        sep2 = ((raw["RA"] - float(ra)) * np.cos(np.radians(float(dec)))) ** 2 \
               + (raw["Dec"] - float(dec)) ** 2
        raw = raw[raw["MasterID"] == raw.loc[sep2.idxmin(), "MasterID"]]
    raw = raw.reset_index(drop=True)

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    t = Time(raw["MJD"].to_numpy(float), format="mjd", scale="utc", location=CATALINA)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    mag = raw["Mag"].to_numpy(float)
    magerr = raw["Magerr"].to_numpy(float)
    flux = 10 ** ((V_ZP_UJY - mag) / 2.5)
    flux_err = flux * magerr / 1.0857                              

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          "crts-clear",
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)