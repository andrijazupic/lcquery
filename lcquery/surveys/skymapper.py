import numpy as np
import pandas as pd
import pyvo
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import time as _time

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

# Siding Spring Observatory, NSW, Australia (1.35 m SkyMapper). Single site;
# topocentric term is sub-21 ms, negligible vs SkyMapper's ~100 s exposures.
SIDING_SPRING = EarthLocation.from_geodetic(
    lon=149.0644 * u.deg, lat=-31.2733 * u.deg, height=1165.0 * u.m)

# SkyMapper ugriz are AB (DR4 ZPs anchored to Gaia XP synthetic photometry).
AB_ZP_UJY = 23.9

TAP_URL = "https://api.skymapper.nci.org.au/public/tap/"
_SVC = None


def _svc():
    global _SVC
    if _SVC is None:
        _SVC = pyvo.dal.TAPService(TAP_URL)
    return _SVC


def _tap_df(adql, retries=3):
    """Cached TAP service + retry with backoff. Returns a DataFrame; re-raises
    the last exception if all attempts fail, so the caller can surface it."""
    last = None
    for attempt in range(retries):
        try:
            return _svc().search(adql).to_table().to_pandas()
        except Exception as e:
            last = e
            if attempt < retries - 1:
                _time.sleep(3 * (attempt + 1))
    raise last


def fetch_skymapper_lc(source_id, ra, dec, radius_arcsec=3.0, clean=True):
    """
    SkyMapper Southern Survey DR4 light curve for one source, via VO TAP.

    time   : BJD_TDB (full JD). images.date + 0.5*exp_time = mid-exposure MJD,
             taken as UTC (topocentric) and barycentric-corrected.
    flux   : AB micro-Jy = 10**((23.9 - mag_psf)/2.5). g/r/i/z share the AB
             scale of BlackGEM/ZTF/NSC/ATLAS; u/v are unique blue bands.
    filter : "skymapper-u/v/g/r/i/z".

    clean  : flags<4 AND nimaflags<5 (SkyMapper's documented good-detection cut).
             Southern survey (Dec <~ +16); northern targets return no match.
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "SkyMapper", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "SkyMapper", "source_id": sid,
                     "observations": len(df), "status": "success"})

    rdeg = radius_arcsec / 3600.0
    try:
        # nearest object: select the distance AS an alias and ORDER BY the alias.
        # (SkyMapper's ADQL engine rejects ORDER BY on the raw DISTANCE() expr,
        #  which is what your old working code avoided.)
        obj = _tap_df(f"""SELECT TOP 1 m.object_id,
                   DISTANCE(POINT('ICRS', m.raj2000, m.dej2000),
                            POINT('ICRS', {float(ra)}, {float(dec)})) AS angdist
            FROM dr4.master AS m
            WHERE 1 = CONTAINS(POINT('ICRS', m.raj2000, m.dej2000),
                               CIRCLE('ICRS', {float(ra)}, {float(dec)}, {rdeg}))
            ORDER BY angdist ASC""")
        if len(obj) == 0:
            return result("no_match")
        objid = int(obj["object_id"][0])

        flag_cut = "AND p.flags < 4 AND p.nimaflags < 5" if clean else "AND p.e_mag_psf > 0"
        tab = _tap_df(f"""SELECT p.filter,
                   (i.date + 0.5 * i.exp_time / 86400.0) AS mjd_mid,
                   p.mag_psf AS mag, p.e_mag_psf AS magerr
            FROM dr4.photometry AS p JOIN dr4.images AS i ON p.image_id = i.image_id
            WHERE p.object_id = {objid} AND p.mag_psf IS NOT NULL {flag_cut}
            ORDER BY mjd_mid""")
    except Exception as e:
        return result(f"query_failed: {type(e).__name__}: {e}")   # surfaces the real cause

    if tab is None or len(tab) == 0:
        return result("no_data")

    for col in ["mjd_mid", "mag", "magerr"]:
        tab[col] = pd.to_numeric(tab[col], errors="coerce")
    tab = tab.dropna(subset=["mjd_mid", "mag", "magerr"])
    tab = tab[tab["magerr"] > 0]
    if len(tab) == 0:
        return result("filtered_out")
    tab = tab.reset_index(drop=True)

    # time: mid-exposure MJD (UTC, topocentric) -> BJD_TDB at the source
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    t = Time(tab["mjd_mid"].to_numpy(float), format="mjd", scale="utc", location=SIDING_SPRING)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    # flux: AB mag -> uJy
    mag = tab["mag"].to_numpy(float)
    magerr = tab["magerr"].to_numpy(float)
    flux = 10 ** ((AB_ZP_UJY - mag) / 2.5)
    flux_err = flux * magerr / 1.0857               # 1.0857 = 2.5/ln(10)

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          ("skymapper-" + tab["filter"].astype(str).str.lower()).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)