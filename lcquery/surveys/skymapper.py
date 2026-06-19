import numpy as np
import pandas as pd
import pyvo
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import time as _time

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

SIDING_SPRING = EarthLocation.from_geodetic(
    lon=149.0644 * u.deg, lat=-31.2733 * u.deg, height=1165.0 * u.m)

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
    --------------------------------------------------------------------------------
    SkyMapper Southern Survey DR4
    --------------------------------------------------------------------------------
    Access & cross-match. VO TAP at the NCI (api.skymapper.nci.org.au/public/tap/).
    Nearest object in dr4.master within radius_arcsec (default 3 arcsec), via
    CONTAINS/CIRCLE with DISTANCE() aliased and ORDER BY on the alias (the engine
    rejects ORDER BY on the raw DISTANCE()). Per-epoch detections then come from
    dr4.photometry joined to dr4.images on image_id. Southern-hemisphere survey
    (~ Dec <= +16 deg, some fields to ~+28 deg); northern targets return no_match.
    Filter label skymapper-<u/v/g/r/i/z>. Detection PSF photometry (mag_psf) from
    Siding Spring Observatory.

    Time -> BJD_TDB. images.date is the MJD at exposure start (UTC; the IMAGE_ID
    encodes the UT shutter-open time, DATE is the precise start MJD). The pipeline
    forms the mid-exposure MJD as date + 0.5*exp_time/86400, using per-image exp_time
    so Main (100 s) and Shallow (shorter) exposures are both correct. The mid-exposure
    UTC is treated as topocentric at Siding Spring and converted:

        t       = Time(date + 0.5*exp_time/86400, format="mjd", scale="utc", location=SIDING_SPRING)
        BJD_TDB = (t.tdb + t.light_travel_time(source_coord, kind="barycentric")).jd

    The half-exposure offset is significant (50 s at a 100 s exposure); omitting it
    would bias every timestamp 50 s early. The observatory choice affects only the
    sub-21 ms topocentric term.

    Flux -> uJy. mag_psf is the calibrated AB PSF magnitude (DR4 zero-points anchored
    to Gaia XP synthetic photometry -- a key change versus the APASS-based DR1).
    Converted via Target_flux[uJy] = 10^((23.9 - mag_psf)/2.5) (ZP 23.9 =
    2.5*log10(3631e6), the exact AB->uJy zero point) and
    Target_flux_err = Target_flux * magerr / 1.0857. System "AB", F0 = 3631 Jy.
    g/r/i/z are cross-comparable with the other AB surveys (modulo small bandpass
    differences); u/v are unique to SkyMapper.

    Cleaning. Always mag_psf IS NOT NULL (SQL) and magerr > 0 (post-query). With
    clean=True, per-detection cuts are flags < 4 and nimaflags < 5 -- SkyMapper's
    documented criteria ("at least one good (flags<4 and nimaflags<5) photometric data
    point"): flags < 4 = SExtractor saturation bit (value 4) unset ("not saturated"),
    which also excludes the bespoke high-value flags (scattered light, cosmic rays),
    keeping only 0-3 (clean / minor neighbour / deblend); nimaflags < 5 (<= 4 flagged
    pixels) matches the NIMAFLAGS = 4 master-table inclusion threshold. With
    clean=False, the only requirement is e_mag_psf > 0. Both flag cuts act per-epoch
    on the photometry table, not the combined master flags; tightening to flags = 0,
    nimaflags = 0 (the strict DR1 calibrator cut) would yield cleaner but sparser
    photometry. The count-based nimaflags (not the imaflags type-bitmask) is the
    column matching the documented threshold. The pipeline deliberately ignores
    use_in_clipped: that flag marks epochs retained in the master-table sigma-clipped
    mean, so filtering on it would preferentially discard the deviating epochs
    (eclipses, outbursts) that carry the variability signal.
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
        return result(f"query_failed: {type(e).__name__}: {e}")  

    if tab is None or len(tab) == 0:
        return result("no_data")

    for col in ["mjd_mid", "mag", "magerr"]:
        tab[col] = pd.to_numeric(tab[col], errors="coerce")
    tab = tab.dropna(subset=["mjd_mid", "mag", "magerr"])
    tab = tab[tab["magerr"] > 0]
    if len(tab) == 0:
        return result("filtered_out")
    tab = tab.reset_index(drop=True)

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    t = Time(tab["mjd_mid"].to_numpy(float), format="mjd", scale="utc", location=SIDING_SPRING)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    mag = tab["mag"].to_numpy(float)
    magerr = tab["magerr"].to_numpy(float)
    flux = 10 ** ((AB_ZP_UJY - mag) / 2.5)
    flux_err = flux * magerr / 1.0857             

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          ("skymapper-" + tab["filter"].astype(str).str.lower()).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)