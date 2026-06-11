import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from ztfquery import lightcurve

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

PALOMAR = EarthLocation.from_geodetic(
    lon=-116.8650 * u.deg, lat=33.3563 * u.deg, height=1712.0 * u.m)

AB_ZP_UJY = 23.9                                     # ZTF is PS1-AB calibrated
_ZTF_BANDS = {"zg": "ztf-g", "zr": "ztf-r", "zi": "ztf-i"}
_REQUIRED = {"mjd", "mag", "magerr", "filtercode"}   # columns a real LC has


def fetch_ztf_lc(source_id, ra, dec, radius_arcsec=1.5, bad_catflags_mask=32768):
    """
    ZTF PSF-photometry light curve(s) for one source, via IRSA (ztfquery).
    Credentials are handled by ztfquery's stored-login flow (~/.ztfquery),
    not passed here. time: BJD_TDB (full JD), from mjd(UTC) at the source
    position. flux: AB uJy = 10**((23.9 - mag)/2.5). filter: ztf-g/r/i.
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "ZTF", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "ZTF", "source_id": sid,
                     "observations": len(df), "status": "success"})

    try:
        lcq = lightcurve.LCQuery.from_position(
            float(ra), float(dec), radius_arcsec, BAD_CATFLAGS_MASK=bad_catflags_mask)
        df = lcq.data
    except Exception:
        return result("search_failed")
    if df is None or len(df) == 0 or not _REQUIRED.issubset(df.columns):
        return result("no_data")

    df = df.dropna(subset=["mjd", "mag", "magerr", "filtercode"])
    df = df[df["filtercode"].isin(_ZTF_BANDS)]
    if len(df) == 0:
        return result("filtered_out")
    df = df.reset_index(drop=True)

    # time: mjd (UTC, topocentric) -> BJD_TDB at the single source position
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    t = Time(df["mjd"].to_numpy(float), format="mjd", scale="utc", location=PALOMAR)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    # flux: AB mag -> uJy
    mag = df["mag"].to_numpy(float)
    magerr = df["magerr"].to_numpy(float)
    flux = 10 ** ((AB_ZP_UJY - mag) / 2.5)
    flux_err = flux * magerr / 1.0857                # 1.0857 = 2.5/ln(10)

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          df["filtercode"].map(_ZTF_BANDS).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)