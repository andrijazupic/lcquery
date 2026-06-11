import numpy as np
import pandas as pd
import pyvo
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import time as _time

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

# Observatorio Astrofisico de Javalambre (OAJ), Teruel, Spain (JAST80/T80).
OAJ = EarthLocation.from_geodetic(
    lon=-1.0163 * u.deg, lat=40.0420 * u.deg, height=1957.0 * u.m)

AB_ZP_UJY = 23.9                                  # J-VAR mags are AB (Oke & Gunn 1983)
_MISSING = (99.0, 99.000)

# broad g/r/i -> short labels (group with other surveys); narrow bands kept distinct
_JVAR_BANDS = {"GSDSS": "jvar-g", "RSDSS": "jvar-r", "ISDSS": "jvar-i",
               "J0395": "jvar-j0395", "J0515": "jvar-j0515",
               "J0660": "jvar-j0660", "J0861": "jvar-j0861"}

SCS_LC = "https://archive.cefca.es/catalogues/vo/cone/jvar-dr1/JVAR.LIGHT_CURVES"


def _scs_df(url, ra, dec, radius_deg, retries=3):
    last = None
    for attempt in range(retries):
        try:
            return pyvo.dal.SCSService(url).search(
                pos=(float(ra), float(dec)), radius=float(radius_deg)).to_table().to_pandas()
        except Exception as e:
            last = e
            if attempt < retries - 1:
                _time.sleep(3 * (attempt + 1))
    raise last


def _to_array(x, as_int=False):
    """Parse a J-VAR array cell (sequence or space-separated string)."""
    if x is None or (np.isscalar(x) and pd.isna(x)):
        return np.array([], int if as_int else float)
    seq = x if isinstance(x, (list, tuple, np.ndarray)) else str(x).split()
    vals = []
    for t in seq:
        try:
            v = float(t)
        except (TypeError, ValueError):
            v = np.nan
        vals.append(v)
    arr = np.asarray(vals, float)
    if not as_int:
        arr[np.isin(arr, _MISSING)] = np.nan
    return arr


def fetch_jvar_lc(source_id, ra, dec, radius_arcsec=3.0, clean=True):
    """
    J-VAR DR1 (JAST80, Javalambre) light curve for one source, via VO Cone Search.

    time   : BJD_TDB (full JD). mjd is per-image UTC (topocentric), barycentric-
             corrected at the source position. flux: AB micro-Jy. filter: jvar-*.

    clean=True  -> flag == 0   (pristine; J-VAR's strict variability-index cut)
    clean=False -> flag < 1024 (J-VAR's documented light-curve "valid" cut)
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "J-VAR", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "J-VAR", "source_id": sid,
                     "observations": len(df), "status": "success"})

    rdeg = radius_arcsec / 3600.0
    try:
        lc = _scs_df(SCS_LC, ra, dec, rdeg)
    except Exception as e:
        return result(f"query_failed: {type(e).__name__}: {e}")
    if lc is None or len(lc) == 0:
        return result("no_match")

    lc = lc.rename(columns={c: c.upper() for c in lc.columns})
    if not {"FILTER", "MJD", "MAG", "MAG_ERR"}.issubset(lc.columns):
        return result("missing_columns")
    has_flags = "FLAGS" in lc.columns

    # nearest object if the cone caught more than one
    if "OBJ_ID" in lc.columns and lc["OBJ_ID"].nunique() > 1 \
            and {"RA", "DEC"}.issubset(lc.columns):
        sep2 = ((lc["RA"] - float(ra)) * np.cos(np.radians(float(dec)))) ** 2 \
               + (lc["DEC"] - float(dec)) ** 2
        lc = lc[lc["OBJ_ID"] == lc.loc[sep2.idxmin(), "OBJ_ID"]]

    # explode the per-filter arrays (mjd/mag/magerr [+flags]) into per-epoch rows
    parts = []
    for _, row in lc.iterrows():
        mjd = _to_array(row["MJD"])
        mag = _to_array(row["MAG"])
        err = _to_array(row["MAG_ERR"])
        flg = _to_array(row["FLAGS"], as_int=True) if has_flags else np.zeros(len(mjd))
        n = min(len(mjd), len(mag), len(err), len(flg))
        if n == 0:
            continue
        parts.append(pd.DataFrame({
            "filter": str(row["FILTER"]),
            "mjd": mjd[:n], "mag": mag[:n], "magerr": err[:n], "flag": flg[:n]}))
    if not parts:
        return result("no_data")
    ph = pd.concat(parts, ignore_index=True)

    ph = ph.dropna(subset=["mjd", "mag", "magerr"])
    ph = ph[ph["magerr"] > 0]
    if has_flags:
        ph = ph[ph["flag"] == 0] if clean else ph[ph["flag"] < 1024]
    if len(ph) == 0:
        return result("filtered_out")
    ph = ph.reset_index(drop=True)

    # time: per-image MJD (UTC) -> BJD_TDB at the source
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    t = Time(ph["mjd"].to_numpy(float), format="mjd", scale="utc", location=OAJ)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    # flux: AB mag -> uJy
    mag = ph["mag"].to_numpy(float)
    magerr = ph["magerr"].to_numpy(float)
    flux = 10 ** ((AB_ZP_UJY - mag) / 2.5)
    flux_err = flux * magerr / 1.0857

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          (ph["filter"].astype(str).str.upper()
                            .map(_JVAR_BANDS)
                            .fillna("jvar-" + ph["filter"].astype(str))).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)