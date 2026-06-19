import numpy as np
import pandas as pd
import pyvo
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import time as _time

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

OAJ = EarthLocation.from_geodetic(
    lon=-1.0163 * u.deg, lat=40.0420 * u.deg, height=1957.0 * u.m)

AB_ZP_UJY = 23.9                                  # J-VAR mags are AB (Oke & Gunn 1983)
_MISSING = (99.0, 99.000)

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
    --------------------------------------------------------------------------------
    J-VAR DR1  (Javalambre VARiability survey)
    --------------------------------------------------------------------------------
    Access & format. J-VAR DR1 light curves via the CEFCA VO Cone Search (SCSService
    on JVAR.LIGHT_CURVES), by position. The service returns one row per
    (object, filter) with array-valued MJD/MAG/MAG_ERR/FLAGS columns (one entry per
    epoch), which the code explodes into per-epoch rows. If the cone catches more
    than one OBJ_ID, the nearest to the input position is kept. Filter labels come
    from FILTER (uppercased) via _JVAR_BANDS: jvar-g/jvar-r/jvar-i for the SDSS broad
    bands and jvar-j0395/j0515/j0660/j0861 for the narrow bands (exactly seven of the
    twelve J-PLUS filters). Detection aperture photometry from the JAST80 telescope
    at the Observatorio Astrofisico de Javalambre (OAJ); >= 11 epochs per field spread 
    over ~1 yr, three images per filter per visit. Per-filter exposures: g 33 s, r 40 s, 
    i 34 s, J0395 87 s, J0515 40 s, J0660 135 s, J0861 160 s; the seven filters are 
    cycled three times per visit, giving ~12.7 min median between consecutive same-band 
    exposures.

    Time -> BJD_TDB. The per-epoch MJD is the per-image observation time on the UTC
    scale, topocentric (the CEFCA/jype pipeline records the observation-frame MJD; no
    barycentric correction is pre-applied). The code treats it as topocentric UTC at
    the OAJ, scale-converts to TDB, and adds the barycentric light-travel time to the
    source:

        t       = Time(MJD, format="mjd", scale="utc", location=OAJ)
        BJD_TDB = (t.tdb + t.light_travel_time(source_coord, kind="barycentric")).jd

    The OAJ site enters only at the sub-millisecond level; a <= tens-of-seconds
    start-vs-midpoint ambiguity on the per-image MJD is negligible for the periods of
    interest.

    Flux -> uJy. J-VAR DR1 magnitudes are on the AB system (Oke & Gunn 1983),
    produced by ensemble differential photometry calibrated to J-PLUS DR3. They are
    converted via Target_flux[uJy] = 10^((23.9 - MAG)/2.5) and
    Target_flux_err = Target_flux * MAG_ERR / 1.0857. Missing/invalid magnitudes
    (sentinel 99.0) are set to NaN and dropped. System "AB", F0 = 3631 Jy.

    Cleaning. The FLAGS column is a summed bitmask: SExtractor native flags (1-128)
    plus CEFCA additions -- 256 (strict saturation, peak > 50000 ADU), 512 (FWHM
    exceeds the proximity limit; blending), 1024 (cross-matched but invalid
    photometry), 2048 (no cross-match within 1.1 arcsec; not detected). Always: drop
    NaN in mjd/mag/magerr and require magerr > 0. Then, matching the J-VAR DR1 paper's
    own definitions: clean=True keeps flag == 0 (the paper's "no issues" cut, used
    for strict variability indices); clean=False keeps flag < 1024 (the paper's
    documented "valid" cut -- a real, useful measurement, excluding invalid photometry
    and non-detections while tolerating SExtractor flags, saturation, and proximity
    warnings).
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

    if "OBJ_ID" in lc.columns and lc["OBJ_ID"].nunique() > 1 \
            and {"RA", "DEC"}.issubset(lc.columns):
        sep2 = ((lc["RA"] - float(ra)) * np.cos(np.radians(float(dec)))) ** 2 \
               + (lc["DEC"] - float(dec)) ** 2
        lc = lc[lc["OBJ_ID"] == lc.loc[sep2.idxmin(), "OBJ_ID"]]

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

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    t = Time(ph["mjd"].to_numpy(float), format="mjd", scale="utc", location=OAJ)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

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