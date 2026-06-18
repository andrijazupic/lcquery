import io
import time as _time
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

# The astro-datalab package is required
try:
    from dl import queryClient as qc
except ImportError:
    qc = None

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

# --- Observatories ---
# CTIO: Cerro Tololo Inter-American Observatory (Chile) - DECam
CTIO = EarthLocation.from_geodetic(
    lon=-70.8065 * u.deg, lat=-30.1690 * u.deg, height=2207.0 * u.m)

# KPNO: Kitt Peak National Observatory (Arizona) - Mosaic3 / 90Prime
KPNO = EarthLocation.from_geodetic(
    lon=-111.5997 * u.deg, lat=31.9633 * u.deg, height=2097.0 * u.m)

# NSC DR2 is AB calibrated.
AB_ZP_UJY = 23.9

_MAG_MAP = {
    "auto":  ("mag_auto",  "magerr_auto"),
    "aper1": ("mag_aper1", "magerr_aper1"),
    "aper2": ("mag_aper2", "magerr_aper2"),
    "aper4": ("mag_aper4", "magerr_aper4"),
    "aper8": ("mag_aper8", "magerr_aper8"),
}

def _run_nsc_query(sql, retries=3):
    """Safely execute an astro-datalab query with exponential backoff retries."""
    if qc is None:
        raise ImportError("The 'astro-datalab' package is not installed. Run: pip install astro-datalab")
        
    for attempt in range(retries):
        try:
            res = qc.query(sql=sql, fmt="csv")
            # qc.query returns a string; if it fails, it usually returns an "ERROR" string
            if res and isinstance(res, str) and not res.startswith("ERROR"):
                return pd.read_csv(io.StringIO(res))
        except Exception:
            pass
        
        if attempt < retries - 1:
            _time.sleep(3 * (attempt + 1))
            
    return None

def fetch_nsc_lc(source_id, ra, dec, radius_arcsec=2.0, mag_choice="auto", band=None, clean=True, schema="nsc_dr2"):
    """
    Fetch NOIRLab Source Catalog (NSC DR2) light curves, separate observatories 
    for precise BJD_TDB timing, convert AB magnitudes to uJy, and return the DataFrame.
    """
    sid = int(source_id)
    empty_out = pd.DataFrame(columns=_COLS)

    def result(status, df=empty_out):
        n = len(df) if df is not None else 0
        return (df, {"survey": "NSC", "source_id": sid, "observations": n, "status": status})

    if mag_choice not in _MAG_MAP:
        raise ValueError(f"mag_choice must be one of {list(_MAG_MAP)}")
    
    mag_col, emag_col = _MAG_MAP[mag_choice]

    # 1. Spatial Crossmatch (Confirmed Postgres LIMIT 1)
    rdeg = radius_arcsec / 3600.0
    sql_obj = f"""
        SELECT o.id
        FROM {schema}.object AS o
        WHERE q3c_radial_query(o.ra, o.dec, {float(ra)}, {float(dec)}, {rdeg})
        ORDER BY q3c_dist(o.ra, o.dec, {float(ra)}, {float(dec)}) ASC
        LIMIT 1
    """
    
    df_obj = _run_nsc_query(sql_obj)
    if df_obj is None:
        return result("search_failed_or_timeout")
    if df_obj.empty:
        return result("no_match")
        
    objectid = str(df_obj.iloc[0, 0])

    # 2. Fetch the time-series measurements
    band_cut = f"AND m.filter = '{band}'" if band else ""
    
    # CRITICAL: Always enforce emag > 0 to protect error propagation, even if clean=False
    qual_cut = f"AND m.flags = 0 AND m.{emag_col} > 0" if clean else f"AND m.{emag_col} > 0"
    
    sql_meas = f"""
        SELECT m.mjd, m.filter, m.{mag_col} AS mag, m.{emag_col} AS magerr, e.instrument, e.exptime
        FROM {schema}.meas AS m
        JOIN {schema}.exposure AS e ON m.exposure = e.exposure
        WHERE m.objectid = '{objectid}'
        AND m.{mag_col} IS NOT NULL
        {band_cut}
        {qual_cut}
        ORDER BY m.mjd
    """
    
    raw = _run_nsc_query(sql_meas)
    if raw is None:
        return result("search_failed_or_timeout")
        
    if raw.empty:
        return result("no_data")

    raw = raw.dropna(subset=["mjd", "mag", "magerr", "instrument", "exptime"]).reset_index(drop=True)
    if len(raw) == 0:
        return result("filtered_out")

    # 3. Dynamic Time Standardization (The CTIO vs KPNO split)
    bjd_tdb = np.zeros(len(raw))
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    # m.mjd is the exposure START. Add 0.5*exptime per row (NSC exposure times vary
    # across DECam / Mosaic3 / 90Prime) to place the timestamp at mid-exposure,
    # matching the convention used for the other surveys.
    raw["mjd_mid"] = raw["mjd"].astype(float) + 0.5 * raw["exptime"].astype(float) / 86400.0
    
    kpno_mask = raw["instrument"].isin(["k4m", "ksb"])
    ctio_mask = ~kpno_mask

    # Calculate precise barycentric times based on which hemisphere took the image
    if kpno_mask.any():
        t_kpno = Time(raw.loc[kpno_mask, "mjd_mid"].to_numpy(float), format="mjd", scale="utc", location=KPNO)
        bjd_tdb[kpno_mask] = (t_kpno.tdb + t_kpno.light_travel_time(coord, kind="barycentric")).jd

    if ctio_mask.any():
        t_ctio = Time(raw.loc[ctio_mask, "mjd_mid"].to_numpy(float), format="mjd", scale="utc", location=CTIO)
        bjd_tdb[ctio_mask] = (t_ctio.tdb + t_ctio.light_travel_time(coord, kind="barycentric")).jd

    # 4. Flux Standardization: AB Magnitude to microJanskys (uJy)
    mag = raw["mag"].to_numpy(float)
    magerr = raw["magerr"].to_numpy(float)
    
    flux_ujy = 10 ** ((AB_ZP_UJY - mag) / 2.5)
    flux_err_ujy = flux_ujy * 0.921034 * magerr

    # 5. Build final DataFrame
    filter_array = ("nsc-" + raw["filter"].astype(str).str.lower()).to_numpy()

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux_ujy,
        "Target_flux_err": flux_err_ujy,
        "Filter":          filter_array,
    }).sort_values("BJD").reset_index(drop=True)

    return result("success", out)