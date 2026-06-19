import io
import time as _time
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

try:
    from dl import queryClient as qc
except ImportError:
    qc = None

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

CTIO = EarthLocation.from_geodetic(
    lon=-70.8065 * u.deg, lat=-30.1690 * u.deg, height=2207.0 * u.m)

KPNO = EarthLocation.from_geodetic(
    lon=-111.5997 * u.deg, lat=31.9633 * u.deg, height=2097.0 * u.m)

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
            if res and isinstance(res, str) and not res.startswith("ERROR"):
                return pd.read_csv(io.StringIO(res))
        except Exception:
            pass
        
        if attempt < retries - 1:
            _time.sleep(3 * (attempt + 1))
            
    return None

def fetch_nsc_lc(source_id, ra, dec, radius_arcsec=2.0, mag_choice="auto", band=None, clean=True, schema="nsc_dr2"):
    """
    --------------------------------------------------------------------------------
    NSC DR2  (NOIRLab Source Catalog DR2)
    --------------------------------------------------------------------------------
    Access. NOIRLab Source Catalog DR2 via the Astro Data Lab query client
    (dl.queryClient, ADQL on nsc_dr2.object, nsc_dr2.meas, nsc_dr2.exposure). A q3c
    cone search on object returns the nearest object id; meas (joined to exposure)
    then gives the per-measurement time series. Magnitude option is selectable (auto
    default, or fixed apertures aper1-aper8; a fixed aperture is often cleaner for
    time-series on point sources). Filter label is nsc-<filter> over u/g/r/i/z/Y/VR
    (lowercased -> nsc-u/g/r/i/z/y/vr). Detection SExtractor aperture photometry
    (mag_auto) on individual chip images.

    Time -> BJD_TDB. meas.mjd is the per-exposure topocentric UTC observation time
    (shutter-open / exposure start; verified against the exposure dateobs). Because
    NSC DR2 combines three instruments at two sites, the barycentric correction is
    applied per observatory, keyed on the exposure-table instrument field: k4m
    (Mayall + Mosaic3) and ksb (Bok + 90Prime), both on Kitt Peak, use the KPNO
    location; c4d (Blanco + DECam) uses CTIO. Each subset is then converted UTC -> TDB
    with the barycentric light-travel time to the source:

        t       = Time(mjd, format="mjd", scale="utc", location=<KPNO or CTIO>)
        BJD_TDB = (t.tdb + t.light_travel_time(source_coord, kind="barycentric")).jd

    (Splitting on instrument rather than parsing the exposure name avoids
    mis-assignment, since DECam names appear in multiple formats, e.g. c4d_... and
    tu....) The site choice enters only at the ~tens-of-ms level. meas.mjd is the
    exposure start, so the pipeline adds 0.5*exptime -- taken per-exposure from the
    joined exposure table, since NSC's exposure times vary across its constituent
    surveys -- to place the timestamp at mid-exposure, matching the convention used for
    the other surveys.

    Flux -> uJy. NSC DR2 magnitudes are AB (zero-points tied to Pan-STARRS1,
    ATLAS-Refcat2, SkyMapper DR1, and r+Gaia-G for VR). Converted via
    Target_flux[uJy] = 10^((23.9 - mag)/2.5) and
    Target_flux_err = Target_flux * 0.921034 * magerr (= flux * magerr / 1.0857). The
    chosen magnitude (mag_auto default) is SExtractor aperture photometry; VR is an
    approximately-AB hybrid band. System "AB", F0 = 3631 Jy.

    Cleaning. The SQL enforces mag IS NOT NULL and magerr > 0 always (the latter even
    when clean=False, protecting error propagation); with clean=True it additionally
    requires flags = 0 (SExtractor FLAGS clean -- no neighbour/blend/saturation/
    truncation). A post-query dropna removes residual nulls. An optional band argument
    restricts to one filter.
    """
    sid = int(source_id)
    empty_out = pd.DataFrame(columns=_COLS)

    def result(status, df=empty_out):
        n = len(df) if df is not None else 0
        return (df, {"survey": "NSC", "source_id": sid, "observations": n, "status": status})

    if mag_choice not in _MAG_MAP:
        raise ValueError(f"mag_choice must be one of {list(_MAG_MAP)}")
    
    mag_col, emag_col = _MAG_MAP[mag_choice]

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

    band_cut = f"AND m.filter = '{band}'" if band else ""
    
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

    bjd_tdb = np.zeros(len(raw))
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    raw["mjd_mid"] = raw["mjd"].astype(float) + 0.5 * raw["exptime"].astype(float) / 86400.0
    
    kpno_mask = raw["instrument"].isin(["k4m", "ksb"])
    ctio_mask = ~kpno_mask

    if kpno_mask.any():
        t_kpno = Time(raw.loc[kpno_mask, "mjd_mid"].to_numpy(float), format="mjd", scale="utc", location=KPNO)
        bjd_tdb[kpno_mask] = (t_kpno.tdb + t_kpno.light_travel_time(coord, kind="barycentric")).jd

    if ctio_mask.any():
        t_ctio = Time(raw.loc[ctio_mask, "mjd_mid"].to_numpy(float), format="mjd", scale="utc", location=CTIO)
        bjd_tdb[ctio_mask] = (t_ctio.tdb + t_ctio.light_travel_time(coord, kind="barycentric")).jd

    mag = raw["mag"].to_numpy(float)
    magerr = raw["magerr"].to_numpy(float)
    
    flux_ujy = 10 ** ((AB_ZP_UJY - mag) / 2.5)
    flux_err_ujy = flux_ujy * 0.921034 * magerr

    filter_array = ("nsc-" + raw["filter"].astype(str).str.lower()).to_numpy()

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux_ujy,
        "Target_flux_err": flux_err_ujy,
        "Filter":          filter_array,
    }).sort_values("BJD").reset_index(drop=True)

    return result("success", out)