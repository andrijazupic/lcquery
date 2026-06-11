import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from google.cloud import bigquery

# --- Configuration & Constants ---
BG_PROJECT = "blackgem-databases-access"
BG_DB = "blackgem-full-source-db.blackgem_fullsource_v2"

# Hardcoded La Silla avoids network timeouts
LA_SILLA = EarthLocation.from_geodetic(lon=-70.7345*u.deg, lat=-29.2584*u.deg, height=2400.0*u.m)

# Lazy client initialization
_CLIENT = None
def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = bigquery.Client(project=BG_PROJECT)
    return _CLIENT

# --- Main Fetch Function ---
def fetch_blackgem_lc(source_id, ra, dec, 
                      client=None, db=BG_DB, qc_keep=("green", "yellow"), clean=True):
    """
    Fetch BlackGEM optimal-photometry, clean quality flags and reference
    templates, standardize to BJD_TDB and uJy, and return 
    both the DataFrame and a metadata dictionary.
    """

    if client is None:
        client = _get_client()

    # 2. Optimized query (stripped out BJD_OBS and MAG_OPT)
    query = f"""
        SELECT i.MJD_OBS, i.FILTER, i.FILENAME, i.QC_FLAG,
               d.FNU_OPT, d.FNUERRTOT_OPT, d.FLAGS_MASK, d.FLAGS_OPT
        FROM `{db}.detections` AS d
        JOIN `{db}.images` AS i ON d.IMAGE_ID = i.IMAGE_ID
        WHERE d.SOURCE_ID = @sid
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("sid", "INT64", int(source_id))]
    )
    
    df = client.query(query, job_config=job_config).to_dataframe()
    
    empty_out = pd.DataFrame(columns=["BJD", "Target_flux", "Target_flux_err", "Filter"])
    if df.empty:
        return empty_out, {"survey": "BlackGEM", "source_id": int(source_id), "observations": 0, "status": "no_data"}

    # 3. Data Cleaning Pipeline
    df = df.dropna(subset=["MJD_OBS", "FNU_OPT", "FNUERRTOT_OPT", "FILTER", "FILENAME"])
    df = df[~df["FILENAME"].str.contains("blackgem-ref", na=False)]
    
    if qc_keep is not None:
        df = df[df["QC_FLAG"].isin(qc_keep)]

    df = df[df["FNUERRTOT_OPT"] > 0]

    if clean:
        df = df[(df["FLAGS_MASK"] == 0) & (df["FLAGS_OPT"] == 0)]
    
    if df.empty:
        return empty_out, {"survey": "BlackGEM", "source_id": int(source_id), "observations": 0, "status": "filtered_out"}
        
    df = df.reset_index(drop=True)

    # 4. Time Standardization: MJD UTC to BJD_TDB (Full Julian Date)
    mjd = df["MJD_OBS"].to_numpy(float)
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    t = Time(mjd, format="mjd", scale="utc", location=LA_SILLA)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    # 5. Build standardized output DataFrame
    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     df["FNU_OPT"].to_numpy(float),
        "Target_flux_err": df["FNUERRTOT_OPT"].to_numpy(float),
        "Filter":          ("blackgem-" + df["FILTER"].astype(str).str.lower()).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)

    # 6. Return
    meta = {
        "survey": "BlackGEM", 
        "source_id": int(source_id),
        "observations": len(out), 
        "status": "success"
    }
    
    return out, meta