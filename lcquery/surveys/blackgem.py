import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from google.cloud import bigquery

BG_PROJECT = "blackgem-databases-access"
BG_DB = "blackgem-full-source-db.blackgem_fullsource_v2"

LA_SILLA = EarthLocation.from_geodetic(lon=-70.7345*u.deg, lat=-29.2584*u.deg, height=2400.0*u.m)

_CLIENT = None
def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = bigquery.Client(project=BG_PROJECT)
    return _CLIENT

def fetch_blackgem_lc(source_id, ra, dec, 
                      client=None, db=BG_DB, qc_keep=("green", "yellow"), clean=True):
    """
    --------------------------------------------------------------------------------
    BlackGEM  (BlackGEM array, full-source database)
    --------------------------------------------------------------------------------
    Access & cross-match. Proprietary BigQuery
    (blackgem-full-source-db.blackgem_fullsource_v2), joining detections to images on
    IMAGE_ID, filtered by SOURCE_ID -- which is the Gaia source ID (per schema), so
    the query is keyed on the Gaia DR3 source_id directly, not on position. The data
    are FORCED optimal photometry (Horne 1986 / Naylor 1998): on each reduced frame
    the BlackBOX/ZOGY pipeline makes a forced optimal-photometry measurement at the
    position of every Gaia DR3 object in the field of view (proper motion applied),
    and the result is stored in the full-source catalogue (Groot et al. 2024). Being
    forced, the light curve is eclipse-complete -- a source keeps a measurement at
    every epoch the field was observed, even in deep eclipse (subject to the QC cuts
    below). Three unit telescopes at ESO La Silla, ~60 s exposures (1-minute fast
    synoptic cadence), optimised Sloan set BG-u/g/r/i/z plus a wide BG-q (440-720 nm)
    filter; per-image zero-point uncertainty ~1%. Filters map as
    "blackgem-" + FILTER.lower() -> blackgem-u/g/q/r/i/z.

    Time -> BJD_TDB. The images.MJD_OBS column is the exposure-midpoint MJD on the
    UTC scale (derived from DATE_OBS, defined as the GPS-timed shutter open/close
    average (GPSSTART+GPSEND)/2 at the image centre). Because it is already the
    midpoint, no half-exposure offset is applied. The midpoint UTC is treated as
    topocentric at La Silla, scale-converted to TDB, and barycentre-corrected at the
    exact source position:

        t       = Time(MJD_OBS, format="mjd", scale="utc", location=LA_SILLA)
        BJD_TDB = (t.tdb + t.light_travel_time(source_coord, kind="barycentric")).jd

    This is re-derived at the source rather than reusing the database's BJD_OBS
    (computed at the image-centre coordinates), making the barycentric term correct
    to the actual target by up to a few seconds. The DB's BJD_OBS is in modified form
    (BJD - 2400000.5), while the pipeline output is full BJD_TDB.

    Flux -> uJy. BlackGEM supplies the optimal-photometry flux density FNU_OPT
    already in uJy on the AB system (schema unit [uJy]; AB tie confirmed by MAG_OPT
    being the optimal AB magnitude). It is taken directly with no conversion:
    Target_flux = FNU_OPT. The error uses FNUERRTOT_OPT -- the total optimal-flux
    error in uJy that includes the photometric-calibration uncertainty (rather than
    the bare FNUERR_OPT), giving a conservative, fully-propagated error bar. System
    "AB", F0 = 3631 Jy.

    Cleaning. Always applied: drop NaN in MJD_OBS, FNU_OPT, FNUERRTOT_OPT, FILTER,
    FILENAME; remove reference-image detections (FILENAME containing "blackgem-ref")
    so only individual-epoch science measurements remain; keep QC_FLAG in
    {green, yellow} (dropping orange and red from the four-level
    green|yellow|orange|red ladder -- a conservative cut that also excludes
    moderate-issue orange); require FNUERRTOT_OPT > 0. With clean=True, additionally
    require FLAGS_MASK == 0 (no bad/masked pixels OR-combined over the source's
    ISO/inner-profile region) and FLAGS_OPT == 0 (no optimal-photometry flags, i.e.
    nominal background and fit). The "blackgem-ref" bucket is the reference image,
    versus the "blackgem-red" science bucket. NOTE: because the data are forced, the
    RAW light curve covers deep eclipses; the QC cuts can still drop the very
    faintest in-eclipse epochs (low SNR), so loosen them if chasing the deepest
    minima.
    """

    if client is None:
        client = _get_client()

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

    mjd = df["MJD_OBS"].to_numpy(float)
    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    t = Time(mjd, format="mjd", scale="utc", location=LA_SILLA)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     df["FNU_OPT"].to_numpy(float),
        "Target_flux_err": df["FNUERRTOT_OPT"].to_numpy(float),
        "Filter":          ("blackgem-" + df["FILTER"].astype(str).str.lower()).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)

    meta = {
        "survey": "BlackGEM", 
        "source_id": int(source_id),
        "observations": len(out), 
        "status": "success"
    }
    
    return out, meta