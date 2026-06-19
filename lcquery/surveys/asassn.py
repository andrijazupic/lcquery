import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

CTIO = EarthLocation.from_geodetic(
    lon=-70.8065 * u.deg, lat=-30.1692 * u.deg, height=2207.0 * u.m)

_MJY_TO_UJY = 1.0e3
_BANDS = {"g": "asassn-g", "v": "asassn-v"}

_CLIENT = None
def _client():
    global _CLIENT
    if _CLIENT is None:
        from pyasassn.client import SkyPatrolClient
        _CLIENT = SkyPatrolClient(verbose=False)
    return _CLIENT


def fetch_asassn_lc(source_id, ra, dec, radius_arcsec=5.0, clean=True):
    """
    --------------------------------------------------------------------------------
    ASAS-SN  (All-Sky Automated Survey for Supernovae, Sky Patrol v2)
    --------------------------------------------------------------------------------
    Access. pyasassn.SkyPatrolClient cone search (default radius 5 arcsec,
    configurable); the nearest asas_sn_id to the input position is selected and its
    light-curve DataFrame is read. Columns used: jd, flux, flux_err, mag, mag_err,
    quality, phot_filter. ASAS-SN is a five-station, ~20-telescope network hosted by
    Las Cumbres Observatory: Brutus (Haleakala, Hawaii), Cassius and Paczynski
    (CTIO, Chile), Leavitt (McDonald, Texas) and Payne-Gaposchkin (SAAO Sutherland,
    South Africa). Each epoch combines (usually three) dithered ~90 s exposures.
    Photometry is FORCED aperture photometry (2-pixel radius) at the catalogue
    position, calibrated to ATLAS Refcat2 -- so the light curve is eclipse-complete.

    Time -> BJD_TDB. ASAS-SN serves HJD on the UTC scale (the jd column holds the
    heliocentric JD, computed at the FIELD CENTRE). The conversion treats the value
    as HJD(UTC) at a representative southern site (CTIO), subtracts the heliocentric
    light-travel time to recover geocentric/topocentric JD(UTC), converts the scale
    UTC -> TDB, and adds the barycentric light-travel time:

        t0      = Time(hjd, format="jd", scale="utc", location=CTIO)
        t_utc   = t0 - t0.light_travel_time(coord, kind="heliocentric")  # HJD -> JD(UTC)
        BJD_TDB = (t_utc.tdb + t_utc.light_travel_time(coord, kind="barycentric")).jd

    Although ASAS-SN is a multi-station network, a single site location is used: the
    choice enters only the topocentric term (<= ~21 ms), and the heliocentric round
    trip barely depends on the site. Both are swamped by ASAS-SN's own timing floor:
    the field-centre HJD is accurate only to ~200 s unless the light curve is
    recomputed with the Aperture Photometry pipeline (this limit is stated verbatim
    in the ASAS-SN documentation). For periods of hours-days the ~200 s floor is a
    fraction of a percent of a cycle; it matters only for very short periods or
    precise eclipse timing.

    Flux -> uJy. The flux/flux_err columns are in mJy and are scaled x1000 to uJy.
    No light-travel or other transformation is applied to flux -- it is a direct unit
    scaling. The photometric system differs by band, and BOTH zero points have been
    verified directly from ASAS-SN's own mag-flux relation (fitting mag versus
    -2.5*log10(flux_uJy) returns slope = 1.0000 with zero residual scatter, i.e. the
    magnitude is computed FROM the flux with a single fixed per-band zero point -- the
    relation is exact, not statistical):

    asassn-g : exact AB. F0 = 3631.0000 Jy (recovered to 13 significant figures),
                i.e. mag_g = 23.900 - 2.5*log10(flux_uJy). System "AB".

    asassn-v : Vega. F0 = 3836.3 Jy (measured; ZP_mag = 23.960), a constant
                +0.060 mag offset from the AB convention -- a Johnson-V-like scale.
                ASAS-SN's V calibration (APASS-tied) genuinely sits ~0.06 mag off a
                textbook Johnson V (whose Vega zero point ~3636 Jy is essentially
                the AB value), so 3836 Jy is NOT the standard V constant: it is the
                effective value that makes ASAS-SN's own flux and mag columns
                self-consistent. The offset is irrelevant for variability and is a
                small systematic only for absolute V photometry. Kept as a distinct
                band. (The fit also vindicates the mJy assumption: a wrong unit
                would have shifted ZP_mag by ~7.5 mag.)

    Cleaning. Applied in all cases (independent of clean): drop rows with NaN
    time/flux/flux_err; drop non-detections (mag_err > 99, sentinel 99.999); require
    flux_err > 0. With clean=True, additionally keep only good images
    (quality == 'G'), discarding bad ('B') and unknown (None). With clean=False, all
    quality classes are kept but non-detections are still removed. (A diagnostic
    confirmed the non-detection sentinels carry flux_err = 99.99 and repeated
    placeholder flux values, so the mag_err > 99 cut is the correct way to remove
    them -- a flux_err > 0 cut would NOT, since 99.99 > 0.)

    Data-quality caveats (not removed by clean): bright sources saturate (roughly
    V <~ 10, g <~ 11) -- watch for bright outliers; and the ~200 s field-centre
    timing floor noted above.
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "ASAS-SN", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "ASAS-SN", "source_id": sid,
                     "observations": len(df), "status": "success"})

    client = _client()
    rdeg = radius_arcsec / 3600.0

    try:
        cat = client.cone_search(ra_deg=float(ra), dec_deg=float(dec),
                                 radius=rdeg, catalog="master_list", download=False)
    except Exception as e:
        return result(f"search_failed: {type(e).__name__}: {e}")
    if cat is None or len(cat) == 0 or "asas_sn_id" not in cat.columns:
        return result("no_match")
    if len(cat) > 1 and {"ra_deg", "dec_deg"}.issubset(cat.columns):
        sep2 = ((cat["ra_deg"] - float(ra)) * np.cos(np.radians(float(dec)))) ** 2 \
               + (cat["dec_deg"] - float(dec)) ** 2
        nearest = int(cat.loc[sep2.idxmin(), "asas_sn_id"])
    else:
        nearest = int(cat["asas_sn_id"].iloc[0])

    try:
        lcs = client.cone_search(ra_deg=float(ra), dec_deg=float(dec),
                                 radius=rdeg, catalog="master_list", download=True)
    except Exception as e:
        return result(f"download_failed: {type(e).__name__}: {e}")
    try:
        df = lcs[nearest].data
    except Exception:
        try:                                          
            ids = list(lcs.stats().index)
            df = lcs[ids[0]].data if ids else None
        except Exception:
            df = None
    if df is None or len(df) == 0:
        return result("no_data")

    tcol = next((c for c in ["jd", "hjd", "HJD"] if c in df.columns), None)
    if tcol is None or "flux" not in df.columns or "flux_err" not in df.columns:
        return result("missing_columns")

    df = df.copy()
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")
    df["flux_err"] = pd.to_numeric(df["flux_err"], errors="coerce")
    df = df.dropna(subset=[tcol, "flux", "flux_err"])
    if "mag_err" in df.columns:                       
        df = df[pd.to_numeric(df["mag_err"], errors="coerce") < 99]
    df = df[df["flux_err"] > 0]
    if clean and "quality" in df.columns:             
        df = df[df["quality"].astype(str) == "G"]
    if len(df) == 0:
        return result("filtered_out")
    df = df.reset_index(drop=True)

    fcol = next((c for c in ["phot_filter", "filter", "Filter"] if c in df.columns), None)
    if fcol is not None:
        band = (df[fcol].astype(str).str.lower().map(_BANDS)
                .fillna("asassn-" + df[fcol].astype(str))).to_numpy()
    else:
        band = np.array(["asassn"] * len(df))       

    coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    hjd = df[tcol].to_numpy(float)
    t0 = Time(hjd, format="jd", scale="utc", location=CTIO)
    t_utc = t0 - t0.light_travel_time(coord, kind="heliocentric")
    bjd_tdb = (t_utc.tdb + t_utc.light_travel_time(coord, kind="barycentric")).jd

    flux = df["flux"].to_numpy(float) * _MJY_TO_UJY
    flux_err = df["flux_err"].to_numpy(float) * _MJY_TO_UJY

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          band,
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)