import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
from ztfquery import lightcurve

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

PALOMAR = EarthLocation.from_geodetic(
    lon=-116.8650 * u.deg, lat=33.3563 * u.deg, height=1712.0 * u.m)

AB_ZP_UJY = 23.9                                  
_ZTF_BANDS = {"zg": "ztf-g", "zr": "ztf-r", "zi": "ztf-i"}
_REQUIRED = {"mjd", "mag", "magerr", "filtercode"}


def fetch_ztf_lc(source_id, ra, dec, radius_arcsec=1.5, bad_catflags_mask=32768):
    """
    --------------------------------------------------------------------------------
    ZTF  (Zwicky Transient Facility, IRSA DR light curves)
    --------------------------------------------------------------------------------
    Access & cross-match. PSF-photometry light curves from the IRSA ZTF lightcurve
    database via ztfquery (LCQuery.from_position, stored-login flow). The
    BAD_CATFLAGS_MASK filter is applied server-side. Per-epoch rows are keyed by oid
    (ZTF assigns a separate oid per field/CCD-quadrant/filter, so one physical source
    can have several light curves per band -- all are retained and merged), restricted
    to zg/zr/zi -> ztf-g/r/i. These IRSA matchfile light curves are DETECTION-based (a
    source must be extracted above the per-image threshold to appear), so deep
    eclipses can drop epochs; ZTF's separate forced-photometry service (more
    eclipse-complete) is not used here. Palomar 48-inch (P48); 30 s public-survey
    exposures.

    Time -> BJD_TDB. The IRSA mjd column is the topocentric MJD (UTC) of the exposure
    START -- confirmed verbatim by the ZTF DR documentation: "the mjd timestamps
    attached to the lightcurve measurements pertain to the start of each exposure
    while hjd and hmjd pertain to the middle, i.e., with 0.5*EXPTIME added". (The
    separate hjd column is therefore the heliocentric mid-exposure time.) The pipeline
    forms the mid-exposure MJD as mjd + 0.5*exptime/86400 (per-epoch exptime, 30 s for
    the public survey), treats it as topocentric UTC at Palomar, and converts:

        t       = Time(mjd + 0.5*exptime/86400, format="mjd", scale="utc", location=PALOMAR)
        BJD_TDB = (t.tdb + t.light_travel_time(source_coord, kind="barycentric")).jd

    Because mjd is topocentric and the START of the exposure (not heliocentric), the
    barycentric correction is applied once (not double-counted) and the half-exposure
    step (15 s at 30 s) is required to reach mid-exposure, bringing ZTF in line with
    the rest of the pipeline. (Equivalently, ZTF's own hjd mid time could be used via
    a heliocentric -> barycentric round trip; the exptime route is chosen for
    consistency.)

    Flux -> uJy. ZTF mag is the calibrated PSF magnitude on the AB system
    (Pan-STARRS1-calibrated, for a g-r = 0 source). Converted via
    Target_flux[uJy] = 10^((23.9 - mag)/2.5) (ZP 23.9 = AB->uJy zero point) and
    Target_flux_err = Target_flux * magerr / 1.0857. System "AB", F0 = 3631 Jy. The
    per-epoch colour coefficient clrcoeff is intentionally NOT applied: correcting it
    needs the simultaneous g-r colour, and it amounts to a near-constant per-band
    offset that does not affect period detection. g/r/i all share the AB scale of the
    pipeline's other optical surveys.

    Cleaning. The IRSA query filters server-side on BAD_CATFLAGS_MASK = 32768,
    removing epochs with catflags bit 15 -- the "cloud-affected and/or
    moon-contamination" flag -- ZTF's recommended baseline for usable photometry
    (exposed as a parameter; tightening to catflags = 0 or mask 65535 also removes
    source-level flags, at the cost of fewer epochs). Post-query: dropna on
    time/mag/magerr/filter/exptime, restriction to zg/zr/zi, and magerr > 0.
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

    df = df.dropna(subset=["mjd", "mag", "magerr", "filtercode", "exptime"])
    df = df[df["filtercode"].isin(_ZTF_BANDS)]
    df = df[df["magerr"] > 0]                        
    if len(df) == 0:
        return result("filtered_out")
    df = df.reset_index(drop=True)

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    mjd_mid = df["mjd"].to_numpy(float) + 0.5 * df["exptime"].to_numpy(float) / 86400.0
    t = Time(mjd_mid, format="mjd", scale="utc", location=PALOMAR)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    mag = df["mag"].to_numpy(float)
    magerr = df["magerr"].to_numpy(float)
    flux = 10 ** ((AB_ZP_UJY - mag) / 2.5)
    flux_err = flux * magerr / 1.0857           

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          df["filtercode"].map(_ZTF_BANDS).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)