import warnings
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

# https://dace.unige.ch/

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

def _imports():                       # pycheops + dace_query are heavy/optional -> lazy
    from dace_query.cheops import Cheops
    from pycheops import Dataset
    return Cheops, Dataset

def fetch_cheops_lc(source_id, ra, dec, radius_arcsec=15.0, clean=True, max_visits=None):
    """
    CHEOPS targeted photometry for one source.

    Search:  dace_query.cheops.Cheops.query_region (coordinate cone) -> file_keys; each
             file_key is extracted with pycheops Dataset.get_lightcurve. CHEOPS is a POINTED
             mission, so almost every all-sky source returns no_data - only approved CHEOPS
             targets have data.
    time   : BJD_TDB  (pycheops time + bjd_ref; already barycentric -> no LTT applied).
    flux   : pycheops returns flux NORMALISED to median=1 (relative, dimensionless); there is
             NO CHEOPS AB zero point. To fit the uJy schema we ANCHOR the relative flux to the
             target's Gaia-G magnitude:  Target_flux = norm_flux * 10**((23.9 - Gmag)/2.5).
             The level then sits near the star's Gaia-G flux with the (precise) relative
             variability scaled onto it. CHEOPS' white band != Gaia G, so the ABSOLUTE level is
             approximate - trust CHEOPS for variability SHAPE/TIMING, not absolute photometry.
    filter : "cheops-broad"  (its own band - never merge with anything else).
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "CHEOPS", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "CHEOPS", "source_id": sid, "observations": len(df), "status": "success"})

    try:
        Cheops, Dataset = _imports()
    except Exception as e:
        return result(f"import_failed:{type(e).__name__}")

    coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = Cheops.query_region(coord, Angle(radius_arcsec, "arcsec"), output_format="pandas")
    except Exception as e:
        return result(f"search_failed:{type(e).__name__}")
    if res is None or len(res) == 0:
        return result("no_data")
    fcol = next((c for c in res.columns if c.lower() == "file_key"), None)
    if fcol is None:
        return result("no_file_key")
    file_keys = list(dict.fromkeys(res[fcol].astype(str).tolist()))     # unique, order-preserving
    if max_visits:
        file_keys = file_keys[:max_visits]

    frames = []
    for fk in file_keys:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                d = Dataset(fk, download_all=False, verbose=False, view_report_on_download=False)
                t, f, fe = d.get_lightcurve(aperture="DEFAULT", decontaminate=True)
                if clean:
                    t, f, fe = d.clip_outliers()
        except Exception:
            continue
        if t is None or len(t) == 0:
            continue
        gmag = getattr(d, "gmag", None)
        if gmag is None or not np.isfinite(gmag):
            continue                                   # no anchor (Gmag is set for all CHEOPS targets)
        anchor = 10 ** ((23.9 - float(gmag)) / 2.5)    # uJy per unit relative flux
        frames.append(pd.DataFrame({
            "BJD":             np.asarray(t, float) + d.bjd_ref,        # already BJD_TDB
            "Target_flux":     np.asarray(f, float) * anchor,
            "Target_flux_err": np.asarray(fe, float) * anchor,
            "Filter":          "cheops-broad",
        }))
    if not frames:
        return result("no_data")
    out = pd.concat(frames, ignore_index=True).sort_values("BJD").reset_index(drop=True)
    return result(None, out)