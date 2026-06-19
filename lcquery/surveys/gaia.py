import numpy as np
import pandas as pd
from astropy.table import Table as ATable
from astropy.time import Time
from astroquery.gaia import Gaia

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

_GAIA_BJD_TCB_REF = 2455197.5           

_ZP_AB = {"G": 25.8010446445, "BP": 25.3539555559, "RP": 25.1039837393}
_C = {b: 10 ** ((23.9 - zp) / 2.5) for b, zp in _ZP_AB.items()} 

_GAIA_WIDE = {
    "G":  {"time": ["g_transit_time", "g_obs_time"],   "flux": ["g_transit_flux", "g_flux"],
           "ferr": ["g_transit_flux_error", "g_flux_error"], "rej": ["variability_flag_g_reject"]},
    "BP": {"time": ["bp_obs_time", "bp_transit_time"], "flux": ["bp_flux", "bp_transit_flux"],
           "ferr": ["bp_flux_error", "bp_transit_flux_error"], "rej": ["variability_flag_bp_reject"]},
    "RP": {"time": ["rp_obs_time", "rp_transit_time"], "flux": ["rp_flux", "rp_transit_flux"],
           "ferr": ["rp_flux_error", "rp_transit_flux_error"], "rej": ["variability_flag_rp_reject"]},
}

_LOGGED_IN = False
def _ensure_login(auth):
    """Optional: log in only if creds are given (not required for most epoch-photometry retrieval)."""
    global _LOGGED_IN
    if auth and not _LOGGED_IN:
        try:
            Gaia.login(user=auth[0], password=auth[1]); _LOGGED_IN = True
        except Exception:
            pass

def _tables(obj):
    if isinstance(obj, ATable):
        yield obj; return
    if hasattr(obj, "to_table"):
        try: yield obj.to_table(); return
        except Exception: pass
    if hasattr(obj, "tables"):
        for t in getattr(obj, "tables", []):
            if hasattr(t, "to_table"): yield t.to_table()
        return
    if hasattr(obj, "resources"):
        for r in getattr(obj, "resources", []):
            for t in getattr(r, "tables", []):
                if hasattr(t, "to_table"): yield t.to_table()

def _pick(df, names):
    for n in names:
        if n in df.columns: return n
    return None

def _rejected(s):
    """Robust truthy test for the variability reject flag (handles bool, str, or bytes)."""
    def t(x):
        if isinstance(x, (bytes, bytearray)): x = x.decode()
        return str(x).strip().lower() in ("true", "t", "1", "1.0")
    return s.map(t)

def _wide_to_long(df, clean):
    """Melt Gaia's wide per-transit epoch-photometry table into long (band, time, flux, ferr)."""
    parts = []
    for band, cand in _GAIA_WIDE.items():
        tcol, fcol = _pick(df, cand["time"]), _pick(df, cand["flux"])
        if not tcol or not fcol:
            continue
        ecol, rcol = _pick(df, cand["ferr"]), _pick(df, cand["rej"])
        sub = pd.DataFrame({
            "time": pd.to_numeric(df[tcol], errors="coerce"),
            "flux": pd.to_numeric(df[fcol], errors="coerce"),
            "ferr": pd.to_numeric(df[ecol], errors="coerce") if ecol else np.nan,
            "band": band,
        })
        if clean and rcol:                                   
            sub = sub[~_rejected(df[rcol]).to_numpy()]
        parts.append(sub.dropna(subset=["time", "flux"]))    
    return pd.concat(parts, ignore_index=True) if parts else None

def fetch_gaia_lc(source_id, ra=None, dec=None, bands=None, valid_only=True, clean=True, auth=None):
    """
    --------------------------------------------------------------------------------
    Gaia DR3  (epoch photometry)
    --------------------------------------------------------------------------------
    Access & format. Gaia DR3 epoch photometry via the DataLink service
    (Gaia.load_data, retrieval_type="EPOCH_PHOTOMETRY", data_structure="INDIVIDUAL",
    format="votable"), keyed on source_id (ra/dec unused). The product is a wide
    per-transit table: one row per transit, with separate per-band columns --
    g_transit_time/g_transit_flux/g_transit_flux_error,
    bp_obs_time/bp_flux/bp_flux_error, rp_obs_time/rp_flux/rp_flux_error -- plus
    per-band variability_flag_*_reject flags. The code melts this to long
    (band, time, flux, ferr), using each band's own time/flux columns. Filter
    labels: gaia-g, gaia-bp, gaia-rp. Photometry is per-transit (every source is
    measured at each predicted transit, down to G ~ 21), so it is effectively forced
    for completeness, though individual transits can be rejected.

    Time -> BJD_TDB. Each band's time column is the barycentric transit time on the
    TCB scale, stored as an offset: (BJD_TCB - 2455197.5) days, where
    2455197.5 = JD(TCB) at 2010-01-01T00:00:00. The code reconstructs the full
    BJD_TCB and converts the scale TCB -> TDB; because the time is already
    barycentric, no light-travel correction is applied:

        bjd_tcb = 2455197.5 + time
        BJD_TDB = Time(bjd_tcb, format="jd", scale="tcb").tdb.jd

    The TCB -> TDB step (~20 s at the Gaia epoch) is applied for consistency with the
    TDB times of the other surveys. The three bands' times genuinely differ by ~30 s
    within a transit (the source crosses the focal plane), and each is carried
    through its own band.

    Flux -> uJy. The per-band flux is the instrumental flux in e-/s, converted to uJy
    on the AB system via the Gaia EDR3/DR3 AB zero-points (Riello et al. 2021):
    mag_AB = ZP - 2.5*log10(flux[e-/s]) with
    ZP = {G: 25.8010446445, BP: 25.3539555559, RP: 25.1039837393}. These are the AB
    zero points, NOT the VEGAMAG ones (the VEGAMAG EDR3 values would be G = 25.6874,
    BP = 25.3385, RP = 24.7479) -- the labelling is correct. Equivalently
    Target_flux[uJy] = flux[e-/s] * C_band and Target_flux_err = flux_error * C_band,
    where C_band = 10^((23.9 - ZP)/2.5) uJy per e-/s. System "AB", F0 = 3631 Jy.

    Cleaning. valid_only=True (sent to the archive as valid_data=True) returns only
    rows with non-null flux and rejected_by_photometry = False, removing
    photometry-rejected and null-flux transits before download. With clean=True, the
    melt additionally drops, per band, transits flagged
    variability_flag_{g,bp,rp}_reject = True (rejected by DPAC variability processing
    or carrying negative/unphysical flux), then requires finite, positive flux
    (flux > 0). Per-band nulls (a transit measured in one band but not another) are
    dropped per band during the melt. Since rejected_by_photometry and
    rejected_by_variability are independent processes, filtering both removes all
    flagged points.
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "Gaia", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "Gaia", "source_id": sid, "observations": len(df), "status": "success"})

    _ensure_login(auth)
    try:
        out = Gaia.load_data(ids=[sid], data_release="Gaia DR3", retrieval_type="EPOCH_PHOTOMETRY",
                             valid_data=bool(valid_only), band=bands, data_structure="INDIVIDUAL",
                             format="votable", verbose=False)
    except Exception as e:
        return result(f"error:{type(e).__name__}")
    if not out:
        return result("no_data")

    long_parts, saw_table = [], False
    for fname, objs in out.items():
        for obj in (objs if isinstance(objs, (list, tuple)) else [objs]):
            for tab in _tables(obj):
                df = tab.to_pandas()
                df.columns = [str(c).strip().lower() for c in df.columns]
                if not any(_pick(df, _GAIA_WIDE[b]["flux"]) for b in _GAIA_WIDE):
                    continue                                
                saw_table = True
                lp = _wide_to_long(df, clean)
                if lp is not None and len(lp):
                    long_parts.append(lp)
    if not saw_table:
        return result("no_table")
    if not long_parts:
        return result("filtered_out")

    d = pd.concat(long_parts, ignore_index=True)
    if bands:                                             
        want = {b.upper() for b in ([bands] if isinstance(bands, str) else bands)}
        d = d[d["band"].isin(want)]
    if clean:
        d = d[np.isfinite(d["flux"]) & (d["flux"] > 0)]
    if len(d) == 0:
        return result("filtered_out")

    bjd_tcb = _GAIA_BJD_TCB_REF + d["time"].to_numpy(float)
    bjd_tdb = Time(bjd_tcb, format="jd", scale="tcb").tdb.jd
    c = d["band"].map(_C).to_numpy(float)
    out_df = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     d["flux"].to_numpy(float) * c,
        "Target_flux_err": d["ferr"].to_numpy(float) * c,
        "Filter":          ("gaia-" + d["band"].astype(str).str.lower()).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out_df)