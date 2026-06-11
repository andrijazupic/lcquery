import numpy as np
import pandas as pd
from astropy.table import Table as ATable
from astropy.time import Time
from astroquery.gaia import Gaia

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

# Gaia DR3 epoch-photometry times are Barycentric JD in TCB, offset by this reference.
_GAIA_BJD_TCB_REF = 2455197.5            # JD(TCB) at 2010-01-01T00:00:00

# Gaia EDR3/DR3 AB zero points (Table 5.2 / Riello+2021): mag_AB = ZP - 2.5*log10(flux[e/s]).
_ZP_AB = {"G": 25.8010446445, "BP": 25.3539555559, "RP": 25.1039837393}
_C = {b: 10 ** ((23.9 - zp) / 2.5) for b, zp in _ZP_AB.items()}   # uJy per e-/s

# The DataLink EPOCH_PHOTOMETRY product is WIDE: one row per transit, separate columns per band
# (and G's time column is named differently from BP/RP). Candidate names cover version drift.
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
        if clean and rcol:                                    # drop variability-rejected transits
            sub = sub[~_rejected(df[rcol]).to_numpy()]
        parts.append(sub.dropna(subset=["time", "flux"]))     # per-band gaps -> drop nulls
    return pd.concat(parts, ignore_index=True) if parts else None

def fetch_gaia_lc(source_id, ra=None, dec=None, bands=None, valid_only=True, clean=True, auth=None):
    """
    Gaia DR3 epoch photometry for one source (keyed on source_id; ra/dec unused).
    time   : BJD_TDB  (per-band Gaia time is BJD in TCB -> reconstruct, convert scale TCB->TDB;
                       already barycentric, so NO light-travel correction is applied).
    flux   : uJy on the AB scale  (e-/s * AB zero point; AB_mag = 23.9 - 2.5*log10(Target_flux)).
    filter : "gaia-g"/"gaia-bp"/"gaia-rp".
    Statuses: no_data (empty product), no_table (no epoch-photometry table), filtered_out, success.
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
                    continue                                  # not the epoch-photometry table
                saw_table = True
                lp = _wide_to_long(df, clean)
                if lp is not None and len(lp):
                    long_parts.append(lp)
    if not saw_table:
        return result("no_table")
    if not long_parts:
        return result("filtered_out")

    d = pd.concat(long_parts, ignore_index=True)
    if bands:                                                 # optional band subset
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