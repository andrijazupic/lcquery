import os
import inspect
import datetime
import importlib
import pandas as pd
import yaml

# survey -> (module under lcquery.surveys, fetcher name). Loaded lazily so a missing
# optional dependency disables only that survey instead of breaking `import lcquery`.
_FETCHERS = {
    "BlackGEM": ("blackgem", "fetch_blackgem_lc"), "TESS": ("tess", "fetch_tess_lc"),
    "K2": ("k2", "fetch_k2_lc"), "ZTF": ("ztf", "fetch_ztf_lc"),
    "CRTS": ("crts", "fetch_crts_lc"), "NSC": ("nsc", "fetch_nsc_lc"),
    "ATLAS": ("atlas", "fetch_atlas_lc"), "SkyMapper": ("skymapper", "fetch_skymapper_lc"),
    "J-VAR": ("jvar", "fetch_jvar_lc"), "ASAS-SN": ("asassn", "fetch_asassn_lc"),
    "OGLE": ("ogle", "fetch_ogle_lc"), "Gaia": ("gaia", "fetch_gaia_lc"),
    "CHEOPS": ("cheops", "fetch_cheops_lc"),
}

DEFAULT_CONFIG = {
    "output": {"base_dir": "lightcurves", "overwrite": False, "write_header_comments": False},
    "defaults": {"clean": True},
    "credentials": {"atlas_token": "", "gaia_user": "", "gaia_password": ""},
    "surveys": {name: {"enabled": True} for name in _FETCHERS},
}


def _get_fetcher(name):
    mod, func = _FETCHERS[name]
    return getattr(importlib.import_module(f"{__package__}.surveys.{mod}"), func)


def load_config(config=None):
    """Resolution order: dict/path -> ./config.yaml -> <repo>/config.yaml
    -> <repo>/config.example.yaml -> built-in DEFAULT_CONFIG."""
    if isinstance(config, dict):
        return config
    if config is None:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        for cand in ("config.yaml",                                    
                     os.path.join(repo_root, "config.yaml"),           
                     os.path.join(repo_root, "config.example.yaml")):  # committed template for clones
            if os.path.exists(cand):
                config = cand
                break
    if config is None:
        return DEFAULT_CONFIG
    with open(config) as f:
        return yaml.safe_load(f) or DEFAULT_CONFIG


def _accepted(func, kwargs):
    sig = inspect.signature(func)
    if any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    names = set(sig.parameters)
    return {k: v for k, v in kwargs.items() if k in names and k not in ("source_id", "ra", "dec")}


def _credential_kwargs(survey, creds):
    if survey == "ATLAS":
        tok = creds.get("atlas_token") or os.environ.get("ATLASFORCED_SECRET_KEY")
        return {"token": tok} if tok else {}
    if survey == "Gaia":
        u = creds.get("gaia_user") or os.environ.get("GAIA_USER")
        p = creds.get("gaia_password") or os.environ.get("GAIA_PASSWORD")
        return {"auth": [u, p]} if (u and p) else {}
    return {}


def _save_csv(df, path, header_lines=None):
    if header_lines:
        with open(path, "w") as f:
            for line in header_lines:
                f.write(f"# {line}\n")
        df.to_csv(path, mode="a", index=False)
    else:
        df.to_csv(path, index=False)


def update_master_metadata(meta_dict, meta_file):
    df_new = pd.DataFrame([meta_dict])
    if os.path.exists(meta_file):
        df = pd.read_csv(meta_file)
        df["source_id"] = df["source_id"].astype("int64")     # keep Gaia IDs exact, never float
        mask = ((df["source_id"] == int(meta_dict["source_id"])) & (df["survey"] == meta_dict["survey"]))
        df_combined = pd.concat([df[~mask], df_new], ignore_index=True)
    else:
        df_combined = df_new
    df_combined.to_csv(meta_file, index=False)


def get_all_lightcurves(source_id, ra, dec, config=None, survey_list=None,
                        overwrite=None, base_dir=None, verbose=True):
    cfg = load_config(config)
    out_cfg  = cfg.get("output", {}) or {}
    defaults = cfg.get("defaults", {}) or {}
    surveys  = cfg.get("surveys", {}) or {}
    creds    = cfg.get("credentials", {}) or {}
    base_dir  = base_dir  if base_dir  is not None else out_cfg.get("base_dir", "lightcurves")
    overwrite = overwrite if overwrite is not None else out_cfg.get("overwrite", False)
    write_hdr = out_cfg.get("write_header_comments", False)
    meta_file = os.path.join(base_dir, "query_metadata.csv")
    os.makedirs(base_dir, exist_ok=True)

    if verbose:
        print(f"\n--- Processing Source ID: {source_id} ---")

    for survey_name in _FETCHERS:
        scfg = surveys.get(survey_name, {}) or {}
        enabled = (survey_name in survey_list) if survey_list is not None else scfg.get("enabled", True)
        if not enabled:
            continue
        try:
            fetch_func = _get_fetcher(survey_name)
        except Exception as e:
            if verbose: print(f"[{survey_name}] unavailable ({type(e).__name__}) - install its dependency to enable. Skipping.")
            continue

        kwargs = {**defaults, **{k: v for k, v in scfg.items() if k != "enabled"}}
        kwargs.update(_credential_kwargs(survey_name, creds))
        kwargs = _accepted(fetch_func, kwargs)

        survey_dir = os.path.join(base_dir, survey_name)
        os.makedirs(survey_dir, exist_ok=True)
        file_path = os.path.join(survey_dir, f"{source_id}.csv")

        if os.path.exists(file_path) and not overwrite:
            if verbose: print(f"[{survey_name}] Cached file found. Skipping download.")
            n = len(pd.read_csv(file_path, comment="#"))
            status = "cached"
        else:
            if verbose: print(f"[{survey_name}] Querying database...")
            try:
                df, survey_meta = fetch_func(source_id, ra, dec, **kwargs)
            except Exception as e:
                if verbose: print(f"[{survey_name}] Fetch raised: {type(e).__name__}: {e}")
                df, survey_meta = pd.DataFrame(), {"observations": 0, "status": f"exception:{type(e).__name__}"}
            n = survey_meta.get("observations", 0)
            status = survey_meta.get("status", "unknown")
            if not df.empty:
                hdr = None
                if write_hdr:
                    flux_sys = "Vega" if survey_name in ("OGLE", "CRTS") else "AB"
                    hdr = [f"survey={survey_name}", f"source_id={source_id}", f"ra={ra} dec={dec}",
                           f"params={kwargs}",
                           f"columns: BJD=BJD_TDB(JD), Target_flux=uJy({flux_sys}), Target_flux_err=uJy({flux_sys}), Filter=band"]
                _save_csv(df, file_path, hdr)
                if verbose: print(f"[{survey_name}] Saved {n} observations to {file_path}")
            elif verbose:
                print(f"[{survey_name}] No valid data found ({status}).")

        radius = kwargs.get("radius_arcsec", kwargs.get("radius_arcmin"))
        runit = "arcsec" if "radius_arcsec" in kwargs else ("arcmin" if "radius_arcmin" in kwargs else "")
        update_master_metadata({
            "source_id": int(source_id), "ra": ra, "dec": dec, "survey": survey_name,
            "observations": n, "status": status, "clean": kwargs.get("clean"),
            "radius": radius, "radius_unit": runit, "time_unit": "BJD_TDB", 
            "flux_unit": "uJy_Vega" if survey_name in ("OGLE", "CRTS") else "uJy_AB",
            "queried_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        }, meta_file)

    if verbose:
        print(f"--- Done with {source_id} ---")