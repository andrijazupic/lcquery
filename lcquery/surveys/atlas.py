import io
import os
import re
import time as _time
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time


_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]
BASEURL = "https://fallingstar-data.com/forcedphot"
_HOST = "https://fallingstar-data.com"

ATLAS_SITE = EarthLocation.from_geodetic(
    lon=-156.2568 * u.deg, lat=20.7067 * u.deg, height=3055.0 * u.m)

ATLAS_FILTERS = {"c": "atlas-c", "o": "atlas-o"}
_CLEAN_COLS = ["x", "y", "maj", "min", "apfit", "mag5sig", "Sky"]


def _make_session():
    """Session that auto-retries dropped connections / 5xx on GET & DELETE.
    POST is NOT retried (default allowed_methods excludes it) to avoid
    submitting duplicate jobs if a queue request's connection drops."""
    s = requests.Session()
    retry = Retry(total=5, connect=5, read=5, backoff_factor=0.5,
                  status_forcelist=[500, 502, 503, 504], raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def _get_headers(token=None):
    tok = token or os.environ.get("ATLASFORCED_SECRET_KEY")
    if not tok:
        raise RuntimeError(
            "ATLAS token not found. Set ATLASFORCED_SECRET_KEY (or pass token=...). "
            f"Get it once from {BASEURL}/api-token-auth/ with your username/password.")
    return {"Authorization": f"Token {tok}", "Accept": "application/json"}


def _abs(u):
    if not u:
        return u
    u = str(u)
    return u if u.startswith("http") else urljoin(_HOST + "/", u)


def _taskid(task_url):
    m = re.search(r"/(\d+)/?$", task_url or "")
    return m.group(1) if m else None


def _queue_job(session, ra, dec, headers, mjd_min, use_reduced, max_retries=6):
    data = {"ra": float(ra), "dec": float(dec),
            "send_email": False, "use_reduced": use_reduced}
    if mjd_min is not None:
        data["mjd_min"] = float(mjd_min)
    tries = 0
    while True:
        try:
            resp = session.post(f"{BASEURL}/queue/", headers=headers, data=data, timeout=30)
        except Exception:
            return None                    
        if resp.status_code == 201:
            return _abs(resp.json().get("url"))
        if resp.status_code == 429:
            msg = resp.json().get("detail", "")
            t_sec = re.findall(r"available in (\d+) seconds", msg)
            t_min = re.findall(r"available in (\d+) minutes", msg)
            wait = int(t_sec[0]) if t_sec else (int(t_min[0]) * 60 if t_min else 10)
            _time.sleep(min(wait, 120))
            tries += 1
            if tries > max_retries:
                return None
            continue
        return None


def _poll_result(session, task_url, headers, max_wait_s, poll_every_s):
    start = _time.time()
    fails = 0
    while True:
        try:
            resp = session.get(task_url, headers=headers, timeout=30)
            if resp.status_code != 200:
                fails += 1
                if fails > 8:
                    return "error", None
                _time.sleep(5)
                continue
            data = resp.json()
        except Exception:                  
            fails += 1
            if fails > 8:
                return "error", None
            _time.sleep(5)
            continue
        if data.get("finishtimestamp"):
            ru = data.get("result_url")
            return ("done", ru) if ru else ("nodata", None)
        if _time.time() - start > max_wait_s:
            return "timeout", None
        _time.sleep(poll_every_s if data.get("starttimestamp") else max(2 * poll_every_s, 4))


def _fetch_result(session, result_url, taskid, headers):
    urls = []
    if result_url:
        urls.append(_abs(result_url))
    if taskid:
        urls.append(f"{BASEURL}/static/results/job{taskid}.txt")
    for u in urls:
        try:
            r = session.get(u, headers=headers, timeout=60)
            if r.status_code == 200 and r.text.strip() and "###MJD" in r.text:
                return r.text
        except Exception:
            continue
    return None


def fetch_atlas_lc(source_id, ra, dec, token=None, 
                   mjd_min=57000.0, use_reduced=True,
                   clean=True, cleanup=True, max_wait_s=9999, poll_every_s=3):
    """
    --------------------------------------------------------------------------------
    ATLAS  (Asteroid Terrestrial-impact Last Alert System, forced photometry)
    --------------------------------------------------------------------------------
    Access. ATLAS Forced Photometry Server (fallingstar-data.com/forcedphot) -- a job
    is queued for the exact (ra, dec); use_reduced=True runs tphot (PSF-fitting
    photometry; Tonry 2011, Sonnett et al. 2013) on the reduced (non-difference)
    images, returning total flux. mjd_min defaults to 57000 (full history; omitting
    it returns only ~30 days). Filters: c (cyan, ~420-650 nm) -> atlas-c, o (orange,
    ~560-820 nm) -> atlas-o. ATLAS is a quadruple 0.5 m system: Haleakala and Mauna
    Loa (Hawaii), El Sauce (Chile), Sutherland (South Africa); 30 s exposures, ~4 per
    field over an hour, calibrated to Refcat2 (Pan-STARRS/Gaia) on the AB system.
    This is FORCED photometry -- a flux is measured at the input position on every
    exposure, so the light curve is eclipse-complete (faint epochs give low or
    slightly-negative flux rather than drop-outs).

    Time -> BJD_TDB. The server reports the exposure MJD on the UTC scale at the
    observer (topocentric), so no barycentric correction is baked in. The reported
    MJD is the exposure MID-POINT, not the start: the ATLAS Solar System Catalog
    README defines "MJD_ob : exposure midpoint time at observer, [MJD-UTC]", and
    Tonry et al.'s 3I/ATLAS paper states the MJD "is the actual mean time of the
    exposure". These are the same per-exposure timestamps the forced-photometry
    server uses, and no ATLAS source describes a "start" convention. The MJD is
    therefore taken DIRECTLY (no half-exposure offset is added), treated as UTC at
    Haleakala, scale-converted to TDB, and barycentre-corrected:

        t       = Time(MJD, format="mjd", scale="utc", location=ATLAS_SITE)
        BJD_TDB = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    (Earlier versions added +15 s on the assumption that MJD was the exposure start;
    that double-counted the half-exposure and placed ATLAS 15 s late relative to the
    other surveys. It has been removed.)

    Flux -> uJy. ATLAS provides flux natively in AB micro-Janskys
    (m_AB = 23.9 - 2.5*log10(uJy)), so Target_flux = uJy and Target_flux_err = duJy
    are taken directly with no conversion. Because use_reduced=True, these are total
    (absolute) fluxes, not template-subtracted differences, and are positive for real
    detections. System "AB", F0 = 3631 Jy.

    Cleaning. Always: drop NaN in MJD/uJy/duJy and require duJy > 0. With clean=True,
    the standard ATLAS image-quality cuts are applied -- err == 0; on-chip position
    (100 <= x,y <= 10460, a 100-px edge exclusion); PSF shape sane
    (1.6 <= maj, min <= 5 pixels, around the ~2-px typical PSF); good aperture fit
    (-1 <= apfit <= -0.1); deep image (mag5sig > 17); dark sky (Sky > 17); and a
    loose error cap (duJy < 10000). These are consistent in spirit with the ATClean
    community recommendations (Rest et al. 2024). With clean=False, only err == 0 and
    a loose depth cut (mag5sig > 10) are applied. Note these cuts filter on
    image/measurement quality, not per-point detection significance, so low-SNR and
    slightly-negative fluxes are retained with their errors for downstream SNR
    filtering.

    Data-quality caveats (not removed by clean): the difference-imaging reference
    templates changed around MJD 58417 (2018-10-26) and 58882 (2020-02-03) -- less
    relevant here since use_reduced=True bypasses difference images, but the
    reference-catalogue calibration epoch still applies; and in crowded fields the
    reduced-image flux can include blends.
    """
    sid = int(source_id)
    headers = _get_headers(token)

    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "ATLAS", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "ATLAS", "source_id": sid,
                     "observations": len(df), "status": "success"})

    text, state = None, None
    try:
        with _make_session() as s:
            task_url = _queue_job(s, ra, dec, headers, mjd_min, use_reduced)
            if not task_url:
                return result("queue_failed_or_full")
            state, result_url = _poll_result(s, task_url, headers, max_wait_s, poll_every_s)
            if state == "done":
                text = _fetch_result(s, result_url, _taskid(task_url), headers)
            if cleanup and ((state == "nodata") or (state == "done" and text is not None)):
                try:
                    s.delete(task_url, headers=headers, timeout=30)
                except Exception:
                    pass
            if state == "timeout":
                return result("polling_timeout")
            if state in ("error", "nodata"):
                return result("no_result_or_data")
    except Exception as e:
        return result(f"request_failed: {type(e).__name__}: {e}")

    if not text:
        return result("fetch_failed_no_text")
    try:
        df = pd.read_csv(io.StringIO(text), sep=r"\s+", engine="python")
    except Exception as e:
        return result(f"parse_failed: {e}")
    df = df.rename(columns={"###MJD": "MJD"})

    if not {"MJD", "uJy", "duJy", "F", "err"}.issubset(df.columns):
        return result("missing_columns")
    for col in ["MJD", "uJy", "duJy", "err"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["MJD", "uJy", "duJy"])
    df = df[df["duJy"] > 0]

    if clean:
        if not set(_CLEAN_COLS).issubset(df.columns):
            return result("missing_clean_columns")
        for c in _CLEAN_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df[
            (df["duJy"] < 10000) & (df["err"] == 0) &
            df["x"].between(100, 10460) & df["y"].between(100, 10460) &
            df["maj"].between(1.6, 5) & df["min"].between(1.6, 5) &
            df["apfit"].between(-1, -0.1) &
            (df["mag5sig"] > 17) & (df["Sky"] > 17)
        ]
    else:
        df = df[df["err"] == 0]
        if "mag5sig" in df.columns:
            df = df[pd.to_numeric(df["mag5sig"], errors="coerce") > 10]

    df = df[df["F"].isin(ATLAS_FILTERS)]
    if len(df) == 0:
        return result("filtered_out")
    df = df.reset_index(drop=True)

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    mjd = df["MJD"].to_numpy(float) 
    t = Time(mjd, format="mjd", scale="utc", location=ATLAS_SITE)
    bjd_tdb = (t.tdb + t.light_travel_time(coord, kind="barycentric")).jd

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     df["uJy"].to_numpy(float),
        "Target_flux_err": df["duJy"].to_numpy(float),
        "Filter":          df["F"].map(ATLAS_FILTERS).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)

    clear_atlas_queue(token, verbose=False)

    return result(None, out)


def clear_atlas_queue(token=None, verbose=True):
    """Delete ALL tasks on your account. Run once to clear stuck/finished jobs."""
    headers = _get_headers(token)
    deleted = 0
    with _make_session() as s:
        url = f"{BASEURL}/queue/"
        while url:
            r = s.get(url, headers=headers, timeout=30)
            r.raise_for_status()
            j = r.json()
            for task in j.get("results", []):
                turl = _abs(task.get("url"))
                if turl:
                    try:
                        s.delete(turl, headers=headers, timeout=30)
                        deleted += 1
                    except Exception:
                        pass
            url = _abs(j.get("next"))
    if verbose:
        print(f"Deleted {deleted} ATLAS tasks.")
    return deleted