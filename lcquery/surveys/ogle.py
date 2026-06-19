import io, os, re
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

LCO = EarthLocation.from_geodetic(
    lon=-70.7003 * u.deg, lat=-29.0083 * u.deg, height=2380.0 * u.m)

_ZP = {"I": 23.4585, "V": 23.9015}

OCVS_ROOTS = [
    "https://ftp.astrouw.edu.pl/ogle/ogle4/OCVS/",
    "https://ftp.astrouw.edu.pl/ogle/ogle3/OIII-CVS/",
]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_MASTER_CACHE = os.path.join(_THIS_DIR, "ogle_ocvs_master.csv")

_HTTP = None
def _http():
    global _HTTP
    if _HTTP is None:
        s = requests.Session()
        s.mount("https://", HTTPAdapter(max_retries=Retry(
            total=4, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])))
        _HTTP = s
    return _HTTP

_ROW_RE = re.compile(
    r"^\s*(?P<sid>OGLE\S+)\b.*?"
    r"(?P<rah>\d{1,2})[:\s](?P<ram>[0-5]\d)[:\s](?P<ras>\d{1,2}(?:\.\d+)?)\s+"
    r"(?P<sign>[+\-\u2212])(?P<ded>\d{1,2})[:\s](?P<dem>[0-5]\d)[:\s](?P<des>\d{1,2}(?:\.\d+)?)")


def _list_dir(url):
    """Parse an Apache/nginx directory index; return (subdirs, files), immediate children only."""
    r = _http().get(url, timeout=60); r.raise_for_status()
    subdirs, files = [], []
    for href in re.findall(r'href="([^"]+)"', r.text, flags=re.I):
        if href.startswith("?") or href in ("../", "/"):
            continue
        full = urljoin(url, href)
        if not full.startswith(url):
            continue
        name = full[len(url):]
        if not name or "/" in name.rstrip("/"):
            continue
        (subdirs if name.endswith("/") else files).append(name)
    return subdirs, files


def _discover_catalogs(roots, maxdepth=4, verbose=True):
    """Walk the OCVS tree; return every directory holding an ident*.dat."""
    found, stack = [], [(r, 0) for r in roots]
    while stack:
        url, depth = stack.pop()
        try:
            subdirs, files = _list_dir(url)
        except Exception:
            continue
        idents = [f for f in files if re.match(r"ident.*\.dat$", f, re.I)]
        if idents:
            found.append((url, idents[0]))
            continue                                  
        if depth < maxdepth:
            for d in subdirs:
                if d.lower().startswith("phot"):      
                    continue
                stack.append((url + d, depth + 1))
        _time.sleep(0.05)
    if verbose: print(f"[OGLE] discovered {len(found)} OCVS catalogues")
    return found


def build_ogle_master(roots=OCVS_ROOTS, cache_path=_MASTER_CACHE,
                      skip_patterns=None, verbose=True):
    """
    Crawl the ENTIRE OGLE OCVS (all classes, regions, phases), parse every ident file,
    build one master (star_id, ra, dec, cat_dir) table. Cached to disk: the heavy first
    run (~1M+ variables, a few hundred MB) happens once.
    skip_patterns: substrings to skip, e.g. ["/lpv/"] to drop the huge long-period files.
    """
    if cache_path and os.path.exists(cache_path):
        if verbose: print(f"[OGLE] loading cached master from {cache_path}")
        return pd.read_csv(cache_path, dtype={"star_id": str})
    cats = _discover_catalogs(roots, verbose=verbose)
    rows = []
    for cat_dir, ident_name in cats:
        if skip_patterns and any(p in cat_dir for p in skip_patterns):
            continue
        try:
            txt = _http().get(cat_dir + ident_name, timeout=240).text
        except Exception:
            if verbose: print(f"[OGLE]   FAILED {cat_dir}")
            continue
        n0 = len(rows)
        for line in txt.splitlines():
            m = _ROW_RE.match(line)
            if not m:
                continue
            try:
                ra = (float(m["rah"]) + float(m["ram"])/60 + float(m["ras"])/3600) * 15.0
                sgn = -1.0 if m["sign"] in ("-", "\u2212") else 1.0
                dec = sgn * (float(m["ded"]) + float(m["dem"])/60 + float(m["des"])/3600)
            except Exception:
                continue
            rows.append((m["sid"], ra, dec, cat_dir))
        if verbose: print(f"[OGLE]   {cat_dir}  (+{len(rows)-n0})")
    df = pd.DataFrame(rows, columns=["star_id", "ra_deg", "dec_deg", "cat_dir"])
    if cache_path and len(df):
        try: df.to_csv(cache_path, index=False)
        except Exception: pass
    if verbose: print(f"[OGLE] master built: {len(df)} variables")
    return df


_MASTER = None
def _master():
    global _MASTER
    if _MASTER is None:
        df = build_ogle_master()
        if df is None or len(df) == 0:
            _MASTER = (None, None)
        else:
            _MASTER = (df, SkyCoord(df["ra_deg"].to_numpy()*u.deg,
                                    df["dec_deg"].to_numpy()*u.deg))
    return _MASTER


def _parse_phot(text, band):
    try:
        raw = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None,
                          usecols=[0, 1, 2], names=["t", "mag", "magerr"], comment="#")
    except Exception:
        return None
    raw = raw.apply(pd.to_numeric, errors="coerce").dropna()
    if len(raw) == 0:
        return None
    t = raw["t"].to_numpy(float)
    raw["hjd"] = np.where(t > 2.4e6, t, t + 2450000.0)
    raw["band"] = band
    return raw


def _fetch_phot(cat_dir, star_id):
    """Discover the catalogue's phot layout (phot/, phot_ogle4/, I/, I_o4/, ...) and fetch I & V."""
    try:
        subdirs, _ = _list_dir(cat_dir)
    except Exception:
        return []
    frames = []
    for pdir in [s for s in subdirs if s.lower().startswith("phot")]:
        try:
            bands, _ = _list_dir(cat_dir + pdir)
        except Exception:
            continue
        for b in bands:
            band = b.rstrip("/").split("_")[0].upper()  
            if band not in ("I", "V"):
                continue
            url = cat_dir + pdir + b + star_id + ".dat"
            try:
                rr = _http().get(url, timeout=90)
            except Exception:
                continue
            if rr.status_code == 200 and rr.text.strip():
                ph = _parse_phot(rr.text, band)
                if ph is not None and len(ph) > 0:
                    frames.append(ph)
    return frames


def fetch_ogle_lc(source_id, ra, dec, radius_arcsec=2.0, clean=True):
    """
    --------------------------------------------------------------------------------
    OGLE  (OGLE-III / OGLE-IV Collection of Variable Stars, OCVS)
    --------------------------------------------------------------------------------
    Access & scope. Crawls the OGLE Collection of Variable Stars (OCVS) across the
    OGLE-III (OIII-CVS) and OGLE-IV (ogle4/OCVS) FTP trees, parsing every ident*.dat
    into a cached master table of (star_id, RA, Dec, catalog_dir). The target
    (RA, Dec) is matched to the nearest OCVS variable within radius_arcsec (default
    2 arcsec); the matched star's phot/I/<id>.dat and phot/V/<id>.dat are fetched.
    Coverage is variable-only -- OGLE serves OCVS light curves only for catalogued
    variables, so a target it observed but never classified returns no_match. Filter
    label is ogle-i / ogle-v. Photometry is PSF / Difference Image Analysis (DIA) from
    the 1.3 m Warsaw Telescope at Las Campanas. The pipeline fetches from every phot*
    subdirectory a catalogue provides, concatenating multi-phase photometry (e.g.
    phot_ogle2/ = OGLE-II 1997-2000, phot/ = OGLE-IV) into one light curve -- all from
    the same telescope, all HJD-2450000, all standard I/V, so time and flux conversion
    is uniform across phases and only a negligible inter-phase zero-point offset
    (~0.01-0.02 mag) is introduced.

    Time -> BJD_TDB. The phot files store time as HJD - 2450000 (Heliocentric Julian
    Date, reduced; UTC-based), confirmed against the OCVS format spec. _parse_phot
    reconstructs full HJD (t + 2450000 for reduced values, pass-through for full
    JD > 2.4e6). Because the time is heliocentric, conversion is a two-step round
    trip:

        t0      = Time(HJD, format="jd", scale="utc", location=LCO)
        t_topo  = t0 - t0.light_travel_time(coord, kind="heliocentric")  # undo helio -> topo UTC
        BJD_TDB = (t_topo.tdb + t_topo.light_travel_time(coord, kind="barycentric")).jd

    It removes the heliocentric light-travel correction OGLE applied (recovering the
    topocentric UTC observation time at the 1.3 m Warsaw Telescope), then re-applies
    the barycentric correction with the UTC -> TDB scale shift. The HJD<->BJD
    difference is up to +/-4 s and varies annually, so the round trip -- not a constant
    offset -- is what produces a correct BJD_TDB. Residuals are all negligible: site
    choice < 21 ms (single site); light-travel time evaluated at the HJD epoch versus
    the true epoch costs << 1 ms; and the UTC-timescale assumption (OGLE doesn't state
    TT) would at worst add a ~constant ~69 s shift, invisible to an hour-to-day period
    search.

    Flux -> uJy. OGLE magnitudes are standard Johnson V / Cousins I (calibrated to
    standard stars; the OGLE-IV I filter closely reproduces the standard system).
    Converted to Vega-system flux density via Target_flux[uJy] = 10^((ZP - mag)/2.5),
    with ZP_V = 23.9015 (Johnson V Vega, 3636 Jy ~ AB) and ZP_I = 23.4585 (Cousins I
    Vega, 2416 Jy -- the standard Bessell value; an alternative ~2550 Jy convention
    exists but differs only by a constant scale). Errors:
    Target_flux_err = Target_flux * magerr / 1.0857 (linearised mag->flux propagation,
    = flux * 0.921 * magerr). The output is a Vega in-band flux density, not AB; since
    OGLE I/V are unique bandpasses never co-fitted with other surveys' fluxes, the
    exact zero point only sets a constant scale and is irrelevant to period detection.
    Per-band overrides: ogle-i {Vega, 2416 Jy, Cousins I},
    ogle-v {Vega, 3636 Jy, Johnson V}.

    Cleaning. The phot files carry no quality flag (three columns only: HJD, mag,
    magerr), so quality control is range-based. _parse_phot coerces to numeric and
    drops non-numeric/NaN rows; after concatenation, drop_duplicates(["hjd","band"])
    removes repeated epochs from overlapping fields. With clean=True the cuts are
    magerr > 0, magerr < 1.0, and 5 < mag < 25 -- loose sanity bounds that strip
    garbage and any sentinel/failed points (which lie outside these ranges) while
    preserving real variability. A final dropna(["hjd","mag","magerr"]) guards the
    output.
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "OGLE", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "OGLE", "source_id": sid,
                     "observations": len(df), "status": "success"})

    df, coord = _master()
    if df is None:
        return result("catalog_unavailable")
    target = SkyCoord(float(ra)*u.deg, float(dec)*u.deg, frame="icrs")
    idx, sep2d, _ = target.match_to_catalog_sky(coord)
    if float(np.ravel(sep2d.arcsec)[0]) > radius_arcsec:
        return result("no_match")
    row = df.iloc[int(np.ravel(np.asarray(idx))[0])]

    frames = _fetch_phot(row["cat_dir"], row["star_id"])
    if not frames:
        return result("no_data")
    ph = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["hjd", "band"])
    if clean:
        ph = ph[(ph["magerr"] > 0) & (ph["magerr"] < 1.0) & (ph["mag"] > 5) & (ph["mag"] < 25)]
    ph = ph.dropna(subset=["hjd", "mag", "magerr"])
    if len(ph) == 0:
        return result("filtered_out")
    ph = ph.reset_index(drop=True)

    hjd = ph["hjd"].to_numpy(float)
    t0 = Time(hjd, format="jd", scale="utc", location=LCO)
    t_utc = t0 - t0.light_travel_time(target, kind="heliocentric")
    bjd_tdb = (t_utc.tdb + t_utc.light_travel_time(target, kind="barycentric")).jd

    mag = ph["mag"].to_numpy(float); magerr = ph["magerr"].to_numpy(float)
    zp = ph["band"].map(_ZP).to_numpy(float)
    flux = 10 ** ((zp - mag) / 2.5)
    flux_err = flux * magerr / 1.0857

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          ("ogle-" + ph["band"].astype(str).str.lower()).to_numpy(),
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)