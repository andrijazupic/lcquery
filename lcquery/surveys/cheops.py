import hashlib
import io
import os
import re
import tarfile
import tempfile
import warnings
import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]
_FILTER = "cheops-white"
CHEOPS_F0_JY = 3264.0957620195                  # SVO CHEOPS/CHEOPS.band Vega zero-point flux, Jy
_APERTURES = ("default", "optimal", "rinf", "rsup")
_KEY = re.compile(r"CH_PR\d{6}_TG\d{6}_V\d{4}", re.I)
_LEAP_FIX_JD = 2458934.5                        # 2020-03-26 00:00 UTC: leap-second table updated
_EVENT_DIGITS = ("saa", "temp", "earth", "moon", "sun", "cr")
_EVENT_MAX = {"saa": 0, "temp": 0, "earth": 1, "moon": 0, "sun": 0, "cr": 2}   # pycheops defaults


def _imports():                                 # dace-query is an optional dependency -> lazy
    from dace_query.cheops import Cheops
    return Cheops


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def _col(tab, name, fill=np.nan):
    """Column as a float array, masked entries -> fill."""
    return np.ma.filled(np.ma.asarray(tab[name], dtype=float), fill)


def _event_digits(event):
    """EVENT is decimal-coded (CHEOPS Observers Manual), NOT a bit mask: units = SAA,
    tens = temperature, hundreds = Earth, thousands = Moon, 10^4 = Sun, 10^5 = cosmic rays."""
    e = np.asarray(event, dtype=np.int64)
    return {"saa": e % 10, "temp": (e // 10) % 10, "earth": (e // 100) % 10,
            "moon": (e // 1000) % 10, "sun": (e // 10000) % 10, "cr": e // 100000}


def _native_key(h):
    """Visit key rebuilt from the FITS header (CH_PRttpppp_TGrrrrvv_Vaapp)."""
    k = ("PROGTYPE", "PROG_ID", "REQ_ID", "VISITCTR", "ARCH_REV", "PROC_NUM")
    if not all(x in h for x in k):
        return None
    try:
        return "CH_PR{:02d}{:04d}_TG{:04d}{:02d}_V{:02d}{:02d}".format(*(int(h[x]) for x in k))
    except (TypeError, ValueError):
        return None


def _lightcurve_files(path, aperture):
    """All (bytes, filename) DRP SCI_COR_Lightcurve-<APERTURE> FITS inside `path`: a FITS
    file, a .tar/.tar.gz archive, or a directory containing either."""
    pat = re.compile(rf"_SCI_COR_Lightcurve-{aperture.upper()}_V\d{{4}}\.fits$", re.I)
    out, loose = [], []
    paths = ([os.path.join(r, f) for r, _, fs in os.walk(path) for f in fs] if os.path.isdir(path) else [path])
    for p in paths:
        name = os.path.basename(p)
        if tarfile.is_tarfile(p):
            with tarfile.open(p) as tar:
                for m in tar.getmembers():
                    if m.isfile() and pat.search(os.path.basename(m.name)):
                        out.append((tar.extractfile(m).read(), os.path.basename(m.name)))
        elif pat.search(name):
            out.append((open(p, "rb").read(), name))
        elif name.lower().endswith(".fits"):
            loose.append((open(p, "rb").read(), name))      # single-file DACE download, default name
    return out if out else loose


def _read(blob):
    """(header, table, units) of the SCI_COR_Lightcurve extension; primary keys merged in."""
    with fits.open(io.BytesIO(blob), memmap=False) as hdul:
        ext = [h for h in hdul[1:] if h.name.upper() == "SCI_COR_LIGHTCURVE"] or [hdul[1]]
        hdr = {**dict(hdul[0].header), **dict(ext[0].header)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tab = Table.read(ext[0])
    return hdr, tab


def _flux_kind(unit):
    """'counts' (electrons), 'rate' (electrons/s) or ValueError; missing unit -> counts (DPDD)."""
    s = str(unit or "").strip().lower().replace(" ", "")
    if s in ("", "none"):
        return "counts", "assumed electrons (no TUNIT; DPDD definition)"
    s = s.replace("electrons", "electron").replace("e-", "electron")
    try:
        parsed = u.Unit(s)
    except ValueError:
        raise ValueError(f"unrecognised FLUX unit {unit!r}")
    if parsed.is_equivalent(u.electron):
        return "counts", str(unit)
    if parsed.is_equivalent(u.electron / u.s):
        return "rate", str(unit)
    raise ValueError(f"FLUX unit {unit!r} is neither electrons nor electrons/s")


def fetch_cheops_lc(source_id, ra, dec, radius_arcsec=10.0, clean=True, aperture="default",
                    decontaminate=False, target_name=None, file_keys=None, lightcurve_files=None,
                    cheops_mag=None, flux_scale_uJy_per_e_s=None, max_visits=None, match_arcsec=2.5,
                    event_max=None, early_time_offset_s=0.0, download_dir=None, query_limit=10000):
    """
    --------------------------------------------------------------------------------
    CHEOPS  (ESA CHaracterising ExOPlanet Satellite, pointed visits, 2020-)
    --------------------------------------------------------------------------------
    Access. DACE (dace-query >= 3.0): Cheops.query_region(SkyCoord, Angle) cone search ->
    one row per visit (file_key); each visit's official Data Reduction Pipeline light
    curve, SCI_COR_Lightcurve-<APERTURE>, is fetched with Cheops.download(file_type=
    "lightcurves", aperture=..., compressed=False) and read directly with astropy -- no
    pycheops, whose reader median-normalises, removes every flux <= 0 and leaves the errors
    of its decontamination unscaled. CHEOPS is POINTED: only approved targets have data,
    so almost every all-sky source returns no_data (Morris et al. 2021 observed six white
    dwarfs, so WD targets exist). DACE serves the PUBLIC GTO visits; public Guest-Observer
    visits are in the ESA CHEOPS Mission Archive, which is not queried -- download them
    there and pass the FITS/tar paths as lightcurve_files (bypasses DACE). file_keys
    (CH_PRxxxxxx_TGxxxxxx_Vxxxx) fetches verified visits directly, bypassing the cone.
    When DACE lists several reprocessings of one visit (e.g. _V0200 and _V0300), ONLY THE
    HIGHEST VERSION is used, and byte-identical files are read once, so no measurement is
    ever counted twice. A response with query_limit rows is search_truncated, never
    silently cut. aperture: "default" (DRP R=25 px), "optimal", "rinf", "rsup"; the file's
    DATANAME must confirm it. max_visits limits the visits fetched; download_dir keeps the
    downloads (otherwise a temporary directory is removed). Filter label: cheops-white.

    Target identity. Each visit FITS carries TARGNAME and RA_TARG/DEC_TARG. Two visits are
    the SAME target if their target positions agree within match_arcsec OR their names
    agree (case/punctuation-insensitive) -- names alone are not used, because one star is
    often catalogued under different names by different programmes, and positions alone
    are not used, because a high-proper-motion target moves between visits. More than one
    target -> status ambiguous_match, nothing returned, never a guess; target_name selects
    one explicitly. match_sep_arcsec reports the kept target's offset from (ra, dec), and
    per visit identity_check compares the requested file_key with the key rebuilt from the
    header (ok / mismatch / unavailable).

    Time -> BJD_TDB. BJD_TIME is the barycentric time at the MIDDLE of each (stacked)
    exposure on the TT scale (TIMESYS = 'TT', TIMEPIXR = 0.5; CHEOPS DPDD issue 1.22,
    Appendix B; Observers Manual issue 7, sec. 3.1.7). Converted by a pure time-scale change
        BJD_TDB = Time(BJD_TIME, format="jd", scale="tt").tdb.jd     (|TDB - TT| <= 1.7 ms);
    no barycentric correction and no half-exposure shift (both already in BJD_TIME). TDB
    files pass unchanged; any other TIMESYS, TIMEPIXR != 0.5 or a non-full-JD BJD_TIME
    rejects the visit (status unexpected_time_frame if that removes everything).
    Early-data caveat: Fortier et al. (2024, App. C) found time stamps 5 s early in data
    processed with an outdated leap-second table, fixed 2020-03-26. Current DACE products
    are reprocessed, so by default (early_time_offset_s=0) nothing is added; visits that
    started before 2020-03-26 are listed in pre_2020_03_26_visits. Set early_time_offset_s=5
    only for a product you have verified is affected; it is applied to those visits only.

    Flux -> uJy (Vega, CHEOPS band; APPROXIMATE level). FLUX/FLUXERR are electrons summed
    over TEXPTIME (= NEXP x EXPTIME, header or per-row column, s) and are converted to a
    count rate C = FLUX/TEXPTIME first -- otherwise visits with different exposure settings
    get spurious level ratios; files already in electrons/s are not divided again; any other
    unit rejects the visit. No public electrons -> Jy calibration exists for science use (the
    Fortier et al. 2024 F0X relation needs the REF_APP_FluxConversion reference file, not
    served by DACE), so ONE positive constant for ALL kept visits puts the rates on the
    star's catalogue CHEOPS-band flux:
        Target_flux[uJy] = C * F_X / median(C over all kept visits),
        F_X = 1e6 * 3264.0957620195 Jy * 10^(-0.4 X)     (SVO CHEOPS Vega zero point),
    X = cheops_mag if given, else the median header MAG_CHPS (CHEOPS-band Vega magnitude
    of the CHEOPS target catalogue); if neither exists, X = Gaia G (MAG_G, Vega) as an
    approximation flagged in anchor_source (CHEOPS and G differ by a colour term), else
    status no_anchor. flux_scale_uJy_per_e_s, if given, replaces the anchor by your own
    calibrated factor. One constant keeps eclipse depths, amplitudes, visit-to-visit levels,
    timing and weights exactly as observed; only the absolute level is approximate (the
    star's variability shifts its median, slow throughput ageing makes visits years apart
    differ by up to a few percent). Use CHEOPS for variability and timing, not on an SED.

    Contamination. decontaminate=False (default) keeps the native DRP aperture flux, like
    SAP flux. decontaminate=True SUBTRACTS the modelled contaminant flux: with CONTA_LC =
    F_contaminants / F_target (DPDD) and the visit's target level F_ref = median(C/(1+c)),
    C_target = C - c * F_ref. Unlike dividing by 1+c, subtraction leaves the target's own
    variability (eclipse depths) undiluted and does not rescale it with roll angle. Errors
    are not changed; the CONTA_LC model error is a common systematic, reported in metadata.

    Cleaning. clean=True keeps rows with STATUS == 0 whose EVENT digits do not exceed the
    pycheops defaults (SAA 0, temperature 0, Earth <= 1, Moon 0, Sun 0, cosmic rays <= 2;
    override with event_max={...}); clean=False keeps every row. Always: finite time, flux
    and error, error > 0. flux > 0 is deliberately NOT imposed; NO sigma clipping, NO
    per-visit normalisation, NO detrending -- roll-angle systematics (98.7-min orbit,
    dangerous for ~1.6 h binaries), orbit gaps and occultation edges are left to the analysis.

    Metadata: survey/source_id/observations/status (recorded by lcquery's master) plus
    target, match_sep_arcsec, anchor_source, anchor_mag, anchor_uJy_per_e_s,
    pre_2020_03_26_visits, visits (per visit: file_key, identity_check, target, RA/Dec,
    DRP version, NEXP/EXPTIME/TEXPTIME, TIMESYS, flux unit, rows kept/total, STATUS != 0
    rows, median CONTA_LC, MAG_CHPS/MAG_G/T_EFF, BJD range, sha256) and failed_visits.
    Statuses: success, partial_success, no_data, import_failed, search_failed,
    search_truncated, ambiguous_match, unexpected_time_frame, no_anchor, download_failed,
    filtered_out. Invalid arguments raise ValueError.

    References. DACE CHEOPS API (dace-query) -- CHEOPS Data Products Definition Document,
    CHEOPS-UGE-SOC-DD-002 issue 1.22, Appendix B (SCI_COR_Lightcurve) -- CHEOPS Observers
    Manual, CHEOPS-UGE-PSO-MAN-001 issue 7, sec. 3.1.5 (events) and 3.1.7 (time) -- Fortier
    et al. 2024, A&A 687, A302 (App. B flux scale, App. C timing) -- SVO Filter Profile
    Service, CHEOPS/CHEOPS.band (Vega zero point) -- pycheops dataset.py (EVENT decoding,
    default thresholds) -- Morris et al. 2021, A&A 651, L12 (CHEOPS white dwarfs).
    """
    sid = int(source_id)
    aperture = str(aperture).lower()
    if aperture not in _APERTURES:
        raise ValueError(f"aperture must be one of {_APERTURES}")
    if not (np.isfinite(radius_arcsec) and radius_arcsec > 0):
        raise ValueError("radius_arcsec must be finite and positive")
    if file_keys is not None and lightcurve_files is not None:
        raise ValueError("give file_keys OR lightcurve_files, not both")
    if flux_scale_uJy_per_e_s is not None and not (np.isfinite(flux_scale_uJy_per_e_s) and flux_scale_uJy_per_e_s > 0):
        raise ValueError("flux_scale_uJy_per_e_s must be finite and positive")
    limits = {**_EVENT_MAX, **(event_max or {})}
    if set(limits) != set(_EVENT_DIGITS):
        raise ValueError(f"event_max keys must be among {_EVENT_DIGITS}")
    want = None if target_name is None else _norm(target_name)

    visits, failures = [], []
    meta = {"survey": "CHEOPS", "source_id": sid, "target": None, "match_sep_arcsec": None,
            "aperture": aperture, "clean": bool(clean), "decontaminate": bool(decontaminate),
            "anchor_source": None, "anchor_mag": None, "anchor_uJy_per_e_s": None,
            "pre_2020_03_26_visits": [], "visits": visits, "failed_visits": failures}

    def result(status, df=None):
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=_COLS), {**meta, "observations": 0, "status": status}
        return df, {**meta, "observations": len(df), "status": status}

    # --- 1. which visits ------------------------------------------------------------------------
    blobs = []                                  # (requested file_key or None, bytes, filename)
    if lightcurve_files is not None:
        paths = [lightcurve_files] if isinstance(lightcurve_files, (str, os.PathLike)) else list(lightcurve_files)
        for p in paths:
            try:
                found = _lightcurve_files(os.path.expanduser(str(p)), aperture)
                if not found:
                    raise ValueError(f"no SCI_COR_Lightcurve-{aperture.upper()} file")
                blobs += [(None, b, n) for b, n in found]
            except Exception as e:
                failures.append({"file_key": None, "file": str(p), "error": f"{type(e).__name__}: {str(e)[:300]}"})
    else:
        try:
            Cheops = _imports()
        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {e}"
            return result("import_failed")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if file_keys is not None:
                    keys = [file_keys] if isinstance(file_keys, str) else list(file_keys)
                    keys = list(dict.fromkeys(str(k).strip().upper() for k in keys))
                    if not keys or any(_KEY.fullmatch(k) is None for k in keys):
                        raise ValueError("file_keys must be CH_PRxxxxxx_TGxxxxxx_Vxxxx identifiers")
                    res = Cheops.query_database(filters={"file_key": {"equal": keys}}, limit=int(query_limit),
                                                output_format="pandas")
                else:
                    res = Cheops.query_region(sky_coord=SkyCoord(float(ra) * u.deg, float(dec) * u.deg),
                                              angle=Angle(float(radius_arcsec), "arcsec"),
                                              limit=int(query_limit), output_format="pandas")
        except Exception as e:
            meta["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            return result("search_failed")
        res = pd.DataFrame(res) if res is not None and not isinstance(res, pd.DataFrame) else res
        if res is None or len(res) == 0:
            return result("no_data")
        if len(res) >= int(query_limit):
            return result("search_truncated")
        cols = {c.lower(): c for c in res.columns}
        if "file_key" not in cols:
            meta["error"] = f"DACE response without file_key; columns {list(res.columns)[:20]}"
            return result("search_failed")
        name_col = next((cols[c] for c in ("obj_id_catname", "target_name", "obj_id") if c in cols), None)
        if want is not None and name_col is not None:
            hit = res[name_col].map(_norm) == want
            if hit.any():
                res = res[hit]
        newest = {}                                    # one entry per visit: highest _Vxxxx
        for k in res[cols["file_key"]].astype(str).str.strip().str.upper():
            stem = k.rsplit("_V", 1)[0]
            if stem not in newest or k > newest[stem]:
                newest[stem] = k
        keys = sorted(newest.values())
        meta["n_visits_discovered"] = len(keys)
        if max_visits:
            keys = keys[:int(max_visits)]
        tmp = None if download_dir else tempfile.TemporaryDirectory(prefix="lcquery-cheops-")
        base = os.path.expanduser(download_dir) if download_dir else tmp.name
        try:
            for fk in keys:
                vdir = os.path.join(base, fk, aperture)
                try:
                    os.makedirs(vdir, exist_ok=True)
                    found = _lightcurve_files(vdir, aperture)                  # reuse a kept download
                    if not found:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            Cheops.download(filters={"file_key": {"equal": [fk]}}, file_type="lightcurves",
                                            aperture=aperture, compressed=False, output_directory=vdir)
                        found = _lightcurve_files(vdir, aperture)
                    if not found:
                        raise ValueError(f"download contains no SCI_COR_Lightcurve-{aperture.upper()} file")
                    best = max(found, key=lambda x: x[1])                     # newest file version
                    blobs.append((fk, best[0], best[1]))
                except Exception as e:
                    failures.append({"file_key": fk, "error": f"{type(e).__name__}: {str(e)[:300]}"})
        finally:
            if tmp is not None:
                tmp.cleanup()

    # --- 2. read, de-duplicate, identity ---------------------------------------------------------
    raw, seen = [], set()
    for fk, blob, fname in blobs:
        digest = hashlib.sha256(blob).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        try:
            hdr, tab = _read(blob)
            if str(hdr.get("TELESCOP", "CHEOPS")).strip().upper() != "CHEOPS":
                raise ValueError("not a CHEOPS product")
            dname = str(hdr.get("DATANAME", "")).strip().upper()     # real DRP files: bare 'DEFAULT'
            ap_in = dname if dname in {a.upper() for a in _APERTURES} else (
                re.search(r"LIGHTCURVE-(DEFAULT|OPTIMAL|RINF|RSUP)\b", dname).group(1)
                if re.search(r"LIGHTCURVE-(DEFAULT|OPTIMAL|RINF|RSUP)\b", dname) else None)
            if dname and ap_in is not None and ap_in != aperture.upper():
                raise ValueError(f"DATANAME {dname} is not the {aperture.upper()} aperture")
            native = _native_key(hdr)
            check = "unavailable" if (fk is None or native is None) else ("ok" if native == fk else "mismatch")
            raw.append({"fk": fk or native, "native": native, "check": check, "file": fname, "hdr": hdr,
                        "tab": tab, "sha256": digest})
        except Exception as e:
            failures.append({"file_key": fk, "file": fname, "error": f"{type(e).__name__}: {str(e)[:300]}"})
    stems = {}                                       # local files may also hold several reprocessings
    for i, r in enumerate(raw):
        st = (r["native"] or r["fk"] or r["sha256"]).rsplit("_V", 1)[0]
        if st not in stems or (r["native"] or "") > (raw[stems[st]]["native"] or ""):
            stems[st] = i
    raw = [raw[i] for i in sorted(stems.values())]
    if not raw:
        return result("download_failed" if failures else "no_data")

    ra_t = np.array([float(r["hdr"].get("RA_TARG", np.nan)) for r in raw])
    de_t = np.array([float(r["hdr"].get("DEC_TARG", np.nan)) for r in raw])
    names = [_norm(r["hdr"].get("TARGNAME", "")) for r in raw]
    if want is not None:
        keep = np.array([n == want for n in names])
        if not keep.any():
            return result("no_data")
    else:
        n = len(raw)                                  # union-find on "same position OR same name"
        parent = list(range(n))
        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        pos = SkyCoord(np.nan_to_num(ra_t) * u.deg, np.nan_to_num(de_t) * u.deg)
        fin = np.isfinite(ra_t) & np.isfinite(de_t)
        for i in range(n):
            for j in range(i + 1, n):
                close = fin[i] and fin[j] and pos[i].separation(pos[j]).arcsec <= match_arcsec
                if close or (names[i] and names[i] == names[j]) or not (fin[i] and fin[j]):
                    parent[root(i)] = root(j)
        groups = {root(i) for i in range(n)}
        if len(groups) > 1:
            meta["candidate_targets"] = sorted({str(r["hdr"].get("TARGNAME", "")) for r in raw})
            return result("ambiguous_match")
        keep = np.ones(n, bool)
    raw = [r for r, k in zip(raw, keep) if k]
    meta["target"] = str(raw[0]["hdr"].get("TARGNAME", ""))
    sep = SkyCoord(float(ra) * u.deg, float(dec) * u.deg).separation(
        SkyCoord(ra_t[keep] * u.deg, de_t[keep] * u.deg)).arcsec
    meta["match_sep_arcsec"] = float(np.nanmin(sep)) if np.isfinite(sep).any() else None

    # --- 3. per visit: time, rate, contamination, flags --------------------------------------------
    frames, n_badtime = [], 0
    for r in raw:
        hdr, tab, fk = r["hdr"], r["tab"], r["fk"]
        try:
            timesys = str(hdr.get("TIMESYS", "")).strip().upper()
            pixr = hdr.get("TIMEPIXR")
            bjd_native = _col(tab, "BJD_TIME")
            if (timesys not in ("TT", "TDB") or (pixr is not None and not np.isclose(float(pixr), 0.5))
                    or np.any(np.isfinite(bjd_native) & ((bjd_native < 2.4e6) | (bjd_native > 2.6e6)))):
                n_badtime += 1
                failures.append({"file_key": fk, "error": f"time frame TIMESYS={timesys!r} TIMEPIXR={pixr!r}"})
                continue
            if "TEXPTIME" in tab.colnames:
                texp = _col(tab, "TEXPTIME")
            elif hdr.get("TEXPTIME") is not None:
                texp = np.full(len(tab), float(hdr["TEXPTIME"]))
            elif hdr.get("NEXP") is not None and hdr.get("EXPTIME") is not None:
                texp = np.full(len(tab), float(hdr["NEXP"]) * float(hdr["EXPTIME"]))
            else:
                raise ValueError("no TEXPTIME / NEXP x EXPTIME")
            kind, unit = _flux_kind(tab["FLUX"].unit)
            flux, err = _col(tab, "FLUX"), _col(tab, "FLUXERR")
            if kind == "counts":
                ok_t = np.isfinite(texp) & (texp > 0)
                flux = np.where(ok_t, flux / np.where(ok_t, texp, 1), np.nan)
                err = np.where(ok_t, err / np.where(ok_t, texp, 1), np.nan)
            c = _col(tab, "CONTA_LC") if "CONTA_LC" in tab.colnames else np.zeros(len(tab))
            good = np.isfinite(bjd_native) & np.isfinite(flux) & np.isfinite(err) & (err > 0)
            n_valid = int(good.sum())
            n_status = int((_col(tab, "STATUS", fill=0) != 0).sum()) if "STATUS" in tab.colnames else None
            if clean:
                if "STATUS" in tab.colnames:
                    good &= _col(tab, "STATUS", fill=1) == 0
                if "EVENT" in tab.colnames:
                    ev = _col(tab, "EVENT", fill=-1)
                    integer = np.isfinite(ev) & (ev >= 0) & (ev == np.floor(ev)) & (ev < 1e6)
                    good &= integer
                    dig = _event_digits(np.where(integer, ev, 0))
                    for k, lim in limits.items():
                        good &= dig[k] <= lim
            if decontaminate:
                cc = np.where(np.isfinite(c) & (c >= 0), c, np.nan)
                good &= np.isfinite(cc)
                if good.any():
                    f_ref = float(np.nanmedian(flux[good] / (1.0 + cc[good])))
                    flux = flux - cc * f_ref
            bjd = bjd_native if timesys == "TDB" else Time(bjd_native, format="jd", scale="tt").tdb.jd
            early = bool(np.nanmin(bjd_native[np.isfinite(bjd_native)]) < _LEAP_FIX_JD) if np.isfinite(bjd_native).any() else False
            if early and early_time_offset_s:
                bjd = bjd + float(early_time_offset_s) / 86400.0
            hv = lambda k: hdr.get(k) if isinstance(hdr.get(k), (int, float, str, bool)) or hdr.get(k) is None else str(hdr.get(k))
            visit = {"file_key": fk, "identity_check": r["check"], "file": r["file"], "sha256": r["sha256"],
                     "target": hv("TARGNAME"), "ra_targ": hv("RA_TARG"), "dec_targ": hv("DEC_TARG"),
                     "obsid": hv("OBSID"), "drp_version": hv("PIPE_VER"), "nexp": hv("NEXP"),
                     "exptime_s": hv("EXPTIME"), "texptime_s": float(np.nanmedian(texp)), "timesys": timesys,
                     "flux_unit": unit, "rows_total": len(tab), "rows_valid": n_valid, "rows_kept": int(good.sum()),
                     "rows_status_nonzero": n_status, "median_conta_lc": float(np.nanmedian(c)) if np.isfinite(c).any() else None,
                     "mag_chps": hv("MAG_CHPS"), "mag_g": hv("MAG_G"), "t_eff": hv("T_EFF"),
                     "pre_2020_03_26": early, "time_offset_s": float(early_time_offset_s) if early else 0.0}
            if early:
                meta["pre_2020_03_26_visits"].append(fk)
            if good.any():
                visit.update(bjd_min=float(bjd[good].min()), bjd_max=float(bjd[good].max()))
                frames.append(pd.DataFrame({"BJD": bjd[good], "rate": flux[good], "rerr": err[good]}))
            visits.append(visit)
        except Exception as e:
            failures.append({"file_key": fk, "file": r["file"], "error": f"{type(e).__name__}: {str(e)[:300]}"})

    if not frames:
        if n_badtime:
            return result("unexpected_time_frame")
        return result("download_failed" if failures and not visits else "filtered_out")
    allp = pd.concat(frames, ignore_index=True)

    # --- 4. one constant for all visits ------------------------------------------------------------
    if flux_scale_uJy_per_e_s is not None:
        k, meta["anchor_source"] = float(flux_scale_uJy_per_e_s), "flux_scale_uJy_per_e_s (caller)"
    else:
        def med(key):
            v = [float(x[key]) for x in visits if x.get(key) not in (None, "")]
            v = [x for x in v if np.isfinite(x) and -5 < x < 30]
            return float(np.median(v)) if v else None
        if cheops_mag is not None:
            x, src = float(cheops_mag), "cheops_mag (caller)"
        elif med("mag_chps") is not None:
            x, src = med("mag_chps"), "MAG_CHPS (CHEOPS-band Vega, target catalogue)"
        elif med("mag_g") is not None:
            x, src = med("mag_g"), "MAG_G as CHEOPS magnitude (approximation: no colour term)"
        else:
            return result("no_anchor")
        ref = float(np.nanmedian(allp.rate))
        if not (np.isfinite(ref) and ref > 0):
            return result("filtered_out")
        k = 1e6 * CHEOPS_F0_JY * 10 ** (-0.4 * x) / ref
        meta.update(anchor_source=src, anchor_mag=x, anchor_reference_rate_e_s=ref)
    meta["anchor_uJy_per_e_s"] = k
    out = pd.DataFrame({"BJD": allp.BJD.to_numpy(float), "Target_flux": allp.rate.to_numpy(float) * k,
                        "Target_flux_err": allp.rerr.to_numpy(float) * k, "Filter": _FILTER})
    out = out.sort_values("BJD", kind="stable").reset_index(drop=True)
    return result("partial_success" if failures else "success", out)
