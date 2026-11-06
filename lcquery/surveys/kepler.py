import re
import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

# Same instrument and bandpass as K2 -> deliberately the SAME convention as k2.py.
# Keep these two constants identical to K2_F0_JY / K2_ZP_INST in k2.py.
KEPLER_F0_JY, KEPLER_ZP_INST = 3241.90, 25.10
KEPLER_TO_UJY = KEPLER_F0_JY * 1e6 * 10 ** (-0.4 * KEPLER_ZP_INST)   # ~0.2957 uJy per e-/s

_BKJD_REF = 2454833                      # BJDREFI of every Kepler light-curve file
_E_PER_S = u.electron / u.s
_OBSMODE = {"long cadence": 1800, "short cadence": 60}


def _kic_number(value):
    """'kplr009591503', 'KIC 9591503', '9591503' or 9591503 -> 9591503."""
    m = re.fullmatch(r"(?:kplr|kic)?\s*0*(\d+)", str(value).strip().lower())
    if m is None:
        raise ValueError(f"unrecognised KIC identifier: {value!r}")
    return int(m.group(1))


def fetch_kepler_lc(source_id, ra, dec, radius_arcsec=2.0, kic_id=None,
                    flux_col="pdcsap_flux", exptime=None, quarter=None,
                    quality_bitmask="default", min_data_release=25, download_dir=None):
    """
    --------------------------------------------------------------------------------
    Kepler  (Kepler prime mission, Q0-Q17, May 2009 - May 2013)
    --------------------------------------------------------------------------------
    Access. Kepler prime-mission light curves via lightkurve / MAST
    (search_lightcurve(coord, mission="Kepler", author="Kepler")) within a small
    cone, downloaded per product (one product = one quarter of long cadence, or one
    month of short cadence). This is SEPARATE from k2.py, which searches
    mission="K2" only and therefore never returns prime-mission data. Coverage is the
    single fixed Kepler field in Cygnus/Lyra (~115 deg^2); every other position
    returns no_data. Filter label is the nominal cadence in seconds, exactly as for
    K2: kepler-1800 (long cadence, 270 coadds of 6.54 s frames ~1765.5 s, i.e.
    29.4 min) and kepler-60 (short cadence, 9 coadds ~58.85 s), taken from the file's
    OBSMODE keyword. Same passband -- the number is the sampling, not a different
    filter. Forced (PDCSAP aperture) photometry -- a flux every cadence,
    eclipse-complete. Optional exptime ("long"/"short"/seconds) and quarter are passed
    straight to lightkurve; download_dir sets lightkurve's cache directory.

    Target identity. source_id is the caller's identifier (normally Gaia DR3), not a
    KIC number. In the default cone search, if the cone contains more than one KIC
    target the status is ambiguous_match and nothing is returned -- light curves of
    different KIC targets are never merged and the code never guesses. Pass kic_id
    (int, "KIC 9591503" or "kplr009591503") to fetch a counterpart you have verified
    yourself; this bypasses the cone, which is also the way to handle stars whose
    proper motion moves them out of a 2" cone (KIC positions are ~J2000-epoch, Gaia DR3
    is epoch 2016.0). A single KIC target inside the cone is accepted, and its
    separation from (ra, dec) is returned as match_sep_arcsec: for faint targets next
    to a brighter Kepler target, check it -- Kepler pixels are 3.98", so a neighbour's
    light curve can look plausible. Every downloaded file must carry the expected
    KEPLERID, and repeated archive records (same dataURI) are downloaded once.

    Time -> BJD_TDB. The TIME column is BKJD = BJD_TDB - 2454833.0 (header BJDREFI =
    2454833, BJDREFF = 0, TIMESYS = 'TDB', TIMEREF = 'SOLARSYSTEM'), the barycentric
    arrival time at the MIDDLE of each cadence (Kepler Archive Manual; reconstruction
    BJD = TIME + BJDREFI + BJDREFF). lightkurve reads this as an astropy Time (format
    bkjd, scale tdb), so lc.time.tdb.jd is the full BJD_TDB. No barycentric correction,
    no TIMECORR, no scale conversion and no half-cadence shift are applied (all are
    already in TIME); the code only VERIFIES that each product really is BJDREFI =
    2454833 on the TDB scale and skips it otherwise (status unexpected_time_frame if
    that removes everything). Timing-error guard: early Kepler data releases reported
    TIME in UTC instead of TDB (MAST "Statement on Kepler Timing Error": +66.184 s up
    to LC cadence 57139 / SC 1702663, +67.184 s after), corrected in later releases.
    Products whose header DATA_REL is below min_data_release (default 25, the final
    MAST reprocessing) are rejected instead of being silently ~1 min off; status is
    old_data_release if that removes everything. No correction is ever applied to
    DR25 products -- they are already right.

    Flux -> uJy (Vega). PDCSAP flux is in e-/s (the unit is checked; anything not
    convertible to e-/s, e.g. a normalised community light curve, is rejected),
    converted exactly as K2 via Target_flux[uJy] = flux[e-/s] * KEPLER_TO_UJY, with
    KEPLER_TO_UJY = F0 * 1e6 * 10^(-0.4*ZP_inst) ~ 0.2957, ZP_inst = 25.10 and
    F0 = 3241.90 Jy (SVO Kepler-band Vega zero-point flux). The metadata unit is
    uJy_Vega. ZP_inst = 25.10 is the GO-programme benchmark (a Kp = 12.0 G2V star
    gives 1.74e5 e-/s pre-flight); the same GO page gives an OBSERVED PDC rate of
    1.72e5 e-/s at Kp = 12.0 (ZP ~25.09), i.e. ~1% away. This is an approximate,
    instrumental in-band flux density, not absolute spectrophotometry: (1) the KIC Kp
    scale is approximately AB (built from SDSS-like g,r) while F0 is a Vega reference
    -- the same documented choice as K2/TESS, a pure constant factor (AB with 3631 Jy
    would give ~0.3312 instead); (2) published Kepler/K2 reductions find aperture- and
    channel-dependent zero points between ~24.7 and 25.3, varying by up to 0.2 mag
    across the 84 readout channels, so the true e-/s -> Jy factor is source- and
    quarter-dependent. One constant is applied to flux AND error, so every
    per-light-curve quantity (eclipse depth, amplitude ratios, period, timing, weights)
    is unchanged, which is what cross-survey period work needs. Do NOT place these
    values on an absolute SED or combine them at the absolute level with other
    surveys' fluxes; a timing validation does not validate the absolute scale.

    Flux column choice. PDCSAP (default) removes instrumental systematics and also
    corrects for crowding (CROWDSAP) and aperture flux fraction (FLFRCSAP); SAP
    (flux_col="sap_flux") does neither. PDC can attenuate or distort genuine
    variability on timescales of many days (Kepler data release notes), so for slow
    astrophysical trends re-run with sap_flux and compare. A constant factor cannot
    undo pipeline processing.

    Cleaning. Download with quality_bitmask="default" (overridable: "none", "hard",
    "hardest" or an integer). In lightkurve 2.x the default rejects cadences flagged
    as attitude tweak, safe mode, coarse point, Earth point, desaturation event,
    manual exclude, detector anomaly, no data (and thruster firing, a K2-era flag); it
    does NOT reject the Argabrightening or cosmic-ray flags. Then keep finite time,
    flux and flux_err with flux_err > 0. flux > 0 is deliberately NOT imposed (valid
    negative PDCSAP fluxes of faint sources are kept, keeping the noise distribution
    unbiased). No sigma clipping, flattening, stitching or per-quarter normalisation:
    quarter-to-quarter level steps (the target lands on a different CCD each quarter)
    are left for downstream handling. Long- and short-cadence products of the same
    epochs are both returned, separated by their Filter label -- they are the same
    photons sampled twice, so do not treat overlapping kepler-60 and kepler-1800 points
    as independent measurements.

    Returned metadata. Besides survey/source_id/observations/status (the only keys
    lcquery's master records), the dict carries kic, match_sep_arcsec,
    candidate_kic_ids, data_releases, n_rejected_old_release, n_rejected_time_frame,
    products (per file: quarter, filter, DATA_REL, FILEVER, cadence and summed
    integration time, rows, BJD range, filename) and failed_products (per file:
    dataURI and error). Statuses: success; partial_success (data returned but some
    product failed to download/read -- see failed_products); no_data; search_failed;
    ambiguous_match; old_data_release; unexpected_time_frame; download_failed;
    filtered_out. Invalid arguments raise ValueError (master records it as an
    exception status).

    Validation (lcquery 1.1.0, lightkurve 2.6.0; four non-Blazhko RRab stars V894,
    V2470, V1510 and V1107 Cyg, 88 MAST products). Every returned row was reproduced
    from the raw FITS files (TIME + BJDREFI + BJDREFF, PDCSAP_FLUX * KEPLER_TO_UJY, same
    quality mask) with 0 ms time and 0 relative flux difference; all products have
    DATA_REL = 25, TIMESYS = TDB, TIMEREF = SOLARSYSTEM, TIMEPIXR = 0.5. Binning the 30
    short cadences inside each long cadence, the two cadences agree to -0.1 +- 0.2 s in
    photon timing (model-free regression, calibrated by a 60 s injection) and to ~2 ms
    in timestamp bookkeeping. Median PDCSAP rates were 0.90-1.43 x the Kp = 12 benchmark
    scaled by the (single-epoch) KIC Kp of these variables, i.e. the uJy scale is good
    to tens of percent per star, as stated under Flux.

    References. Kepler Archive Manual (MAST, KDMC-10008), light-curve file format and
    time columns: https://archive.stsci.edu/kepler/manuals/archive_manual.pdf --
    MAST Statement on Kepler Timing Error:
    https://archive.stsci.edu/kepler/timing_error.html -- Kepler GO calibration and
    count-rate benchmarks: https://nexsci.caltech.edu/workshop/2012/keplergo/
    CalibrationSN.shtml -- channel zero points: Ridden-Harper et al. 2020 (K2:
    Background Survey), arXiv:2006.15045, sec. 2.1; aperture zero points 25.0-25.3:
    Aigrain et al. 2015, arXiv:1412.6304, sec. 2.3 (24.7 for a 1-pixel aperture:
    Nardiello et al. 2015, arXiv:1510.09180) -- validation stars: Nemec et al.
    2011, MNRAS 417, 1022 -- lightkurve search/download API:
    https://lightkurve.github.io/lightkurve/
    """
    sid = int(source_id)
    flux_col = str(flux_col).lower()
    if flux_col not in ("pdcsap_flux", "sap_flux"):
        raise ValueError("flux_col must be 'pdcsap_flux' or 'sap_flux'")
    if not (np.isfinite(radius_arcsec) and radius_arcsec > 0):
        raise ValueError("radius_arcsec must be finite and positive")
    want_kic = None if kic_id is None else _kic_number(kic_id)

    products, failures = [], []
    meta = {"survey": "Kepler", "source_id": sid, "kic": None, "match_sep_arcsec": None,
            "candidate_kic_ids": [], "flux_column": flux_col,
            "flux_scale_uJy_per_e_s": KEPLER_TO_UJY, "quality_bitmask": str(quality_bitmask),
            "data_releases": [], "n_rejected_old_release": 0, "n_rejected_time_frame": 0,
            "products": products, "failed_products": failures}

    def result(status, df=None):
        if df is None or len(df) == 0:
            return pd.DataFrame(columns=_COLS), {**meta, "observations": 0, "status": status}
        return df, {**meta, "observations": len(df), "status": status}

    try:
        if want_kic is not None:
            sr = lk.search_lightcurve(f"KIC {want_kic}", mission="Kepler", author="Kepler",
                                      exptime=exptime, quarter=quarter)
        else:
            coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
            sr = lk.search_lightcurve(coord, radius=float(radius_arcsec) * u.arcsec,
                                      mission="Kepler", author="Kepler",
                                      exptime=exptime, quarter=quarter)
        if len(sr) == 0:
            return result("no_data")
        kics = np.array([_kic_number(x) for x in np.asarray(sr.target_name).astype(str)])
    except Exception as e:
        meta["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        return result("search_failed")

    cands = sorted(set(kics.tolist()))
    meta["candidate_kic_ids"] = cands
    if want_kic is not None:
        if want_kic not in cands:
            return result("no_data")
        kic = want_kic
    else:
        if len(cands) != 1:
            return result("ambiguous_match")
        kic = cands[0]
        dist = np.asarray(sr.distance.to_value(u.arcsec), float)
        meta["match_sep_arcsec"] = float(np.nanmin(dist))
    meta["kic"] = kic

    uris = (np.asarray(sr.table["dataURI"]).astype(str) if "dataURI" in sr.table.colnames
            else np.arange(len(sr)).astype(str))
    frames, releases, seen = [], set(), set()
    for i in np.flatnonzero(kics == kic):
        uri = uris[i]
        if uri in seen:                        # repeated archive record: never duplicate rows
            continue
        seen.add(uri)
        try:
            lc = sr[int(i)].download(quality_bitmask=quality_bitmask, download_dir=download_dir)
            if lc is None:
                raise ValueError("download returned no light curve")
            if _kic_number(lc.meta.get("KEPLERID", -1)) != kic:
                raise ValueError("downloaded file belongs to a different KIC target")
            if lc.time.scale != "tdb" or int(lc.meta.get("BJDREFI", _BKJD_REF)) != _BKJD_REF:
                meta["n_rejected_time_frame"] += 1
                continue
            rel = lc.meta.get("DATA_REL")
            if rel is not None:
                releases.add(int(rel))
                if int(rel) < min_data_release:
                    meta["n_rejected_old_release"] += 1
                    continue
            unit = lc[flux_col].unit
            if unit is None or not unit.is_equivalent(_E_PER_S):
                raise ValueError(f"{flux_col} unit {unit} is not e-/s")
            scale = unit.to(_E_PER_S)
            mode = str(lc.meta.get("OBSMODE", "")).strip().lower()
            cadence = _OBSMODE.get(mode, int(round(float(sr.exptime[int(i)].value))))
            pdf = lc.to_pandas()
            bjd = np.asarray(lc.time.tdb.jd, float)            # BKJD + 2454833, TDB, mid-cadence
            f = pdf[flux_col].to_numpy(float) * scale
            e = pdf[f"{flux_col}_err"].to_numpy(float) * scale
            good = np.isfinite(bjd) & np.isfinite(f) & np.isfinite(e) & (e > 0)
            if not good.any():
                continue
            filt = f"kepler-{cadence}"
            frames.append(pd.DataFrame({
                "BJD":             bjd[good],
                "Target_flux":     f[good] * KEPLER_TO_UJY,
                "Target_flux_err": e[good] * KEPLER_TO_UJY,
                "Filter":          filt,
            }))
            nfrm, itime = lc.meta.get("NUM_FRM"), lc.meta.get("INT_TIME")
            products.append({
                "quarter": lc.meta.get("QUARTER"), "filter": filt, "data_release": rel,
                "file_version": lc.meta.get("FILEVER"), "rows": int(good.sum()),
                "cadence_s": float(lc.meta["TIMEDEL"]) * 86400 if lc.meta.get("TIMEDEL") else None,
                "integration_s": float(itime) * int(nfrm) if itime and nfrm else None,
                "bjd_min": float(bjd[good].min()), "bjd_max": float(bjd[good].max()),
                "filename": str(sr.table["productFilename"][i]) if "productFilename" in sr.table.colnames else uri})
        except Exception as e:
            failures.append({"data_uri": uri, "error": f"{type(e).__name__}: {str(e)[:300]}"})

    meta["data_releases"] = sorted(releases)
    if not frames:
        if meta["n_rejected_old_release"]:
            return result("old_data_release")
        if meta["n_rejected_time_frame"]:
            return result("unexpected_time_frame")
        return result("download_failed" if failures else "filtered_out")
    out = pd.concat(frames, ignore_index=True).sort_values("BJD", kind="stable").reset_index(drop=True)
    return result("partial_success" if failures else "success", out)
