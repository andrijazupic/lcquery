import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

K2_F0_JY, K2_ZP_INST = 3241.90, 25.10
K2_TO_UJY = K2_F0_JY * 1e6 * 10 ** (-0.4 * K2_ZP_INST)        


def fetch_k2_lc(source_id, ra, dec, radius_arcsec=2.0,
                authors=("K2",), flux_col="pdcsap_flux"):
    """
    --------------------------------------------------------------------------------
    K2  (Kepler / K2 mission)
    --------------------------------------------------------------------------------
    Access. K2 light curves via lightkurve / MAST
    (search_lightcurve(coord, mission="K2", author=["K2"])) within a small cone,
    downloaded per product. Filter label is the cadence in seconds (e.g. k2-1800,
    k2-60). Forced (PDCSAP aperture) photometry from the Kepler spacecraft -- a flux
    every cadence, eclipse-complete. Nominal cadences: long ~1766 s (270 coadds;
    labelled 1800), short ~58.85 s (9 coadds; labelled 60).

    Time -> BJD_TDB. K2 times are stored as BKJD (Barycentric Kepler Julian Date
    = BJD_TDB - 2454833.0; offset 2454833.0 = UTC 2009-01-01 12:00:00), already
    corrected to the solar-system barycentre and on the TDB scale. lightkurve
    represents this as an astropy Time (format bkjd, scale tdb), so lc.time.jd yields
    the full BJD_TDB directly. No barycentric correction and no scale conversion are
    applied -- both are already present in the Kepler/K2 data.

    Flux -> uJy (Vega). The PDCSAP flux is in e-/s, converted to an in-band flux
    density via Target_flux[uJy] = flux[e-/s] * K2_TO_UJY, where
    K2_TO_UJY = F0 * 1e6 * 10^(-0.4*ZP_inst) ~ 0.30, with ZP_inst = 25.10 and
    F0 = 3241.90 Jy (SVO Kepler-band Vega zero-point flux). The result is a
    Vega-system in-band flux density (Fnu = F0_Vega * 10^(-0.4*Kp)), not AB; the
    metadata unit is uJy_Vega. NOTE on ZP_inst: 25.10 is the PRE-FLIGHT GO-site value
    (a Kp=12 G2V star giving 1.74e5 e-/s implies ZP = 25.10 exactly). The in-flight
    empirical zero points, fitted against EPIC Kp magnitudes, cluster a little higher
    -- ~25 to 25.3 for full-aperture/PDCSAP photometry (Aigrain et al. 2015; Libralato
    et al.). The difference is a constant ~20% in absolute flux on a bandpass that
    matches no other survey, so it is immaterial for cross-survey period work; the
    absolute scale (and the Vega/AB choice) is documented as approximate. To sit
    exactly on the EPIC Kp scale, set ZP_inst = 25.3. Errors (PDCSAP_FLUX_ERR, also
    e-/s) use the same factor.

    Cleaning. Download with quality_bitmask="default", applying the standard Kepler/K2
    quality mask (excluding cadences flagged for attitude tweaks, safe mode, coarse
    pointing, Argabrightening, cosmic-ray hits in the aperture, manual exclude, etc.).
    Then drop NaN in flux/flux_err and require flux_err > 0. flux > 0 is deliberately
    NOT imposed: negative PDCSAP fluxes are valid measurements for faint sources and
    are retained to keep the noise distribution unbiased. Multiple products for one
    source are concatenated.
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "K2", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "K2", "source_id": sid,
                     "observations": len(df), "status": "success"})

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    try:
        sr = lk.search_lightcurve(coord, radius=radius_arcsec * u.arcsec,
                                  mission="K2", author=list(authors))
    except Exception:
        return result("search_failed")
    if len(sr) == 0:
        return result("no_data")

    frames = []
    for i in range(len(sr)):
        try:
            lc = sr[i].download(quality_bitmask="default")
            if lc is None:
                continue
            cadence = int(round(float(sr.exptime[i].value)))
            pdf = lc.to_pandas()
            pdf["bjd_tdb"] = lc.time.jd                     
            pdf = pdf.dropna(subset=[flux_col, f"{flux_col}_err"])
            pdf = pdf[pdf[f"{flux_col}_err"] > 0]
            if pdf.empty:
                continue
            frames.append(pd.DataFrame({
                "BJD":             pdf["bjd_tdb"].to_numpy(float),
                "Target_flux":     pdf[flux_col].to_numpy(float) * K2_TO_UJY,         
                "Target_flux_err": pdf[f"{flux_col}_err"].to_numpy(float) * K2_TO_UJY,
                "Filter":          f"k2-{cadence}",
            }))
        except Exception:
            continue

    if not frames:
        return result("filtered_out")
    out = pd.concat(frames, ignore_index=True).sort_values("BJD").reset_index(drop=True)
    return result(None, out)