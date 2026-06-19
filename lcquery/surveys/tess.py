import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

TESS_F0_JY, TESS_ZP_INST = 2631.88, 20.44
TESS_TO_UJY = TESS_F0_JY * 1e6 * 10 ** (-0.4 * TESS_ZP_INST)  


def fetch_tess_lc(source_id, ra, dec, radius_arcsec=2.0,
                  authors=("SPOC", "TESS-SPOC"), flux_col="pdcsap_flux"):
    """
    --------------------------------------------------------------------------------
    TESS  (Transiting Exoplanet Survey Satellite)
    --------------------------------------------------------------------------------
    Access. TESS light curves via lightkurve / MAST
    (search_lightcurve(coord, mission="TESS", author=["SPOC","TESS-SPOC"])),
    downloaded per product. Filter label is tess-spoc-<cadence_s> -- both the SPOC
    pipeline (120 s / 20 s, from target-pixel files) and the TESS-SPOC FFI HLSP
    (1800 s primary mission, 600 s EM1, 200 s EM2) are labelled spoc, distinguished
    only by cadence (e.g. tess-spoc-120, tess-spoc-600). Forced (PDCSAP aperture)
    photometry -- a flux every cadence, eclipse-complete.

    Time -> BJD_TDB. TESS times are stored as BTJD (Barycentric TESS Julian Date
    = BJD_TDB - 2457000.0), already corrected to the solar-system barycentre and on
    the TDB scale. lightkurve carries this as an astropy Time (format btjd, scale
    tdb), so lc.time.jd yields the full BJD_TDB directly. No barycentric correction
    and no scale conversion are applied -- both are already present in the TESS data.

    Flux -> uJy (Vega). PDCSAP flux is in e-/s, converted via
    Target_flux[uJy] = flux[e-/s] * TESS_TO_UJY, where
    TESS_TO_UJY = F0 * 1e6 * 10^(-0.4*ZP_inst) ~ 17.5, with ZP_inst = 20.44 (TESS
    instrumental count-rate zero point, Tmag = 20.44 - 2.5*log10(e-/s); TESS
    Instrument Handbook) and F0 = 2631.88 Jy (SVO TESS.Red Vega zero-point flux). The
    result is a Vega-system in-band flux density (Fnu = F0_Vega * 10^(-0.4*Tmag)),
    not AB; the metadata unit is uJy_Vega. NOTE on F0: 2631.88 Jy (SVO TESS.Red Vega)
    is one of two legitimate conventions -- the MIT TESS pipeline instead uses 2416 Jy
    (Cousins I) for its physical conversion. The two are not interchangeable but
    differ only by a constant scale; the SVO TESS.Red value is internally consistent
    for an in-band flux density and is the documented choice here. As a constant
    factor it does not affect variability, and the TESS bandpass matches no other
    survey, so the absolute scale and the Vega/AB choice are immaterial for
    cross-survey period work. Errors (PDCSAP_FLUX_ERR, also e-/s) use the same factor.

    Cleaning. Download with quality_bitmask="default", applying the standard TESS
    quality mask (attitude tweaks, safe mode, coarse pointing, Earth/Moon in the FOV,
    cosmic rays, stray light, manual exclude, etc.). Then drop NaN in flux/flux_err
    and require flux_err > 0; flux > 0 is deliberately NOT imposed, so valid negative
    PDCSAP fluxes on faint sources are retained to keep the noise distribution
    unbiased. Multiple sectors/products for one source are concatenated.
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "TESS", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "TESS", "source_id": sid,
                     "observations": len(df), "status": "success"})

    coord = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")
    try:
        sr = lk.search_lightcurve(coord, radius=radius_arcsec * u.arcsec,
                                  mission="TESS", author=list(authors))
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
            author  = str(sr.author[i]).lower().replace(" ", "-")
            if author.startswith("tess-"):
                author = author[5:]        
            cadence = int(round(float(sr.exptime[i].value)))
            pdf = lc.to_pandas()
            pdf["bjd_tdb"] = lc.time.jd                     
            pdf = pdf.dropna(subset=[flux_col, f"{flux_col}_err"])
            pdf = pdf[pdf[f"{flux_col}_err"] > 0]
            if pdf.empty:
                continue
            frames.append(pd.DataFrame({
                "BJD":             pdf["bjd_tdb"].to_numpy(float),
                "Target_flux":     pdf[flux_col].to_numpy(float) * TESS_TO_UJY,       
                "Target_flux_err": pdf[f"{flux_col}_err"].to_numpy(float) * TESS_TO_UJY,
                "Filter":          f"tess-{author}-{cadence}",
            }))
        except Exception:
            continue

    if not frames:
        return result("filtered_out")
    out = pd.concat(frames, ignore_index=True).sort_values("BJD").reset_index(drop=True)
    return result(None, out)