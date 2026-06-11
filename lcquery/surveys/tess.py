import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

# e-/s -> in-band AB uJy.  F0 = SVO TESS/TESS.Red Vega zero point (Jy);
# ZP_inst from TESS FAQ (Tmag = ZP_inst - 2.5*log10(flux[e-/s])).
TESS_F0_JY, TESS_ZP_INST = 2631.88, 20.44
TESS_TO_UJY = TESS_F0_JY * 1e6 * 10 ** (-0.4 * TESS_ZP_INST)   # ~17.5 uJy per e-/s


def fetch_tess_lc(source_id, ra, dec, radius_arcsec=2.0,
                  authors=("SPOC", "TESS-SPOC"), flux_col="pdcsap_flux"):
    """
    TESS light curve(s) for one source via lightkurve / MAST.

    time   : BJD_TDB (full JD). TESS times are already barycentric + TDB
             (BTJD = BJD_TDB - 2457000); .jd gives BJD_TDB directly.
    flux   : TESS-band AB flux density in micro-Jy (PDCSAP e-/s * TESS_TO_UJY).
             Same UNIT as BlackGEM uJy, but a different bandpass -> not directly
             comparable for the same star (colour term + SED differences).
    filter : "<pipeline> <cadence_s>", e.g. "SPOC 120", "TESS-SPOC 600".
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
                author = author[5:]          # "TESS-SPOC" -> "spoc", avoids tess-tess-spoc
            cadence = int(round(float(sr.exptime[i].value)))
            pdf = lc.to_pandas()
            pdf["bjd_tdb"] = lc.time.jd                       # already barycentric -> BJD_TDB
            pdf = pdf.dropna(subset=[flux_col, f"{flux_col}_err"])
            pdf = pdf[pdf[f"{flux_col}_err"] > 0]
            if pdf.empty:
                continue
            frames.append(pd.DataFrame({
                "BJD":             pdf["bjd_tdb"].to_numpy(float),
                "Target_flux":     pdf[flux_col].to_numpy(float) * TESS_TO_UJY,        # uJy
                "Target_flux_err": pdf[f"{flux_col}_err"].to_numpy(float) * TESS_TO_UJY,
                "Filter":          f"tess-{author}-{cadence}",
            }))
        except Exception:
            continue

    if not frames:
        return result("filtered_out")
    out = pd.concat(frames, ignore_index=True).sort_values("BJD").reset_index(drop=True)
    return result(None, out)