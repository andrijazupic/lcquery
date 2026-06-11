import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u
from astropy.coordinates import SkyCoord

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

# e-/s -> in-band AB uJy.  F0 = SVO Kepler/Kepler.K Vega zero point (Jy);
# ZP_inst ~25.1 from Kepler Instrument Handbook (Kp=12 -> 1.74e5 e-/s).
# The Kepler count-rate ZP is uncertain at ~0.2 mag (~20% on the factor).
K2_F0_JY, K2_ZP_INST = 3241.90, 25.10
K2_TO_UJY = K2_F0_JY * 1e6 * 10 ** (-0.4 * K2_ZP_INST)         # ~0.30 uJy per e-/s


def fetch_k2_lc(source_id, ra, dec, radius_arcsec=2.0,
                authors=("K2",), flux_col="pdcsap_flux"):
    """
    K2 light curve(s) for one source via lightkurve / MAST.

    time   : BJD_TDB (full JD). K2 times are already barycentric + TDB
             (BKJD = BJD_TDB - 2454833); .jd gives BJD_TDB directly.
    flux   : Kepler-band AB flux density in micro-Jy (PDCSAP e-/s * K2_TO_UJY).
             Same UNIT as BlackGEM uJy, different bandpass (see TESS note).
    filter : cadence in seconds only, e.g. "1800" or "60".
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
            pdf["bjd_tdb"] = lc.time.jd                       # already barycentric -> BJD_TDB
            pdf = pdf.dropna(subset=[flux_col, f"{flux_col}_err"])
            pdf = pdf[pdf[f"{flux_col}_err"] > 0]
            if pdf.empty:
                continue
            frames.append(pd.DataFrame({
                "BJD":             pdf["bjd_tdb"].to_numpy(float),
                "Target_flux":     pdf[flux_col].to_numpy(float) * K2_TO_UJY,         # uJy
                "Target_flux_err": pdf[f"{flux_col}_err"].to_numpy(float) * K2_TO_UJY,
                "Filter":          f"k2-{cadence}",
            }))
        except Exception:
            continue

    if not frames:
        return result("filtered_out")
    out = pd.concat(frames, ignore_index=True).sort_values("BJD").reset_index(drop=True)
    return result(None, out)