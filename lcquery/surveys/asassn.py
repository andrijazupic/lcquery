import numpy as np
import pandas as pd
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time

_COLS = ["BJD", "Target_flux", "Target_flux_err", "Filter"]

# ASAS-SN is a 5-station network (Hawaii, Texas, SAAO, CTIO x2). Single
# representative site (CTIO); the topocentric term is <21 ms, far below the
# ~minute cadence and the ~tens-of-seconds HJD->BJD terms.
CTIO = EarthLocation.from_geodetic(
    lon=-70.8065 * u.deg, lat=-30.1692 * u.deg, height=2207.0 * u.m)

# pyasassn 'flux' is mJy on the AB scale (verified: flux / 10**(-0.4*mag) = 3631 Jy),
# so flux_uJy = flux_mJy * 1e3 is AB and mag_AB = 23.9 - 2.5*log10(flux_uJy) exactly.
_MJY_TO_UJY = 1.0e3
_BANDS = {"g": "asassn-g", "v": "asassn-v"}

_CLIENT = None
def _client():
    global _CLIENT
    if _CLIENT is None:
        from pyasassn.client import SkyPatrolClient
        _CLIENT = SkyPatrolClient(verbose=False)
    return _CLIENT


def fetch_asassn_lc(source_id, ra, dec, radius_arcsec=5.0, clean=True):
    """
    ASAS-SN Sky Patrol light curve for one source, via the pyasassn client.

    time   : BJD_TDB (full JD). ASAS-SN serves HJD(UTC); we undo the
             heliocentric light-travel time, then apply the barycentric+TDB
             correction (identical to the other surveys' final step).
    flux   : AB micro-Jy = flux(mJy)*1e3 (ASAS-SN flux is 3631-Jy AB-scaled).
             g shares the AB g scale of your other surveys; V is Johnson V
             (treated as AB, ~0.02-0.04 mag Vega offset, like CRTS), a distinct band.
    filter : "asassn-g" / "asassn-v".

    clean  : True -> good images only (quality == 'G'); False -> keep all.
             Non-detections (mag_err > 99) are always dropped.
    """
    sid = int(source_id)
    def result(status, df=None):
        if df is None or len(df) == 0:
            return (pd.DataFrame(columns=_COLS),
                    {"survey": "ASAS-SN", "source_id": sid, "observations": 0, "status": status})
        return (df, {"survey": "ASAS-SN", "source_id": sid,
                     "observations": len(df), "status": "success"})

    client = _client()
    rdeg = radius_arcsec / 3600.0

    # 1. nearest ASAS-SN target in the cone (peek without downloading)
    try:
        cat = client.cone_search(ra_deg=float(ra), dec_deg=float(dec),
                                 radius=rdeg, catalog="master_list", download=False)
    except Exception as e:
        return result(f"search_failed: {type(e).__name__}: {e}")
    if cat is None or len(cat) == 0 or "asas_sn_id" not in cat.columns:
        return result("no_match")
    if len(cat) > 1 and {"ra_deg", "dec_deg"}.issubset(cat.columns):
        sep2 = ((cat["ra_deg"] - float(ra)) * np.cos(np.radians(float(dec)))) ** 2 \
               + (cat["dec_deg"] - float(dec)) ** 2
        nearest = int(cat.loc[sep2.idxmin(), "asas_sn_id"])
    else:
        nearest = int(cat["asas_sn_id"].iloc[0])

    # 2. download the cone's curves, then pick the nearest target's light curve
    try:
        lcs = client.cone_search(ra_deg=float(ra), dec_deg=float(dec),
                                 radius=rdeg, catalog="master_list", download=True)
    except Exception as e:
        return result(f"download_failed: {type(e).__name__}: {e}")
    try:
        df = lcs[nearest].data
    except Exception:
        try:                                          # fall back to first curve in cone
            ids = list(lcs.stats().index)
            df = lcs[ids[0]].data if ids else None
        except Exception:
            df = None
    if df is None or len(df) == 0:
        return result("no_data")

    tcol = next((c for c in ["jd", "hjd", "HJD"] if c in df.columns), None)
    if tcol is None or "flux" not in df.columns or "flux_err" not in df.columns:
        return result("missing_columns")

    df = df.copy()
    df["flux"] = pd.to_numeric(df["flux"], errors="coerce")
    df["flux_err"] = pd.to_numeric(df["flux_err"], errors="coerce")
    df = df.dropna(subset=[tcol, "flux", "flux_err"])
    if "mag_err" in df.columns:                       # drop non-detections (mag_err > 99)
        df = df[pd.to_numeric(df["mag_err"], errors="coerce") < 99]
    df = df[df["flux_err"] > 0]
    if clean and "quality" in df.columns:             # good images only
        df = df[df["quality"].astype(str) == "G"]
    if len(df) == 0:
        return result("filtered_out")
    df = df.reset_index(drop=True)

    # band label from the filter column (g / V)
    fcol = next((c for c in ["phot_filter", "filter", "Filter"] if c in df.columns), None)
    if fcol is not None:
        band = (df[fcol].astype(str).str.lower().map(_BANDS)
                .fillna("asassn-" + df[fcol].astype(str))).to_numpy()
    else:
        band = np.array(["asassn"] * len(df))         # <-- VERIFY: V/g not separated here

    # time: HJD(UTC) -> undo heliocentric LTT -> BJD_TDB
    coord = SkyCoord(float(ra) * u.deg, float(dec) * u.deg, frame="icrs")
    hjd = df[tcol].to_numpy(float)
    t0 = Time(hjd, format="jd", scale="utc", location=CTIO)
    t_utc = t0 - t0.light_travel_time(coord, kind="heliocentric")
    bjd_tdb = (t_utc.tdb + t_utc.light_travel_time(coord, kind="barycentric")).jd

    # flux: mJy (AB) -> uJy
    flux = df["flux"].to_numpy(float) * _MJY_TO_UJY
    flux_err = df["flux_err"].to_numpy(float) * _MJY_TO_UJY

    out = pd.DataFrame({
        "BJD":             bjd_tdb,
        "Target_flux":     flux,
        "Target_flux_err": flux_err,
        "Filter":          band,
    }).sort_values("BJD").reset_index(drop=True)
    return result(None, out)