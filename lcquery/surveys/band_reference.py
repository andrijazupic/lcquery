# The light curves carry a `Filter` label;
# join it here to know exactly what each flux column means.
SURVEY_REFERENCE = {
    "ATLAS":     {"system": "AB",   "zp_Jy": 3631,    "phot": "forced (PSF)",               "timing": "mid-exposure (native)",                        "exposure": "30 s",                "bands": "c,o"},
    "BlackGEM":  {"system": "AB",   "zp_Jy": 3631,    "phot": "forced (optimal)",           "timing": "mid (GPS-exact)",                              "exposure": "~60 s",               "bands": "u,g,q,r,i,z"},
    "Gaia":      {"system": "AB",   "zp_Jy": 3631,    "phot": "per-transit (to G~21)",      "timing": "barycentric (native)",                         "exposure": "per-transit",         "bands": "g,bp,rp"},
    "J-VAR":     {"system": "AB",   "zp_Jy": 3631,    "phot": "detection (aperture)",       "timing": "(±tens s, start/mid uncertain)",               "exposure": "varies",              "bands": "g,r,i,j0395,j0515,j0660,j0861"},
    "NSC":       {"system": "AB",   "zp_Jy": 3631,    "phot": "detection (aperture)",       "timing": "mid-exposure",                                 "exposure": "varies",              "bands": "u,g,r,i,z,y,vr"},
    "SkyMapper": {"system": "AB",   "zp_Jy": 3631,    "phot": "detection (PSF)",            "timing": "mid-exposure",                                 "exposure": "100 s (Main)",        "bands": "u,v,g,r,i,z"},
    "ZTF":       {"system": "AB",   "zp_Jy": 3631,    "phot": "detection (PSF, matchfile)", "timing": "mid-exposure",                                 "exposure": "30 s",                "bands": "g,r,i"},
    "CRTS":      {"system": "Vega", "zp_Jy": 3636,    "phot": "detection (aperture)",       "timing": "(±15 s, start/mid uncertain)",                 "exposure": "30 s",                "bands": "clear (≈V)"},
    "OGLE":      {"system": "Vega", "zp_Jy": None,    "phot": "detection (PSF/DIA)",        "timing": "HJD (helio round-trip; start/mid unverified)", "exposure": "varies",              "bands": "i,v"},
    "K2":        {"system": "Vega", "zp_Jy": 3241.9,  "phot": "forced (PDCSAP)",            "timing": "barycentric mid (native)",                     "exposure": "60/1800 s (cadence)", "bands": "Kepler"},
    "TESS":      {"system": "Vega", "zp_Jy": 2631.88, "phot": "forced (PDCSAP)",            "timing": "barycentric mid (native)",                     "exposure": "20-1800 s (cadence)", "bands": "TESS"},
    "ASAS-SN":   {"system": "per-band","zp_Jy": None, "phot": "forced (aperture)",          "timing": "HJD, ~200 s field-centre floor",               "exposure": "~90 s",               "bands": "g (AB), v (Vega)"},
   #"CHEOPS":    {"system": "UNAUDITED","zp_Jy": None,"phot": "forced (aperture)",          "timing": "?",                                            "exposure": "?",                   "bands": "white"},
}
# Per-band overrides where bands differ from the survey default (only these need them).
BAND_OVERRIDE = {
    "asassn-g": {"system": "AB",   "zp_Jy": 3631},
    "asassn-v": {"system": "Vega", "zp_Jy": 3836.3, "note": "Johnson-V-like, +0.060 mag from AB; measured (slope 1.000, zero scatter)"},
    "ogle-i":   {"system": "Vega", "zp_Jy": 2416, "note": "Cousins I"},
    "ogle-v":   {"system": "Vega", "zp_Jy": 3636, "note": "Johnson V (≈AB)"},
}

_PREFIX = {"asassn":"ASAS-SN","atlas":"ATLAS","blackgem":"BlackGEM","crts":"CRTS",
           "gaia":"Gaia","jvar":"J-VAR","k2":"K2","tess":"TESS","nsc":"NSC",
           "ogle":"OGLE","skymapper":"SkyMapper","ztf":"ZTF","cheops":"CHEOPS"}

def _survey_of(filter_label):
    return _PREFIX.get(filter_label.split("-")[0])

def band_info(filter_label):
    survey = _survey_of(filter_label)
    info = dict(SURVEY_REFERENCE.get(survey, {}))
    info.update(BAND_OVERRIDE.get(filter_label, {}))   # band override wins
    info.update(filter=filter_label, survey=survey)
    return info

def write_band_reference(path):
    import pandas as pd
    rows = [{"survey": s, **v} for s, v in SURVEY_REFERENCE.items()]
    rows += [{"survey": _survey_of(b), "filter": b, **v} for b, v in BAND_OVERRIDE.items()]
    pd.DataFrame(rows).to_csv(path, index=False)