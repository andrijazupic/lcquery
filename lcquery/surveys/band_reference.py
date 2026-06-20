# The light curves carry a `Filter` label;
# join it here to know exactly what each flux column means.
SURVEY_REFERENCE = {
    "ATLAS":     {"system": "AB",   "zp_Jy": 3631,    "phot": "forced (PSF)",               "timing": "topocentric UTC, mid-exposure (native)",      "exposure": "30 s",                            "cadence": "~1-2 d (4x30 s/visit over ~1 hr)",          "bands": "c,o"},
    "BlackGEM":  {"system": "AB",   "zp_Jy": 3631,    "phot": "forced (optimal)",           "timing": "topocentric UTC, mid-exposure (GPS)",         "exposure": "60 s",                            "cadence": "1 min-2 hr (program-dependent)",           "bands": "u,g,q,r,i,z"},
    "Gaia":      {"system": "AB",   "zp_Jy": 3631,    "phot": "per-transit (to G~21)",      "timing": "barycentric TCB, FoV-transit (native)",       "exposure": "4.42 s/CCD (~44 s transit)",      "cadence": "irregular; 106.5 min in-pair; ~40 transits (DR3)",     "bands": "g,bp,rp"},
    "J-VAR":     {"system": "AB",   "zp_Jy": 3631,    "phot": "detection (aperture)",       "timing": "topocentric UTC, start/mid unstated",         "exposure": "g33 r40 i34 J0395-87 J0515-40 J0660-135 J0861-160 s", "cadence": "~11 epochs/~1 yr (~12.7 min same-band; full visit ~40 min)", "bands": "g,r,i,j0395,j0515,j0660,j0861"},
    "NSC":       {"system": "AB",   "zp_Jy": 3631,    "phot": "detection (aperture)",       "timing": "topocentric UTC, start (DATE-OBS; +0.5exp)",  "exposure": "varies (archival)",               "cadence": "irregular (archival)",                     "bands": "u,g,r,i,z,y,vr"},
    "SkyMapper": {"system": "AB",   "zp_Jy": 3631,    "phot": "detection (PSF)",            "timing": "topocentric UTC, start (+0.5exp applied)",    "exposure": "100 s (Main), 5-40 s (u40 v20 g5 r5 i10 z20 s)",  "cadence": "irregular/relaxed",                        "bands": "u,v,g,r,i,z"},
    "ZTF":       {"system": "AB",   "zp_Jy": 3631,    "phot": "detection (PSF, matchfile)", "timing": "topocentric UTC, start (+0.5exp applied)",    "exposure": "30 s (public); 30-300 s (private)","cadence": "~2-3 d (public); ~40 s-min (HC programs)",      "bands": "g,r,i"},
    "CRTS":      {"system": "Vega", "zp_Jy": 3636,    "phot": "detection (aperture)",       "timing": "topocentric UTC, start/mid unstated (<=15 s)","exposure": "30 s",                            "cadence": "~2 weeks (2-4 wk); 4 exp/visit over ~30 min", "bands": "clear (≈V)"},
    "OGLE":      {"system": "Vega", "zp_Jy": None,    "phot": "detection (PSF/DIA)",        "timing": "heliocentric UTC (HJD), start/mid unstated",  "exposure": "100 s I, 150 s V; 25 s disk",     "cadence": "19-60 min (bulge) to 1-3 d (disk/MC)",     "bands": "i,v"},
    "K2":        {"system": "Vega", "zp_Jy": 3241.9,  "phot": "forced (PDCSAP)",            "timing": "barycentric TDB, mid-cadence (native)",       "exposure": "6.02 s/frame",                    "cadence": "58.85 s (SC) / 1766 s (LC)",               "bands": "Kepler"},
    "TESS":      {"system": "Vega", "zp_Jy": 2631.88, "phot": "forced (PDCSAP)",            "timing": "barycentric TDB, mid-cadence (native)",       "exposure": "2 s/frame",                       "cadence": "20/120 s (TPF), 200/600/1800 s (FFI)",     "bands": "TESS"},
    "ASAS-SN":   {"system": "per-band","zp_Jy": None, "phot": "forced (aperture)",          "timing": "heliocentric UTC (HJD), field-centre ~200 s", "exposure": "90 s (x3/epoch)",                 "cadence": "~nightly (g); ~2-3 d (V)",                 "bands": "g (AB), v (Vega)"},
   #"CHEOPS":    {"system": "UNAUDITED","zp_Jy": None,"phot": "forced (aperture)",          "timing": "?",                                           "exposure": "?",                               "cadence": "?",                                        "bands": "white"},
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