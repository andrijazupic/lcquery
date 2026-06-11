# lcquery

**Multi-survey time-domain light curves on a single, physically uniform schema.**

Given one source's Gaia DR3 `source_id` and ICRS coordinates, `lcquery` queries a dozen
photometric archives, standardizes every light curve to **BJD_TDB** time and
**micro-Jansky** flux, and writes one CSV per source per survey. The point is
comparability: a ZTF point, a Gaia point, and an ASAS-SN point all land on the same time
and flux axes, so you can overplot and model them together instead of wrangling each
survey's native conventions one at a time.

## Surveys

`lcquery` covers twelve surveys (CHEOPS is implemented but disabled by default — see Notes):

| Survey | Filter labels | System | Queried by |
|---|---|---|---|
| BlackGEM | `blackgem-u/g/q/r/i/z` | AB | Gaia DR3 ID |
| TESS | `tess-spoc-20/120/200/600/1800` † | AB | position (cone) |
| K2 | `k2-60/1800` † | AB | position (cone) |
| ZTF | `ztf-g/r/i` | AB | position (cone) |
| CRTS | `crts-clear` (≈ Johnson V) | **Vega** | position (cone) |
| NSC | `nsc-u/g/r/i/z/y/vr` | AB | position (cone) |
| ATLAS | `atlas-c/o` | AB | exact position (forced) |
| SkyMapper | `skymapper-u/v/g/r/i/z` | AB | position (cone) |
| J-VAR | `jvar-g/r/i` + `jvar-j0395/j0515/j0660/j0861` | AB | position (cone) |
| ASAS-SN | `asassn-g/v` | AB | position (cone) |
| OGLE | `ogle-i/v` | **Vega** | position (cone) |
| Gaia | `gaia-g/bp/rp` | AB | Gaia DR3 ID |

† TESS and K2 are each a single broad band; the trailing number is the exposure
cadence in **seconds**, so `tess-spoc-120` and `tess-spoc-1800` are the same passband
sampled at 2 min vs 30 min.

## How the light curves are standardized

- **Time** → Barycentric Julian Date in the TDB standard (`BJD_TDB`, full JD). Each
  survey's native timestamps (topocentric MJD, heliocentric JD, TCB, …) are converted to
  a common barycentric-TDB frame, so epochs from different surveys are directly aligned.
- **Flux** → `Target_flux` and `Target_flux_err` in **micro-Janskys (µJy)**. Most surveys
  (TESS, K2, ZTF, NSC, ATLAS, SkyMapper, J-VAR, ASAS-SN, Gaia) are on the **AB** system;
  **CRTS and OGLE are on the Vega system**.


## Installation

Requires Python 3.9+.

```bash
git clone https://github.com/andrijazupic/lcquery.git
cd lcquery
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"     # all surveys
```

Use `pip install -e .` for the ten core surveys only; `[all]` adds the heavier optional
clients for **BlackGEM** (Google BigQuery) and **CHEOPS** (DACE).

## Authentication

Most surveys are public and need no setup. Three have requirements:

**ATLAS** — register at <https://fallingstar-data.com/forcedphot/>, then fetch your token
once:

```python
import requests
r = requests.post("https://fallingstar-data.com/forcedphot/api-token-auth/",
                  data={"username": "YOUR_USERNAME", "password": "YOUR_PASSWORD"})
print(r.json()["token"])
```

Paste the printed token into `config.yaml` under `credentials: → atlas_token:` (or export
it as the `ATLASFORCED_SECRET_KEY` environment variable).

**Gaia** — optional. You can add `gaia_user` / `gaia_password` to `config.yaml`, but
epoch-photometry queries work fine without logging in.

**BlackGEM** — requires Google Cloud access. Authenticate once with the Google Cloud CLI
(`gcloud auth application-default login`); no token goes in the config.

## Quick start

```python
from lcquery import get_all_lightcurves

# SS Cygni
source_id = 3398501003057791360
ra  = 87.2655554543
dec = 19.0727344751

get_all_lightcurves(source_id, ra, dec, verbose=True)
```

This writes CSVs to disk (it does not return a DataFrame).

## Output

Light curves are written under `lightcurves/`, one folder per survey, each CSV named by the
source's Gaia DR3 ID:

```
lightcurves/
├── ZTF/3398501003057791360.csv
├── Gaia/3398501003057791360.csv
├── ATLAS/3398501003057791360.csv
├── …
└── query_metadata.csv          # one row per (source, survey): status, counts, settings used
```

Each CSV has four columns — `BJD`, `Target_flux`, `Target_flux_err`, `Filter` — preceded by
a commented `#` header recording the survey, `source_id`, coordinates, and column units.
Because of that header, read the files back with `comment="#"`:

```python
import pandas as pd
df = pd.read_csv("lightcurves/ZTF/4668163021600295552.csv", comment="#")
```

`lcquery` caches: if a CSV already exists it is skipped on the next run. Pass
`overwrite=True` (or set it in the config) to re-download.

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit it.
Per survey you can set:

- `enabled` — turn a survey on or off.
- `radius_arcsec` — the cone-search radius. *(BlackGEM, Gaia,
  and ATLAS take no radius — see Notes.)*
- `clean` — when true, discards low-quality points using cuts adapted from each survey's own
  documentation and literature (quality flags, blend/saturation limits, detection thresholds).

## Notes

- **Gaia and BlackGEM are queried by identity, not position.** The `source_id` you pass must
  be a **Gaia DR3 source_id** — it is used to cross-match the source directly, so ra/dec are
  used only for the time correction, never for the lookup. Pass an ID from another system and
  these two return no data.
- **ATLAS has no search radius** because it does *forced photometry*: you hand the server an
  exact (ra, dec) and it measures the flux at that precise point on every difference image,
  whether or not anything is catalogued there. There is no catalogue to search, so there is
  nothing for a radius to bound — the coordinates *are* the query.
- **"No data" is often real coverage, not a bug.** Surveys have footprints and selection
  functions: southern surveys (SkyMapper, OGLE) won't have northern targets, CRTS excludes the
  Galactic plane, OGLE only contains catalogued variables, and so on. The `status` column in
  `query_metadata.csv` records *why* each survey came back empty.
- **The first OGLE query is slow.** OGLE has no positional API, so on first use `lcquery`
  crawls and caches the full OGLE Collection of Variable Stars (~1.2M entries) to
  `lcquery/surveys/ogle_ocvs_master.csv`. This one-time build takes a while; every query
  afterward reads the cache.
- **CHEOPS** is implemented but disabled by default (DACE service is down).