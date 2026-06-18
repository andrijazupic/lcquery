# lcquery

**Multi-survey time-domain light curves on a single, physically uniform schema.**

Given one source's Gaia DR3 `source_id` and ICRS coordinates, `lcquery` queries a dozen
photometric archives, standardizes every light curve to **BJD_TDB** time and
**micro-Jansky** flux, and writes one CSV per source per survey. The point is
comparability: a ZTF point, a Gaia point, and an ASAS-SN point all land on the same time
and flux axes, so you can overplot and model them together instead of wrangling each
survey's native conventions one at a time.

## Surveys

`lcquery` covers twelve surveys:

| Survey | Filter labels | System | Photometry | Queried by |
|---|---|---|---|---|
| BlackGEM | `blackgem-u/g/q/r/i/z` | AB | detection (optimal) | Gaia DR3 ID |
| TESS | `tess-spoc-20/120/200/600/1800` † | **Vega** | forced (PDCSAP) | position (cone) |
| K2 | `k2-60/1800` † | **Vega** | forced (PDCSAP) | position (cone) |
| ZTF | `ztf-g/r/i` | AB | detection (PSF, matchfile) | position (cone) |
| CRTS | `crts-clear` (≈ Johnson V) | **Vega** | detection (aperture) | position (cone) |
| NSC | `nsc-u/g/r/i/z/y/vr` | AB | detection (aperture) | position (cone) |
| ATLAS | `atlas-c/o` | AB | forced (PSF) | exact position (forced) |
| SkyMapper | `skymapper-u/v/g/r/i/z` | AB | detection (PSF) | position (cone) |
| J-VAR | `jvar-g/r/i` + `jvar-j0395/j0515/j0660/j0861` | AB | detection (aperture) | position (cone) |
| ASAS-SN | `asassn-g/v` | **mixed** ‡ | forced (aperture) | position (cone) |
| OGLE | `ogle-i/v` | **Vega** | detection (PSF/DIA) | position (cone) |
| Gaia | `gaia-g/bp/rp` | AB | per-transit (to G≈21) | Gaia DR3 ID |

† TESS and K2 are each a single broad band; the trailing number is the exposure cadence in
**seconds**, so `tess-spoc-120` and `tess-spoc-1800` are the same passband sampled at 2 min
vs 30 min.

‡ ASAS-SN is the one survey whose two bands sit on different systems: **`asassn-g` is AB,
`asassn-v` is Vega** (a Johnson-V-like band, empirically ≈ +0.06 mag from AB). Its files
therefore carry both systems — resolve per band via the `Filter` column and
`band_reference.csv` (below), and its `query_metadata.csv` flux unit reads `uJy_mixed`.

## How light curves are standardized

### Time → `BJD_TDB`

Every timestamp is converted to **Barycentric Julian Date on the TDB scale** (full JD), so
epochs from different surveys are directly comparable. Surveys arrive in one of three native
conventions, each handled differently:

- **Topocentric (observer-frame) MJD in UTC** — ATLAS, BlackGEM, CRTS, J-VAR, NSC,
  SkyMapper, ZTF. Converted UTC → TDB and given a barycentric light-travel correction
  computed at the source position.
- **Heliocentric JD (HJD)** — ASAS-SN, OGLE. The heliocentric correction is first *undone*
  (recovering topocentric UTC), then redone to the barycentre. Because the HJD↔BJD difference
  varies through the year (up to a few seconds), this round trip — not a constant offset — is
  what yields a correct `BJD_TDB`.
- **Already barycentric** — Gaia (TCB transit time), K2 (BKJD), TESS (BTJD). No light-travel
  correction is applied; only the time scale is put onto TDB (a ~20 s shift for Gaia's TCB).

Where a survey reports the **exposure start** and the exposure length is known, the pipeline
adds half the exposure to place the timestamp at **mid-exposure**, matching the rest of the
pipeline (ATLAS, NSC, SkyMapper, ZTF; BlackGEM is already mid-exposure via GPS timing). NSC's
exposure time is read per-row, since it varies across its constituent surveys.

Absolute-timing floors worth knowing: **ASAS-SN's field-centre HJD is good only to ~200 s**
unless recomputed with its aperture pipeline, and **CRTS / J-VAR carry a start-vs-mid
ambiguity of order 10–60 s**. The per-survey timing convention is recorded in `band_reference.csv`.

### Flux → µJy

`Target_flux` and `Target_flux_err` are in **micro-Janskys**. For magnitude-based surveys the
conversion is `flux = 10^((ZP − mag) / 2.5)`, with `ZP = 23.9` for AB (3631 Jy). Surveys that
deliver flux natively are scaled directly: ATLAS and BlackGEM already report AB µJy; Gaia,
TESS, and K2 convert from e⁻/s with a fixed per-band zero point.

Two photometric systems are in play:

- **AB** — BlackGEM, Gaia, ZTF, NSC, ATLAS, SkyMapper, J-VAR, and ASAS-SN `g`.
- **Vega** — CRTS, OGLE, TESS, K2, and ASAS-SN `v`, each with a band-specific zero point
  (Kepler 3241.90 Jy, TESS 2631.88 Jy, Cousins-I 2416 Jy, Johnson-V 3636 Jy; exact values in
  `band_reference.csv`).

**Negative and low-significance fluxes are kept on purpose.** The forced-photometry surveys
(ATLAS, TESS, K2, ASAS-SN) retain faint and slightly-negative measurements with their error
bars rather than clipping at `flux > 0`, because discarding them would bias the noise floor
that SNR-based period searches depend on. Expect negative points in faint light curves since they
are valid measurements, not errors.

## Data cleaning

Most surveys carry a `clean` switch (default `True`):

- **`clean=True`** applies that survey's documented quality cuts: quality flags, blend and
  saturation limits, on-chip position, PSF-shape and aperture-fit sanity, image-depth
  thresholds, adapted from each survey's own pipeline papers.
- **`clean=False`** keeps more points but still removes the obvious garbage that would break
  downstream code: NaN times/fluxes, non-positive errors, and non-detection sentinels.

A few surveys (TESS, K2, ZTF) apply their standard quality mask unconditionally and expose no
`clean` toggle; their `clean` column in `query_metadata.csv` is therefore blank.

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

Paste the printed token into `config.yaml` under `credentials: → atlas_token:` (or export it
as the `ATLASFORCED_SECRET_KEY` environment variable).

**Gaia** — optional. You can add `gaia_user` / `gaia_password` to `config.yaml`, but
epoch-photometry queries work fine without logging in.

**BlackGEM** — requires Google Cloud access. Authenticate once with the Google Cloud CLI
(`gcloud auth application-default login`); no token goes in the config.

## Quick start

```python
from lcquery import get_all_lightcurves

# Pass a Gaia DR3 source_id together with its ICRS coordinates
source_id = 3398501003057791360
ra  = 87.2655554543
dec = 19.0727344751

get_all_lightcurves(source_id, ra, dec, verbose=True)
```

This writes CSVs to disk (it does not return a DataFrame). To restrict a single call to a
subset of surveys, pass `survey_list=["ZTF", "TESS"]`.

## Output

Light curves are written under `lightcurves/`, one folder per survey, each CSV named by the
source's Gaia DR3 ID:

```
lightcurves/
├── ZTF/3398501003057791360.csv
├── Gaia/3398501003057791360.csv
├── ATLAS/3398501003057791360.csv
├── …
├── query_metadata.csv          # one row per (source, survey): status, counts, settings used
└── band_reference.csv          # machine-readable data dictionary (see below)
```

**Light-curve CSVs.** Each has four columns — `BJD`, `Target_flux`, `Target_flux_err`,
`Filter` — preceded by a optional commented `#` header (see Configuration) recording the survey, `source_id`, coordinates, and column units. Read them back with `comment="#"`:

```python
import pandas as pd
df = pd.read_csv("lightcurves/ZTF/3398501003057791360.csv", comment="#")
```

**`query_metadata.csv`** logs one row per (source, survey) with: `source_id`, `ra`, `dec`,
`survey`, `observations`, `status` (`success`, `no_match`, `no_data`, `filtered_out`, …),
`clean`, `radius` / `radius_unit`, `time_unit` (always `BJD_TDB`), `flux_unit` (`uJy_AB`,
`uJy_Vega`, or `uJy_mixed` for ASAS-SN), and `queried_utc`. The `status` column tells you
*why* a survey returned nothing, and the file is updated in place — re-running a source
overwrites its rows rather than duplicating them.

**`band_reference.csv`** is a self-describing data dictionary written on every run. It has one
row per survey giving the photometric `system`, zero-point flux (`zp_Jy`), `phot` type,
`timing` convention, `cadence`, and `bands`, plus extra rows for the per-band overrides where
a survey's bands differ (`asassn-g`/`asassn-v`, `ogle-i`/`ogle-v`). Those are the rows with
the `filter` column filled in.

**Caching.** If a CSV already exists it is skipped on the next run. Pass `overwrite=True` (or
set it in the config) to re-download.

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit it. Globally you can set:
- `base_dir` — where CSVs + `query_metadata.csv` and `band_reference` go.
- `overwrite` — re-download even if a cached CSV exists.
- `write_header_comments` — prepend commented provenance to each CSV.
- `clean`  — see **Data cleaning** above. *(TESS, K2, and ZTF clean unconditionally and ignore this.)*

Per survey you can set:
- `enabled` — turn a survey on or off (or restrict a single call with `survey_list=[...]`).
- `radius_arcsec` — the cone-search radius. *(BlackGEM, Gaia, and ATLAS take no radius — see Notes.)*
- `clean` — see **Data cleaning** above. *(TESS, K2, and ZTF clean unconditionally and ignore this.)*

## Notes

- **Gaia and BlackGEM are queried by identity, not position.** The `source_id` you pass must
  be a **Gaia DR3 source_id** — it cross-matches the source directly, so ra/dec are used only
  for the time correction, never for the lookup. Pass an ID from another system and these two
  return no data.

- **ATLAS has no search radius** because it does *forced photometry*: the server gets an
  exact (ra, dec) and it measures the flux at that precise point on every image, whether or
  not anything is catalogued there. There is no catalogue to search, so nothing for a radius
  to bound so the coordinates *are* the query.

- **"No data" is expected due to sky coverage.** Surveys have footprints and selection
  functions: southern surveys (SkyMapper, OGLE) won't have northern targets, CRTS excludes the
  Galactic plane, OGLE only contains catalogued variables, Gaia DR3 epoch photometry is
  released for only a subset of sources, and K2 covers only its campaign fields. The `status`
  column in `query_metadata.csv` records why each survey came back empty.

- **The first OGLE query is slow.** OGLE has no positional API, so on first use `lcquery`
  crawls and caches the full OGLE Collection of Variable Stars (~1.2M entries) to
  `lcquery/surveys/ogle_ocvs_master.csv`. This one-time build takes a while; every query
  afterward reads the cache.

- **CHEOPS** is implemented but disabled by default (the DACE service is down and the code hasn't been tested) and is omitted from `band_reference.csv`.