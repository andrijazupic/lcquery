# lcquery

Multi-survey time-domain light-curve queries on a unified schema. Given a source's
Gaia DR3 `source_id` and ICRS coordinates, lcquery fans out to a dozen photometric
archives and standardizes every light curve to **BJD_TDB** time and **AB µJy** flux
(Vega zero point for OGLE and CRTS), writing one CSV per source per survey.

**Surveys:** BlackGEM, TESS, K2, ZTF, CRTS, NSC, ATLAS, SkyMapper, J-VAR, ASAS-SN,
OGLE, Gaia (plus CHEOPS, off by default).

## Install
```bash
git clone https://github.com/<username>/lcquery.git
cd lcquery
pip install -e .          # core: 10 surveys
pip install -e ".[all]"   # + BlackGEM (BigQuery) and CHEOPS (DACE)
```

## Usage
```python
from lcquery import get_all_lightcurves
get_all_lightcurves(2533263627776673024, 16.5959241329, -0.2490213677)
```

## Configuration
Copy `config.example.yaml` to `config.yaml` and edit per-survey settings
(enable/radius/clean) and your ATLAS token. `config.yaml` is gitignored;
if absent, settings fall back to `config.example.yaml`.