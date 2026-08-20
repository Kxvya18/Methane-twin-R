# MethaneTwin-R

**Physics-Informed Spatiotemporal Machine Learning for Probabilistic Detection, Source Attribution, and Quantification of Methane Super-Emitters**

MethaneTwin-R is an end-to-end research codebase, not a notebook demo. It includes:

- real NASA EMIT V002 methane data acquisition,
- ERA5 meteorology acquisition and temporal/spatial alignment,
- immutable raw-data manifests and SHA-256 provenance,
- raster quality control and geospatial dataset construction,
- physics-based plume simulation for pretraining,
- classical inverse modelling and ML baselines,
- a wind-conditioned multi-task deep model,
- synthetic pretraining -> real EMIT fine-tuning,
- hyperparameter optimization,
- conformal emission-rate uncertainty,
- detection / segmentation / source localization / emission-rate evaluation,
- geographic holdout splits,
- domain-shift diagnostics,
- source/facility attribution,
- Bayesian persistence estimates across repeated observations,
- experiment matrices / ablations / bootstrap intervals,
- FastAPI inference,
- an interactive Streamlit investigation simulator,
- Docker,
- tests,
- reproducibility and model-registry artifacts.

## Scientific data basis

The default real-data path uses **NASA EMIT Level 2B Methane Plume Complexes V002 (`EMITL2BCH4PLM`)** and optional **Methane Enhancement V002 (`EMITL2BCH4ENH`)** background scenes. EMIT V002 plume products include plume COGs plus JSON/GeoJSON metadata with emissions and uncertainty; the enhancement product includes methane enhancement, sensitivity and per-pixel uncertainty products. EMIT products are nominally ~60 m at the equator.

Meteorology is aligned from **ERA5** through the Copernicus Climate Data Store API.

The repository deliberately does not bundle NASA or ERA5 data. Downloading real data requires your own Earthdata/CDS credentials and acceptance of their terms.

## What is and is not "ground truth"

- EMIT manually vetted plume products are treated as high-confidence observational labels for plume presence/shape.
- EMIT V002 emission estimates are observational reference labels, not perfect physical truth.
- Exact source-coordinate supervision is used only when the metadata actually exposes a source coordinate.
- If exact source coordinates are absent, the project can derive an **upwind physics proxy**. Proxy localization is clearly marked and is never reported as true-source localization accuracy.
- Physics simulations have exact source/emission ground truth and are used for pretraining and controlled inverse-problem evaluation.
- Real EMIT test regions are never used for hyperparameter selection or uncertainty calibration.

---

# 1. Setup on macOS / Linux

```bash
cd MethaneTwin-R
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[research]"
```

For Apple Silicon, PyTorch normally installs a compatible wheel through pip. The code automatically uses MPS when available, otherwise CUDA or CPU.

Run tests:

```bash
python -m pytest -q
```

---

# 2. Credentials

Copy:

```bash
cp .env.example .env
```

For NASA Earthdata, either:

1. create `~/.netrc` using your Earthdata credentials, or
2. set `EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`.

For ERA5/CDS, configure the CDS API credentials according to the CDS API instructions. Modern `cdsapi` can use the user configuration file / token expected by CDS.

Never commit credentials.

---

# 3. Download real EMIT V002 data

Edit the geographic/time window in:

```text
configs/research.yaml
```

Then:

```bash
methanetwin data download-emit \
  --config configs/research.yaml
```

This downloads both:

```text
EMITL2BCH4PLM V002
EMITL2BCH4ENH V002
```

according to your configured limits.

For a first local run, use:

```bash
methanetwin data download-emit \
  --config configs/quick.yaml
```

The quick configuration reduces *volume*, not scientific stages.

---

# 4. Build the EMIT event catalogue

```bash
methanetwin data catalog \
  --config configs/research.yaml
```

This:

- pairs plume `.tif` and metadata `.json/.geojson`,
- reads geospatial bounds,
- extracts plume centroid,
- attempts emission-rate + uncertainty extraction,
- extracts exact source coordinates only when present,
- records label quality,
- computes file hashes,
- creates `data/interim/emit_events.parquet`.

Inspect metadata extraction quality:

```bash
methanetwin data metadata-report \
  --config configs/research.yaml
```

Do not train if emission-unit parsing is ambiguous.

---

# 5. Download ERA5 event meteorology

```bash
methanetwin data download-era5 \
  --config configs/research.yaml
```

The downloader requests, caches, and aligns:

- 10 m U wind,
- 10 m V wind,
- 2 m temperature,
- surface pressure,
- boundary-layer height

for each EMIT event.

The result is:

```text
data/interim/era5_events.parquet
```

---

# 6. Build the real ML dataset

```bash
methanetwin data build \
  --config configs/research.yaml
```

This creates:

```text
data/processed/
├── samples/
│   ├── <sample_id>.npz
│   └── ...
├── manifest.parquet
├── splits.parquet
├── normalization.json
├── quality_report.json
└── provenance.json
```

Positive samples are built from real EMIT plume COGs.

High-confidence negative patches are sampled from EMIT enhancement scenes only when the patch remains below configured methane-enhancement thresholds.

The split is **geographic/group based**, not random-pixel splitting:

```text
train
validation
calibration
test
```

---

# 7. Generate physics pretraining data

```bash
methanetwin physics generate \
  --config configs/research.yaml
```

The simulator varies:

- source location,
- emission rate,
- wind speed/direction,
- dispersion,
- atmospheric/background noise,
- cloud/quality masking,
- retrieval noise.

Every synthetic scene has exact source coordinates, emission rate and plume mask.

---

# 8. Train classical baselines

```bash
methanetwin train baselines \
  --config configs/research.yaml
```

Baselines include:

- physics inversion,
- logistic/RF plume detection,
- Random Forest emission regression,
- Extra Trees emission regression,
- HistGradientBoosting emission regression.

Handcrafted plume features include:

- integrated enhancement,
- maximum enhancement,
- plume area,
- weighted centroid,
- principal-axis elongation,
- orientation,
- gradient statistics,
- wind alignment,
- downwind extent.

Baselines are mandatory; the deep model is not allowed to "win" by comparison only with a trivial constant predictor.

---

# 9. Hyperparameter optimization

```bash
methanetwin train tune \
  --config configs/research.yaml
```

Optuna tunes only on training/validation groups. It never reads calibration/test metrics for optimization.

Tuned parameters include:

- learning rate,
- weight decay,
- dropout,
- encoder width,
- attention heads,
- physics loss weight,
- source loss weight,
- emission NLL weight.

---

# 10. Physics pretraining

```bash
methanetwin train pretrain \
  --config configs/research.yaml
```

The model is a **wind-conditioned multi-task encoder-attention-decoder**.

Outputs:

1. plume detection probability,
2. pixel plume segmentation,
3. normalized source coordinate,
4. log emission rate,
5. predictive log variance.

Loss:

```text
detection BCE
+ segmentation BCE/Dice
+ masked source Smooth-L1
+ heteroscedastic emission NLL
+ physics reconstruction consistency
```

---

# 11. Real-data fine-tuning

```bash
methanetwin train finetune \
  --config configs/research.yaml
```

Protocol:

1. load synthetic-pretrained weights,
2. freeze the early encoder for warm adaptation,
3. train task heads on real EMIT training regions,
4. unfreeze,
5. continue with lower learning rate,
6. choose epoch/checkpoint using only geographic validation regions,
7. preserve calibration and final test regions.

The output explicitly compares:

```text
synthetic-pretrained -> real test
vs
real-fine-tuned -> real test
```

so fine-tuning is claimed as useful only if the held-out real-data metrics improve.

---

# 12. Calibrate uncertainty

```bash
methanetwin train calibrate \
  --config configs/research.yaml
```

The calibration split is untouched by training/hyperparameter search.

It produces:

- detection temperature scaling,
- conformal emission residual quantile,
- target/empirical coverage,
- model-calibration artifact.

---

# 13. Final evaluation

```bash
methanetwin evaluate \
  --config configs/research.yaml
```

Metrics include:

### Detection
- precision
- recall
- F1
- AUROC
- average precision

### Segmentation
- IoU
- Dice

### Source localization
- pixel error on synthetic exact-ground-truth scenes
- geodesic error only for real samples with actual source labels
- proxy results are separated

### Emission
- MAE
- RMSE
- MAPE
- log-MAE

### Uncertainty
- conformal interval coverage
- mean interval width
- detection calibration error

### Physics
- reconstruction consistency

### Generalization
- unseen-region test performance

Results are written under:

```text
artifacts/evaluation/
```

---

# 14. Ablations

```bash
methanetwin evaluate ablate \
  --config configs/research.yaml
```

The experiment suite includes:

- no wind,
- no physics loss,
- no attention,
- standard attention vs wind-conditioned attention,
- no synthetic pretraining,
- frozen pretrained model,
- full fine-tuning,
- no uncertainty head,
- classical baseline vs deep model.

---

# 15. Experiment matrix

```bash
python scripts/run_experiment_matrix.py \
  --config configs/research.yaml
```

The matrix varies random seeds and physics/noise regimes and produces aggregated mean, SD and bootstrap CIs.

---

# 16. Source/facility attribution

Provide a facility inventory CSV:

```text
facility_id,name,latitude,longitude,facility_type
```

Path is configured in YAML.

Then:

```bash
methanetwin attribute \
  --config configs/research.yaml
```

The system ranks candidate facilities using:

- distance from inferred source,
- wind consistency,
- facility-type prior,
- repeated-event evidence.

It does **not** claim legal/operational responsibility; it outputs a ranked attribution hypothesis.

---

# 17. Bayesian persistence analysis

```bash
methanetwin persistence \
  --config configs/research.yaml
```

Where enhancement-scene coverage supports it, the system creates source observation opportunities and estimates a Beta-Binomial posterior:

```text
P(source active on an observed opportunity | data)
```

This separates recurrent sources from one-off detections.

---

# 18. Interactive investigation simulator

```bash
python -m streamlit run simulator/app.py
```

The simulator has two modes.

## Physics Lab

Interactively change:

- Q (kg CH4/h)
- wind speed
- wind direction
- source position
- dispersion
- retrieval noise
- cloud masking

Then generate a fresh plume and run the trained model.

It displays:

```text
true vs estimated source
true vs estimated Q
localization error
prediction interval
segmentation
physics reconstruction
```

## Real EMIT Investigator

Choose a real held-out EMIT scene and inspect:

- enhancement image
- predicted plume mask
- detected probability
- emission estimate + interval
- source estimate
- label-quality status
- candidate facility ranking
- repeated-observation evidence

This is a simulator/investigation application, not a static dashboard reading one CSV.

---

# 19. API

```bash
uvicorn api.main:app --reload --port 8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

---

# 20. Docker

```bash
docker compose up --build
```

Services:

```text
API       http://localhost:8000
Simulator http://localhost:8501
```

Data and model artifacts are mounted as volumes rather than baked into the image.

---

# 21. Complete one-command orchestration

After credentials are configured:

```bash
python scripts/run_full_pipeline.py \
  --config configs/research.yaml
```

Stages are checkpointed. If a long data-download stage is already complete, it is not repeated unless `--force` is supplied.

---

# Research validity rules

Do not put a result on a CV until `artifacts/evaluation/final_test_metrics.json` exists from a real held-out test run.

Do not call:
- EMIT emission estimates "perfect ground truth",
- upwind proxy coordinates "true source coordinates",
- feature importance "causality",
- random train/test splits "generalization".

Read:

```text
docs/RESEARCH_PROTOCOL.md
docs/DATA_CARD.md
docs/MATHEMATICS.md
docs/VALIDITY_AND_LIMITATIONS.md
```
