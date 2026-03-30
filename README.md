# Crop Classification from Satellite Imagery
**ECE414 — Fundamentals of Remote Sensing and Earth Observation**
**Assignment 2 | Iowa & Illinois | 2023 Training / 2024 Deployment**

---

## Submission Files

| File | Description |
|------|-------------|
| [submission.ipynb](submission.ipynb) | Main submission notebook — all four phases with figures and analysis |
| [Final_Report.md](Final_Report.md) | Written report covering methodology, results, and conclusions |

---

## Repository Layout

```
remote_project_2/
├── submission.ipynb          Main submission notebook
├── Final_Report.md           Written report
├── README.md                 This file
│
├── core.py                   Shared GEE helpers: Landsat L8+L9 collection,
│                             cloud masking, spectral indices (NDVI/EVI/LSWI/GCVI)
├── feature_registry.py       Feature registry and builder: FEATURE_SETS (v1–v4),
│                             harmonic/shape/robust image builders
│
├── phase1.py                 Phase 1: CDL crop area analysis → outputs/phase1_*.png
├── phase2.py                 Phase 2: Phenology sampling + separability plots
│                             → outputs/phase2_*.png, outputs/phase2_stats.csv
├── phase3_create.py          Phase 3: GEE Random Forest training (v1–v4),
│                             exports training assets, saves outputs/phase3_metrics.json
├── phase3_analysis.py        Phase 3: Confusion matrices + feature importance plots
│                             → outputs/phase3_cm_*.png, outputs/phase3_importance_*.png
├── phase4_create.py          Phase 4: GEE generalization (county-level prediction maps)
├── phase4_analysis.py        Phase 4: Accuracy comparison + map panel figures
│
├── outputs/                  All generated figures and metrics (PNGs, JSONs, CSVs)
│
└── local_model_testing/      Full local ML pipeline (sklearn + XGBoost)
    ├── README.md             Local testing sub-readme
    ├── auth.py               GEE auth (service account or OAuth fallback)
    │
    │   — Data export (GEE → Drive → local) —
    ├── export_full_dataset.py        Export 2023 v4 training features (7 classes)
    ├── export_boost_rare_classes.py  Export 2022 Alfalfa+Oats boost samples
    ├── export_2022_full.py           Export full 2022 7-class dataset
    ├── export_2024_data.py           Export 2024 McLean IL + Renville MN test features
    ├── export_sar_to_drive.py        Export Sentinel-1 SAR features
    ├── download_from_drive.py        Download all exports from Google Drive → data/
    │
    │   — Dataset prep —
    ├── merge_boost_samples.py        Merge 2022 boost + 2023 base → raw_full_boosted.csv
    ├── create_final_dataset.py       Final merged dataset with splits
    ├── patch_metrics_reports.py      Adds per-class report to local_metrics.json
    │
    │   — Model training —
    ├── train_local_models.py         8-model comparison (LR/SVM/RF/GBT/XGB/MLP)
    │                                 → outputs/local_metrics.json, local_cm_*.png
    ├── tune_xgb.py                   Optuna 60-trial HPO on XGBoost (macro F1, 3-fold CV)
    │                                 → outputs/xgb_tuned_cm.png, xgb_recall_comparison.png
    ├── retrain_boosted.py            Retrain XGB+RF on 2022-boosted dataset
    │                                 → outputs/boosted_*.png, boosted_metrics.json
    │
    │   — Evaluation & visualization —
    ├── evaluate_2024.py              Apply all models to 2024 McLean + Renville test data
    │                                 → outputs/gen_metrics.json, gen_cm_*.png, gen_summary.png
    ├── plot_spatial_maps.py          Scatter maps: CDL reference vs predicted, error maps
    │                                 → ../outputs/phase4_*_spatial_map.png, *_error_map.png
    ├── cloud_and_cm_analysis.py      SAR fusion experiment (Landsat + Sentinel-1)
    │                                 → outputs/cloud_comparison.png, final_cm_*.png
    │
    ├── data/                         Local CSVs (downloaded from Drive, not committed)
    ├── outputs/                      Local model outputs (metrics, plots)
    └── auth_keys/                    Service account JSON key (not committed)
```

---

## Setup

```bash
# Create venv and install dependencies
py -3.10 -m venv venv
venv\Scripts\pip install earthengine-api pandas numpy matplotlib seaborn \
    scikit-learn xgboost optuna openpyxl requests googleapiclient

# Authenticate GEE (one-time)
venv\Scripts\python -c "import ee; ee.Authenticate()"
```

GEE project: `theta-grid-99720`
Service account key: `local_model_testing/auth_keys/theta-grid-99720-ea12c2bea3c1.json`

---

## Run Order

### GEE Pipeline (Phases 1–4)
```bash
venv\Scripts\python phase1.py
venv\Scripts\python phase2.py
venv\Scripts\python phase3_create.py    # submits GEE export tasks
venv\Scripts\python phase3_analysis.py
venv\Scripts\python phase4_create.py
venv\Scripts\python phase4_analysis.py
```

### Local Model Pipeline (Phase 3.2, 3.3, Phase 4)
```bash
cd local_model_testing

# 1. Export from GEE
..\venv\Scripts\python export_full_dataset.py
..\venv\Scripts\python export_boost_rare_classes.py
..\venv\Scripts\python export_2024_data.py
..\venv\Scripts\python download_from_drive.py

# 2. Prepare dataset
..\venv\Scripts\python merge_boost_samples.py

# 3. Train and tune
..\venv\Scripts\python train_local_models.py
..\venv\Scripts\python tune_xgb.py
..\venv\Scripts\python retrain_boosted.py
..\venv\Scripts\python patch_metrics_reports.py   # adds per-class report to metrics JSON

# 4. Evaluate on 2024 counties
..\venv\Scripts\python evaluate_2024.py
..\venv\Scripts\python plot_spatial_maps.py
..\venv\Scripts\python cloud_and_cm_analysis.py
```

---

## Key Results

| Component | Result |
|-----------|--------|
| GEE RF v4 (2023 test, 5 classes) | **87.2%**, κ = 0.838 |
| XGBoost base (7 classes, local) | **90.3%**, κ = 0.838 |
| XGBoost + class weights | **90.7%**, κ = 0.846 |
| XGBoost + Optuna tuning | **90.5%**, κ = 0.844 |
| Alfalfa recall after 2022 augmentation | 63% → **78%** |
| Oats recall after 2022 augmentation | 69% → **86%** |
| McLean IL 2024 (in-distribution) | **46–49%** (July cloud-masked) |
| Renville MN 2024 (out-of-distribution) | **72.4%**, κ = 0.668 |
| McLean IL + SAR fusion | **59%** (+13pp over Landsat-only) |
