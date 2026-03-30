# Local Model Testing

Parallel path to the GEE pipeline — train & evaluate sklearn models locally
on data exported from Earth Engine.

## Setup: New GEE Credentials

1. Go to https://console.cloud.google.com
2. **IAM & Admin → Service Accounts** → create or select an account
3. **Keys → Add Key → JSON** → download the file
4. Drop it here as **`gee_creds.json`** (next to `auth.py`)
   - OR set env var: `set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\key.json`
5. Test: ``..\venv\Scripts\python auth.py``

## Run Order

```
Step 1 — Export data from GEE (needs internet + GEE auth):
    ..\venv\Scripts\python export_training_data.py

Step 2 — Train local models (offline, no GEE):
    ..\venv\Scripts\python train_local_models.py

Step 3 — Compare local vs GEE results:
    ..\venv\Scripts\python compare_to_gee.py
```

## Folder Layout

```
local_model_testing/
  auth.py                   GEE init (service account or OAuth fallback)
  export_training_data.py   GEE -> local CSV (run once)
  train_local_models.py     sklearn RF / GBT / SVM / LR on local data
  compare_to_gee.py         side-by-side plots vs Phase 3 GEE RF
  gee_creds.json            YOUR service account key (not committed)
  data/
    training_all_features.csv
    v1_train.csv / v1_test.csv
    v2_train.csv / v2_test.csv  ...
  models/                   (saved model pickles — optional)
  outputs/
    local_metrics.json
    local_cm_{v}_{model}.png
    local_importance_{v}_{model}.png
    compare_accuracy.png
    compare_summary.txt
```

## Models Tested

| Name    | Description                          |
|---------|--------------------------------------|
| RF_200  | Random Forest 200 trees (mirrors GEE)|
| RF_500  | Random Forest 500 trees              |
| GBT     | Gradient Boosting 200 trees          |
| SVM     | SVM RBF kernel (scaled)              |
| LR      | Logistic Regression baseline         |
