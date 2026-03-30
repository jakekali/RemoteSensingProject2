# patch_metrics_reports.py
# Adds per-class 'report' to local_metrics.json for models needed by
# retrain_boosted.py (XGB_weighted, RF_500).
# Run: ..\venv\Scripts\python patch_metrics_reports.py

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs')

CLASS_MAP = {0:'Other', 1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             26:'Dbl_Crop', 28:'Oats', 36:'Alfalfa'}

print("Loading training data...")
df = pd.read_csv(os.path.join(DATA_DIR, 'raw_full_boosted.csv'))
meta = ['cdl_code','crop_name','dataset','region','label_year',
        'image_year','split','longitude','latitude','cdl_group']
feat_cols = [c for c in df.columns if c not in meta]

df_tr = df[df['split']=='train'].dropna(subset=feat_cols)
df_te = df[df['split']=='test'].dropna(subset=feat_cols)
X_tr, y_tr = df_tr[feat_cols].values, df_tr['cdl_code'].values.astype(int)
X_te, y_te = df_te[feat_cols].values, df_te['cdl_code'].values.astype(int)
sw = compute_sample_weight('balanced', y=y_tr)

classes     = np.sort(np.unique(y_tr))
code_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_code = {i: c for c, i in code_to_idx.items()}
y_tr_idx    = np.array([code_to_idx[c] for c in y_tr])

metrics_path = os.path.join(OUT_DIR, 'local_metrics.json')
with open(metrics_path) as f:
    all_metrics = json.load(f)

def add_report(name, preds):
    present = sorted(set(y_te))
    names   = [CLASS_MAP.get(c, str(c)) for c in present]
    rep = classification_report(y_te, preds, labels=present,
                                target_names=names, output_dict=True,
                                zero_division=0)
    if name not in all_metrics:
        all_metrics[name] = {}
    all_metrics[name]['report'] = rep
    print(f"  {name}: report added")

# XGB_weighted
print("Training XGB_weighted...")
m = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                  subsample=0.8, colsample_bytree=0.8,
                  eval_metric='mlogloss', n_jobs=-1, random_state=42, verbosity=0)
m.fit(X_tr, y_tr_idx, sample_weight=sw)
preds = np.array([idx_to_code[i] for i in m.predict(X_te)])
add_report('XGB_weighted', preds)

# RF_500
print("Training RF_500...")
rf = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                             n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
add_report('RF_500', rf.predict(X_te))

with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nPatched: {metrics_path}")
