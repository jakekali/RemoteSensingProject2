# process_drive_csv.py
# Processes the CSV downloaded from Google Drive into clean local training data.
# Run after downloading training_full_v4_7class.csv from Drive.

import os, sys, json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from feature_registry import FEATURE_SETS

DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
DRIVE_CSV = os.path.join(DATA_DIR, 'training_full_v4_7class.csv')

CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}

feature_names = FEATURE_SETS['v4']

print(f"Reading {DRIVE_CSV}...")
df = pd.read_csv(DRIVE_CSV)
print(f"  {len(df)} rows x {len(df.columns)} cols")

# ── Parse .geo -> lat/lon ─────────────────────────────────────────────────────
if '.geo' in df.columns:
    def parse_coords(geo_str):
        try:
            geo = json.loads(str(geo_str))
            c = geo.get('coordinates', [None, None])
            return round(float(c[0]), 6), round(float(c[1]), 6)
        except Exception:
            return None, None
    df[['longitude','latitude']] = df['.geo'].apply(
        lambda g: pd.Series(parse_coords(g))
    )
    df = df.drop(columns=['.geo'])
    print(f"  Parsed lat/lon from .geo")

# ── Drop rows with missing feature values ─────────────────────────────────────
feat_cols = [c for c in feature_names if c in df.columns]
missing_feats = [c for c in feature_names if c not in df.columns]
if missing_feats:
    print(f"  WARNING: {len(missing_feats)} features not in CSV: {missing_feats}")

before = len(df)
df = df.dropna(subset=feat_cols)
df = df[(df[feat_cols] != -9999).all(axis=1)]
print(f"  Dropped {before - len(df)} rows with missing data ({len(df)} remain)")

# ── Add metadata ──────────────────────────────────────────────────────────────
df['crop_name']  = df['cdl_code'].map(CLASS_MAP).fillna('Unknown')
df['dataset']    = 'full_v4_7class'
df['region']     = 'Iowa+Illinois'
df['label_year'] = 2023
df['image_year'] = 2023

# Stratified 80/20 split per class
rng = np.random.default_rng(seed=42)
df['split'] = 'train'
for code in df['cdl_code'].unique():
    idx  = df.index[df['cdl_code'] == code]
    test = rng.choice(idx, size=int(len(idx) * 0.2), replace=False)
    df.loc[test, 'split'] = 'test'

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, 'raw_full.csv')
df.to_csv(out_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nSaved: {out_path}")
print("\n" + "="*55)
print("CLASS DISTRIBUTION")
print("="*55)
summary = df.groupby('crop_name').agg(
    total=('cdl_code','count'),
    train=('split', lambda x: (x=='train').sum()),
    test=('split',  lambda x: (x=='test').sum()),
).sort_values('total', ascending=False)
print(summary.to_string())
print(f"\nTotal: {len(df)} rows | {len(feat_cols)} features | {df['cdl_code'].nunique()} classes")

under = summary[summary['total'] < 700]
if len(under):
    print(f"\nWARNING — classes under 700: {list(under.index)}")
else:
    print("All classes >= 700 samples")

# ── Update provenance ─────────────────────────────────────────────────────────
prov_path = os.path.join(DATA_DIR, 'provenance.json')
with open(prov_path) as f:
    prov = json.load(f)
prov['datasets']['raw_full.csv'] = {
    'source': 'Google Drive export: training_full_v4_7class.csv',
    'downloaded_at': datetime.now().isoformat(),
    'rows': len(df), 'features': len(feat_cols),
    'classes': {str(k): v for k, v in CLASS_MAP.items()},
    'class_counts': df['crop_name'].value_counts().to_dict(),
    'notes': '7-class (5 crops + Dbl_Crop + Other). v4 features. Iowa+Illinois 2023. 80/20 stratified.',
}
with open(prov_path, 'w') as f:
    json.dump(prov, f, indent=2)

print(f"\nProvenance updated. Ready for training.")
