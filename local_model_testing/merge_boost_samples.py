# local_model_testing/merge_boost_samples.py
# Merges the boosted Alfalfa+Oats samples into raw_full.csv.
# Run after downloading boost_alfalfa_oats_2022.csv and boost_alfalfa_oats_2023.csv
#
# Run: ..\venv\Scripts\python merge_boost_samples.py

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime
from feature_registry import FEATURE_SETS

DATA_DIR  = os.path.join(os.path.dirname(__file__), 'data')
feat_cols = FEATURE_SETS['v4']

CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}

meta = ['cdl_code','crop_name','dataset','region','label_year',
        'image_year','split','longitude','latitude','cdl_group']

def process_boost_csv(path, year):
    """Load, clean, parse geo, drop missing features."""
    df = pd.read_csv(path)
    print(f"  {os.path.basename(path)}: {len(df)} rows raw")

    # Parse .geo
    if '.geo' in df.columns:
        def parse(g):
            try:
                import json as _j
                c = _j.loads(str(g))['coordinates']
                return round(float(c[0]),6), round(float(c[1]),6)
            except: return None, None
        df[['longitude','latitude']] = df['.geo'].apply(
            lambda g: pd.Series(parse(g)))
        df = df.drop(columns=['.geo'])

    # Drop missing
    fc = [c for c in feat_cols if c in df.columns]
    before = len(df)
    df = df.dropna(subset=fc)
    df = df[(df[fc] != -9999).all(axis=1)]
    print(f"  After cleaning: {len(df)} rows ({before-len(df)} dropped)")

    # Add metadata
    df['crop_name']  = df['cdl_code'].map(CLASS_MAP)
    df['dataset']    = f'boost_{year}'
    df['region']     = 'Iowa+Illinois'
    df['label_year'] = year
    df['image_year'] = year

    return df

# ── Load boost files ──────────────────────────────────────────────────────────
print("Loading boost CSVs...")
boost_dfs = []
for year in [2023, 2022]:
    path = os.path.join(DATA_DIR, f'boost_alfalfa_oats_{year}.csv')
    if not os.path.exists(path):
        print(f"  MISSING: {path} — download from Drive first")
        continue
    df = process_boost_csv(path, year)
    boost_dfs.append(df)
    print(f"  Class counts: {df['crop_name'].value_counts().to_dict()}")

if not boost_dfs:
    raise SystemExit("No boost files found. Download from Drive first.")

# ── Load existing raw_full.csv ────────────────────────────────────────────────
full_path = os.path.join(DATA_DIR, 'raw_full.csv')
df_full   = pd.read_csv(full_path)
print(f"\nExisting raw_full.csv: {len(df_full)} rows")
print("Before:")
print(df_full['crop_name'].value_counts().to_string())

# ── Merge ─────────────────────────────────────────────────────────────────────
# Align columns
all_cols = list(df_full.columns)
boost_combined = pd.concat(boost_dfs, ignore_index=True)

# Keep only cols present in both
shared = [c for c in all_cols if c in boost_combined.columns]
missing_in_boost = [c for c in all_cols if c not in boost_combined.columns]
if missing_in_boost:
    print(f"\nColumns in full but not boost (will be NaN): {missing_in_boost}")
    for c in missing_in_boost:
        boost_combined[c] = None

boost_combined = boost_combined[all_cols]

# Assign reproducible stratified split
rng = np.random.default_rng(seed=99)
boost_combined['split'] = 'train'
for code in boost_combined['cdl_code'].unique():
    idx  = boost_combined.index[boost_combined['cdl_code'] == code]
    test = rng.choice(idx, size=int(len(idx) * 0.2), replace=False)
    boost_combined.loc[test, 'split'] = 'test'

df_merged = pd.concat([df_full, boost_combined], ignore_index=True)

print(f"\nAfter merging {len(boost_combined)} boost samples:")
print(df_merged['crop_name'].value_counts().to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
# Keep raw_full.csv as backup, write raw_full_boosted.csv
out_path = os.path.join(DATA_DIR, 'raw_full_boosted.csv')
df_merged.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}  ({len(df_merged)} rows)")

# Show year breakdown for the two boosted classes
print("\nYear breakdown for Alfalfa + Oats:")
for name in ['Alfalfa', 'Oats']:
    sub = df_merged[df_merged['crop_name'] == name]
    by_year = sub['image_year'].value_counts().sort_index()
    print(f"  {name}: {by_year.to_dict()}")

print("\nDone. To retrain with boosted data, update train_local_models.py to use raw_full_boosted.csv")
