# local_model_testing/download_data.py
# Downloads all available training assets from GEE to local CSVs.
#
# DATA PROVENANCE
# ---------------
# Source project : heroic-goal-412401
# Training region: Iowa + Illinois (TIGER/2018/States)
# Labels         : USDA NASS CDL 2023 (30m crop type map)
# Imagery        : Landsat 8+9 C02 T1_L2, 2023-01-01 -> 2023-12-31
#                  Sentinel-1 GRD (SAR), 2023 (in v3_sar asset)
# Classes        : 1=Corn, 5=Soybeans, 24=Winter_Wheat, 28=Oats, 36=Alfalfa
# Split          : NO pre-split in assets — we generate reproducible 80/20 locally
#
# Assets downloaded:
#   training_samples_v4       2081 rows  42 features  (v1-v4 Landsat)
#   training_samples_v3_landsat 4000 rows  32 features  (larger sample, Landsat harmonics)
#   training_samples_v3_sar   4000 rows  44 features  (same + Sentinel-1 SAR)
#
# Outputs -> data/
#   raw_v4.csv                    all v4 Landsat features + lat/lon + label
#   raw_landsat.csv               larger Landsat harmonic dataset
#   raw_sar.csv                   Landsat + SAR combined
#   provenance.json               metadata for every dataset
#
# NOTE: Uses heroic-goal-412401 OAuth credentials (assets live there).
#       New service account (theta-grid-99720) used for any new GEE compute.
#
# Run: ..\venv\Scripts\python download_data.py

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ee
import pandas as pd
import numpy as np
import json
from datetime import datetime

# ── Auth: use original project OAuth to read assets ───────────────────────────
# Assets are owned by heroic-goal-412401; service account can't read them yet.
print("Initialising GEE with heroic-goal-412401 (asset owner)...")
ee.Initialize(project='heroic-goal-412401')
print("  OK")

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

ASSET_ROOT = 'projects/heroic-goal-412401/assets/project2'

LABEL_MAP = {1: 'Corn', 5: 'Soybeans', 24: 'Winter_Wheat', 28: 'Oats', 36: 'Alfalfa'}

# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════

def fc_to_dataframe(asset_name, label='downloading'):
    """Download a GEE FeatureCollection to a pandas DataFrame with lat/lon."""
    print(f"\n  {label}...")
    fc = ee.FeatureCollection(f'{ASSET_ROOT}/{asset_name}')
    n = fc.size().getInfo()
    print(f"    {n} rows in asset")

    # Batch download in pages to avoid memory issues
    all_records = []
    page_size   = 500
    offset      = 0

    while offset < n:
        batch = fc.toList(page_size, offset).getInfo()
        for feat in batch:
            props = dict(feat['properties'])
            # Extract lat/lon from geometry
            geo = feat.get('geometry')
            if geo and geo.get('coordinates'):
                coords = geo['coordinates']
                props['longitude'] = round(coords[0], 6)
                props['latitude']  = round(coords[1], 6)
            all_records.append(props)
        offset += page_size
        print(f"    downloaded {min(offset, n)}/{n} rows...", end='\r')

    print(f"    downloaded {n}/{n} rows     ")
    df = pd.DataFrame(all_records)
    return df


def add_metadata(df, dataset_name):
    """Add provenance columns."""
    df = df.copy()
    df['dataset']    = dataset_name
    df['region']     = 'Iowa+Illinois'
    df['label_year'] = 2023
    df['image_year'] = 2023
    df['crop_name']  = df['cdl_code'].map(LABEL_MAP).fillna('Unknown')

    # Reproducible 80/20 train/test split using numpy seed
    rng = np.random.default_rng(seed=42)
    df['split'] = np.where(
        rng.random(len(df)) < 0.8, 'train', 'test'
    )
    return df


# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD 1: training_samples_v4 (42 Landsat features)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("DATASET 1: training_samples_v4")
print("  42 Landsat features (monthly snapshots + shape + harmonics + robust)")
print("  2081 samples | Iowa+Illinois | CDL 2023 | Landsat 2023")
print("="*55)

df_v4 = fc_to_dataframe('training_samples_v4', 'Downloading v4 Landsat features')
df_v4 = add_metadata(df_v4, 'v4_landsat')

path_v4 = os.path.join(DATA_DIR, 'raw_v4.csv')
df_v4.to_csv(path_v4, index=False)
print(f"  Saved: {path_v4}  ({len(df_v4)} rows x {len(df_v4.columns)} cols)")
print(f"  Class distribution:\n{df_v4['crop_name'].value_counts().to_string()}")
print(f"  Train/test: {df_v4['split'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD 2: training_samples_v3_landsat (4000 rows, Landsat harmonics)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("DATASET 2: training_samples_v3_landsat")
print("  32 Landsat harmonic features (NIR/GCVI/SWIR1/SWIR2)")
print("  4000 samples | Iowa+Illinois | CDL 2023 | Landsat 2023")
print("="*55)

df_ls = fc_to_dataframe('training_samples_v3_landsat', 'Downloading Landsat harmonics')
df_ls = add_metadata(df_ls, 'v3_landsat')

path_ls = os.path.join(DATA_DIR, 'raw_landsat.csv')
df_ls.to_csv(path_ls, index=False)
print(f"  Saved: {path_ls}  ({len(df_ls)} rows x {len(df_ls.columns)} cols)")
print(f"  Class distribution:\n{df_ls['crop_name'].value_counts().to_string()}")
print(f"  Train/test: {df_ls['split'].value_counts().to_dict()}")

# ══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD 3: training_samples_v3_sar (4000 rows, Landsat + Sentinel-1 SAR)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("DATASET 3: training_samples_v3_sar")
print("  44 features: 32 Landsat harmonics + 12 Sentinel-1 SAR bands")
print("  4000 samples | Iowa+Illinois | CDL 2023 | SAR 2023")
print("  SAR bands: VV/VH/VHVV x (jul, aug, mean, std)")
print("="*55)

df_sar = fc_to_dataframe('training_samples_v3_sar', 'Downloading SAR + Landsat features')
df_sar = add_metadata(df_sar, 'v3_sar')

path_sar = os.path.join(DATA_DIR, 'raw_sar.csv')
df_sar.to_csv(path_sar, index=False)
print(f"  Saved: {path_sar}  ({len(df_sar)} rows x {len(df_sar.columns)} cols)")
print(f"  Class distribution:\n{df_sar['crop_name'].value_counts().to_string()}")
print(f"  Train/test: {df_sar['split'].value_counts().to_dict()}")

# ── Identify SAR-only columns ──────────────────────────────────────────────────
sar_only_cols = [c for c in df_sar.columns if c.startswith('SAR_')]
print(f"\n  SAR feature columns ({len(sar_only_cols)}): {sar_only_cols}")

# ══════════════════════════════════════════════════════════════════════════════
# PROVENANCE JSON
# ══════════════════════════════════════════════════════════════════════════════
provenance = {
    'downloaded_at': datetime.now().isoformat(),
    'source_project': 'heroic-goal-412401',
    'training_region': {
        'states': ['Iowa', 'Illinois'],
        'boundary': 'TIGER/2018/States',
    },
    'labels': {
        'source': 'USDA NASS CDL',
        'year': 2023,
        'classes': LABEL_MAP,
    },
    'imagery': {
        'landsat': 'LANDSAT/LC08+LC09/C02/T1_L2  2023-01-01 to 2023-12-31',
        'sar': 'COPERNICUS/S1_GRD  2023 (Sentinel-1)',
    },
    'split': 'reproducible 80/20  seed=42  no pre-split in GEE assets',
    'datasets': {
        'raw_v4.csv': {
            'source_asset': 'training_samples_v4',
            'rows': len(df_v4),
            'feature_cols': [c for c in df_v4.columns
                             if c not in ('cdl_code','crop_name','dataset','region',
                                          'label_year','image_year','split',
                                          'longitude','latitude')],
            'notes': 'All v1-v4 features: monthly snapshots + shape + harmonics + robust stats',
        },
        'raw_landsat.csv': {
            'source_asset': 'training_samples_v3_landsat',
            'rows': len(df_ls),
            'feature_cols': [c for c in df_ls.columns
                             if c not in ('cdl_code','crop_name','dataset','region',
                                          'label_year','image_year','split',
                                          'longitude','latitude')],
            'notes': 'Larger sample, Landsat harmonic features for raw bands (NIR/GCVI/SWIR)',
        },
        'raw_sar.csv': {
            'source_asset': 'training_samples_v3_sar',
            'rows': len(df_sar),
            'feature_cols': [c for c in df_sar.columns
                             if c not in ('cdl_code','crop_name','dataset','region',
                                          'label_year','image_year','split',
                                          'longitude','latitude')],
            'sar_cols': sar_only_cols,
            'notes': 'Same points as raw_landsat.csv but includes Sentinel-1 SAR features',
        },
    },
}

prov_path = os.path.join(DATA_DIR, 'provenance.json')
with open(prov_path, 'w') as f:
    json.dump(provenance, f, indent=2)
print(f"\nSaved: {prov_path}")

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("DOWNLOAD COMPLETE")
print("="*55)
total_mb = sum(
    os.path.getsize(os.path.join(DATA_DIR, f)) / 1e6
    for f in ['raw_v4.csv','raw_landsat.csv','raw_sar.csv']
)
print(f"  raw_v4.csv       {len(df_v4):>5} rows  {len([c for c in df_v4.columns if c not in ('cdl_code','crop_name','dataset','region','label_year','image_year','split','longitude','latitude')]):>3} features")
print(f"  raw_landsat.csv  {len(df_ls):>5} rows  {len([c for c in df_ls.columns if c not in ('cdl_code','crop_name','dataset','region','label_year','image_year','split','longitude','latitude')]):>3} features")
print(f"  raw_sar.csv      {len(df_sar):>5} rows  {len([c for c in df_sar.columns if c not in ('cdl_code','crop_name','dataset','region','label_year','image_year','split','longitude','latitude')]):>3} features  (incl. {len(sar_only_cols)} SAR)")
print(f"  Total size: {total_mb:.1f} MB")
print(f"\n  All samples: Iowa + Illinois | CDL 2023 | Landsat/SAR 2023")
print(f"  Next: run train_local_models.py")
