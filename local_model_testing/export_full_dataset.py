# local_model_testing/export_full_dataset.py
# Full dataset export: 7 classes, 700+ samples each, v4 Landsat features.
#
# CLASS STRUCTURE (revised):
#   1  = Corn
#   5  = Soybeans
#   24 = Winter_Wheat
#   28 = Oats
#   36 = Alfalfa
#   26 = Dbl_Crop_WinWht_Soy   <-- own class (distinct mid-season NDVI dip)
#    0 = Other/Non-crop         <-- pasture + forest + developed + water + fallow
#
# SAMPLING TARGETS:
#   Crop classes (1,5,24,28,36): request 1000/class -> expect ~800+ survive
#   Dbl_Crop (26):               request 2000       -> expect ~960+ survive (48%)
#   Other (0):                   request 2000/group x5 groups -> ~700+ survive combined
#
# OUTPUT:
#   data/raw_full.csv     all 7 classes, v4 features, lat/lon, split, provenance cols
#   data/provenance.json  updated
#
# Run: ..\venv\Scripts\python export_full_dataset.py

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import ee
import numpy as np
import pandas as pd
from datetime import datetime

print("Initialising GEE...")
from auth import init_ee
init_ee(verbose=True)

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
ASSET_ROOT = 'projects/theta-grid-99720/assets/local_training'
os.makedirs(DATA_DIR, exist_ok=True)

# ── Study region ──────────────────────────────────────────────────────────────
study_region = (
    ee.FeatureCollection('TIGER/2018/States')
    .filter(ee.Filter.inList('NAME', ['Iowa', 'Illinois']))
    .geometry()
)

cdl_2023 = (
    ee.ImageCollection('USDA/NASS/CDL')
    .filter(ee.Filter.calendarRange(2023, 2023, 'year'))
    .first().select('cropland').clip(study_region)
)

CLASS_MAP = {
    1:  'Corn',
    5:  'Soybeans',
    24: 'Winter_Wheat',
    28: 'Oats',
    36: 'Alfalfa',
    26: 'Dbl_Crop',
    0:  'Other',
}

# ── Sampling config ───────────────────────────────────────────────────────────
# key: (cdl_codes_to_sample, n_raw_to_request, output_cdl_code)
SAMPLE_PLAN = {
    # 5 target crops — agricultural fields, high Landsat coverage (~80% survival)
    'Corn':         ([1],                   1000,  1),
    'Soybeans':     ([5],                   1000,  5),
    'Winter_Wheat': ([24],                  1000, 24),
    'Oats':         ([28],                  1000, 28),
    'Alfalfa':      ([36],                  1000, 36),
    # Double crop — own class, ~48% survival
    'Dbl_Crop':     ([26],                  2000, 26),
    # Other sub-groups — combined into code 0; ~36-55% survival per group
    'other_pasture':    ([176, 37],         2000,  0),
    'other_forest':     ([141, 142, 143],   2000,  0),
    'other_developed':  ([121, 122, 123, 124], 2000, 0),
    'other_water_wet':  ([111, 190, 195],   2000,  0),
    'other_fallow':     ([61, 63, 64],      2000,  0),
}

SEED = 42

# ══════════════════════════════════════════════════════════════════════════════
# BUILD 2023 FEATURE IMAGE (same as Phase 3)
# ══════════════════════════════════════════════════════════════════════════════
from core import build_landsat_collection, monthly_composite
from feature_registry import (build_feature_image, build_shape_image,
                               build_harmonic_image, build_robust_image)

print("\nBuilding 2023 feature image (server-side)...")
landsat_2023 = build_landsat_collection('2023-01-01', '2023-12-31', study_region)
monthly_2023 = {m: monthly_composite(m, landsat_2023) for m in range(1, 13)}

sources = {
    **{n: monthly_2023[i+1] for i, n in enumerate(
        ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])},
    'shapes':    build_shape_image(monthly_2023),
    'harm_NDVI': build_harmonic_image(landsat_2023, 'NDVI'),
    'harm_EVI':  build_harmonic_image(landsat_2023, 'EVI'),
    'harm_LSWI': build_harmonic_image(landsat_2023, 'LSWI'),
    'harm_GCVI': build_harmonic_image(landsat_2023, 'GCVI'),
    'robust':    build_robust_image(monthly_2023),
}

feature_image, feature_names = build_feature_image('v4', sources)
print(f"  {len(feature_names)} feature bands ready")

# ══════════════════════════════════════════════════════════════════════════════
# SAMPLE EACH CLASS
# ══════════════════════════════════════════════════════════════════════════════
print("\nSampling classes...")

all_fcs = []

for group_name, (cdl_codes, n_raw, out_code) in SAMPLE_PLAN.items():
    print(f"  {group_name:<20} codes={cdl_codes}  requesting {n_raw}...", end='', flush=True)

    # Build mask for these CDL codes
    mask = cdl_2023.eq(cdl_codes[0])
    for c in cdl_codes[1:]:
        mask = mask.Or(cdl_2023.eq(c))

    remapped = (cdl_2023.updateMask(mask)
                .remap(from_=ee.List(cdl_codes),
                       to=ee.List([1] * len(cdl_codes)),
                       defaultValue=0)
                .rename('class_idx').selfMask())

    try:
        samples = remapped.stratifiedSample(
            numPoints=n_raw,
            classBand='class_idx',
            region=study_region,
            scale=30,
            seed=SEED,
            geometries=True,
            dropNulls=True
        ).map(lambda f: f.set('cdl_code', out_code).set('cdl_group', group_name))

        n_got = samples.size().getInfo()
        print(f" got {n_got}")
        all_fcs.append(samples)
    except Exception as e:
        print(f" FAILED: {e}")

# Merge all
print("\nMerging all samples...")
merged_fc = all_fcs[0]
for fc in all_fcs[1:]:
    merged_fc = merged_fc.merge(fc)

# Sample v4 features at all points
print("Extracting v4 features at all sample points...")
sampled_fc = (
    feature_image.unmask(-9999)
    .sampleRegions(
        collection=merged_fc,
        properties=['cdl_code', 'cdl_group'],
        scale=30,
        geometries=True
    )
)

# ══════════════════════════════════════════════════════════════════════════════
# EXPORT TO GOOGLE DRIVE — needed for large sampleRegions computation
# Interactive mode times out on 19k pts x 42 complex bands; batch export works.
# ══════════════════════════════════════════════════════════════════════════════
DRIVE_FOLDER  = 'gee_exports'
DRIVE_FILE    = 'training_full_v4_7class'

print(f"\nExporting to Google Drive ({DRIVE_FOLDER}/{DRIVE_FILE}.csv)...")
task = ee.batch.Export.table.toDrive(
    collection=sampled_fc,
    description=DRIVE_FILE,
    folder=DRIVE_FOLDER,
    fileNamePrefix=DRIVE_FILE,
    fileFormat='CSV',
    selectors=feature_names + ['cdl_code', 'cdl_group', '.geo'],
)
task.start()
print(f"  Task ID: {task.id}")

while True:
    status = task.status()
    state  = status['state']
    if state == 'COMPLETED':
        print("  Export COMPLETE")
        break
    elif state in ('FAILED', 'CANCELLED'):
        raise RuntimeError(f"Export failed: {status.get('error_message','unknown')}")
    else:
        print(f"  Waiting ({state})...", end='\r')
        time.sleep(20)

# ══════════════════════════════════════════════════════════════════════════════
# READ FROM DRIVE — user must download CSV from Drive to data/ first
# OR we auto-download via Google Drive API if credentials allow
# ══════════════════════════════════════════════════════════════════════════════
# Check if file was auto-downloaded already
drive_local = os.path.join(DATA_DIR, f'{DRIVE_FILE}.csv')

if not os.path.exists(drive_local):
    print(f"\n{'='*55}")
    print("ACTION REQUIRED:")
    print(f"  1. Go to Google Drive -> '{DRIVE_FOLDER}' folder")
    print(f"  2. Download '{DRIVE_FILE}.csv'")
    print(f"  3. Save it to: {drive_local}")
    print(f"  4. Re-run this script from the '# READ CSV' section")
    print(f"{'='*55}")
    print("\nOR just run:  python process_drive_csv.py")
    raise SystemExit(0)

print(f"\nReading {drive_local}...")
df_raw = pd.read_csv(drive_local)

# ── Parse .geo column for lat/lon ─────────────────────────────────────────────
if '.geo' in df_raw.columns:
    import json as _json
    def parse_coords(geo_str):
        try:
            geo = _json.loads(str(geo_str))
            coords = geo.get('coordinates', [None, None])
            return round(float(coords[0]), 6), round(float(coords[1]), 6)
        except Exception:
            return None, None
    df_raw[['longitude', 'latitude']] = df_raw['.geo'].apply(
        lambda g: pd.Series(parse_coords(g))
    )
    df_raw = df_raw.drop(columns=['.geo'])

df = df_raw.copy()

# ── Drop rows with any missing feature values (-9999) ────────────────────────
feat_cols = [c for c in df.columns if c in feature_names]
before    = len(df)
df        = df[(df[feat_cols] != -9999).all(axis=1)]
dropped   = before - len(df)
print(f"  {len(df_raw)} rows read, dropped {dropped} with missing data ({len(df)} remain)")

# ── Add metadata ──────────────────────────────────────────────────────────────
df['crop_name']  = df['cdl_code'].map(CLASS_MAP).fillna('Unknown')
df['dataset']    = 'full_v4_7class'
df['region']     = 'Iowa+Illinois'
df['label_year'] = 2023
df['image_year'] = 2023

# Reproducible 80/20 split stratified by class
rng  = np.random.default_rng(seed=42)
df['split'] = 'train'
for code in df['cdl_code'].unique():
    idx  = df.index[df['cdl_code'] == code]
    test = rng.choice(idx, size=int(len(idx) * 0.2), replace=False)
    df.loc[test, 'split'] = 'test'

# ── Save ──────────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, 'raw_full.csv')
df.to_csv(out_path, index=False)
print(f"\nSaved: {out_path}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("CLASS DISTRIBUTION")
print("="*55)
summary = df.groupby('crop_name').agg(
    total=('cdl_code', 'count'),
    train=('split', lambda x: (x=='train').sum()),
    test=('split',  lambda x: (x=='test').sum()),
).sort_values('total', ascending=False)
print(summary.to_string())

print(f"\nTotal: {len(df)} rows | {len(feat_cols)} features | 7 classes")
classes_under_700 = summary[summary['total'] < 700]
if len(classes_under_700):
    print(f"\nWARNING — classes under 700 samples:")
    print(classes_under_700.to_string())
else:
    print("\nAll classes >= 700 samples")

# ── Update provenance ──────────────────────────────────────────────────────────
prov_path = os.path.join(DATA_DIR, 'provenance.json')
with open(prov_path) as f:
    prov = json.load(f)

prov['datasets']['raw_full.csv'] = {
    'source_asset': 'streamed_direct (no intermediate asset)',
    'downloaded_at': datetime.now().isoformat(),
    'rows': len(df),
    'classes': {str(k): v for k, v in CLASS_MAP.items()},
    'class_counts': df['crop_name'].value_counts().to_dict(),
    'feature_cols': feat_cols,
    'notes': (
        '7-class dataset (5 crops + Dbl_Crop + Other). '
        'v4 Landsat features. Iowa+Illinois 2023. '
        '80/20 stratified split per class.'
    ),
}
with open(prov_path, 'w') as f:
    json.dump(prov, f, indent=2)
print(f"Provenance updated: {prov_path}")
print("\nDone. Run train_local_models.py next.")
