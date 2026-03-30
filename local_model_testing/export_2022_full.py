# local_model_testing/export_2022_full.py
# Export full 7-class dataset for 2022 — all categories.
#
# WHY 2022:
#   Inter-annual training (2022 + 2023) teaches the model crop PHENOLOGICAL
#   PATTERNS rather than year-specific spectral values.  This directly improves
#   generalisation to 2024 test counties (McLean IL, Renville MN).
#
# COVERS ALL 7 CLASSES:
#   1  = Corn
#   5  = Soybeans
#   24 = Winter_Wheat
#   28 = Oats           (also covered by boost script; extra diversity here)
#   36 = Alfalfa        (also covered by boost script; extra diversity here)
#   26 = Dbl_Crop
#    0 = Other  (pasture, forest, developed, water/wetland, fallow sub-groups)
#
# SEED: 789  (distinct from 2023 seed=42 and boost seeds 123/456)
#   => different geographic points from any previously sampled year.
#
# OUTPUT  (Google Drive -> gee_exports/):
#   training_2022_v4_7class.csv
#
# Auth: theta-grid-99720 service account (auth.py)
#
# Run:
#   cd D:\remote_project_2
#   venv\Scripts\python local_model_testing/export_2022_full.py

import sys, os, time
# core.py / feature_registry.py live in the project root (one level up)
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)   # -> D:/remote_project_2  (core, feature_registry)
sys.path.insert(0, _HERE)   # -> D:/remote_project_2/local_model_testing  (auth)

print("Initialising GEE (theta-grid-99720 service account)...")
from auth import init_ee
init_ee(verbose=True)

import ee

DRIVE_FOLDER = 'gee_exports'
DRIVE_FILE   = 'training_2022_v4_7class'
YEAR         = 2022
SEED         = 789          # distinct from seed=42 (2023), 123/456 (boost)

CLASS_MAP = {
    1:  'Corn',
    5:  'Soybeans',
    24: 'Winter_Wheat',
    28: 'Oats',
    36: 'Alfalfa',
    26: 'Dbl_Crop',
    0:  'Other',
}

# ── Study region: Iowa + Illinois (matches 2023 training region) ──────────────
study_region = (
    ee.FeatureCollection('TIGER/2018/States')
    .filter(ee.Filter.inList('NAME', ['Iowa', 'Illinois']))
    .geometry()
)

# ── 2022 CDL ──────────────────────────────────────────────────────────────────
cdl_2022 = (
    ee.ImageCollection('USDA/NASS/CDL')
    .filter(ee.Filter.calendarRange(YEAR, YEAR, 'year'))
    .first().select('cropland').clip(study_region)
)

# ── Sampling plan (mirrors export_full_dataset.py) ────────────────────────────
# key: group_name -> (cdl_codes_to_sample, n_raw_to_request, output_cdl_code)
SAMPLE_PLAN = {
    'Corn':             ([1],                   1000,  1),
    'Soybeans':         ([5],                   1000,  5),
    'Winter_Wheat':     ([24],                  1000, 24),
    'Oats':             ([28],                  1000, 28),
    'Alfalfa':          ([36],                  1000, 36),
    'Dbl_Crop':         ([26],                  2000, 26),
    'other_pasture':    ([176, 37],             2000,  0),
    'other_forest':     ([141, 142, 143],       2000,  0),
    'other_developed':  ([121, 122, 123, 124],  2000,  0),
    'other_water_wet':  ([111, 190, 195],       2000,  0),
    'other_fallow':     ([61, 63, 64],          2000,  0),
}

# ── Build 2022 Landsat feature image ─────────────────────────────────────────
from core import build_landsat_collection, monthly_composite
from feature_registry import (
    build_feature_image, build_shape_image,
    build_harmonic_image, build_robust_image,
)

print(f"\nBuilding {YEAR} Landsat feature image (server-side)...")
landsat_2022 = build_landsat_collection(f'{YEAR}-01-01', f'{YEAR}-12-31', study_region)
monthly_2022 = {m: monthly_composite(m, landsat_2022) for m in range(1, 13)}

sources = {
    **{n: monthly_2022[i+1] for i, n in enumerate(
        ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])},
    'shapes':    build_shape_image(monthly_2022),
    'harm_NDVI': build_harmonic_image(landsat_2022, 'NDVI'),
    'harm_EVI':  build_harmonic_image(landsat_2022, 'EVI'),
    'harm_LSWI': build_harmonic_image(landsat_2022, 'LSWI'),
    'harm_GCVI': build_harmonic_image(landsat_2022, 'GCVI'),
    'robust':    build_robust_image(monthly_2022),
}

feature_image, feature_names = build_feature_image('v4', sources)
print(f"  {len(feature_names)} feature bands ready")

# ── Sample each class ─────────────────────────────────────────────────────────
print("\nSampling all classes from 2022 CDL...")
all_fcs = []

for group_name, (cdl_codes, n_raw, out_code) in SAMPLE_PLAN.items():
    print(f"  {group_name:<22} codes={str(cdl_codes):<20} n={n_raw}", end='', flush=True)

    mask = cdl_2022.eq(cdl_codes[0])
    for c in cdl_codes[1:]:
        mask = mask.Or(cdl_2022.eq(c))

    remapped = (
        cdl_2022.updateMask(mask)
        .remap(from_=ee.List(cdl_codes),
               to=ee.List([1] * len(cdl_codes)),
               defaultValue=0)
        .rename('class_idx').selfMask()
    )

    try:
        samples = remapped.stratifiedSample(
            numPoints=n_raw,
            classBand='class_idx',
            region=study_region,
            scale=30,
            seed=SEED,
            geometries=True,
            dropNulls=True,
        ).map(lambda f: f.set(
            'cdl_code',   out_code,
            'cdl_group',  group_name,
            'label_year', YEAR,
            'image_year', YEAR,
        ))

        n_got = samples.size().getInfo()
        print(f"  -> got {n_got}")
        all_fcs.append(samples)

    except Exception as e:
        print(f"  -> FAILED: {e}")

# ── Merge & extract v4 features ───────────────────────────────────────────────
print(f"\nMerging {len(all_fcs)} groups...")
merged_fc = all_fcs[0]
for fc in all_fcs[1:]:
    merged_fc = merged_fc.merge(fc)

print("Extracting v4 features at all points (server-side)...")
sampled_fc = (
    feature_image.unmask(-9999)
    .sampleRegions(
        collection=merged_fc,
        properties=['cdl_code', 'cdl_group', 'label_year', 'image_year'],
        scale=30,
        geometries=True,
    )
)

# ── Export to Drive ───────────────────────────────────────────────────────────
print(f"\nSubmitting Drive export: {DRIVE_FOLDER}/{DRIVE_FILE}.csv ...")
task = ee.batch.Export.table.toDrive(
    collection=sampled_fc,
    description=DRIVE_FILE,
    folder=DRIVE_FOLDER,
    fileNamePrefix=DRIVE_FILE,
    fileFormat='CSV',
    selectors=feature_names + ['cdl_code', 'cdl_group', 'label_year', 'image_year', '.geo'],
)
task.start()
print(f"  Task ID : {task.id}")
print(f"  Monitor : https://code.earthengine.google.com/tasks")

# ── Poll until done ───────────────────────────────────────────────────────────
print("\nWaiting for export to complete...")
while True:
    status = task.status()
    state  = status['state']
    if state == 'COMPLETED':
        print("\n  [DONE] Export complete!")
        break
    elif state in ('FAILED', 'CANCELLED'):
        raise RuntimeError(f"Export failed: {status.get('error_message', 'unknown')}")
    else:
        print(f"  State: {state} ...", end='\r')
        time.sleep(30)

print(f"\n{'='*60}")
print("NEXT STEPS")
print(f"{'='*60}")
print(f"1. Download from Drive -> gee_exports/{DRIVE_FILE}.csv")
print(f"2. Save to:  local_model_testing/data/{DRIVE_FILE}.csv")
print(f"3. Run:      python local_model_testing/merge_2022_2023.py")
print(f"{'='*60}")
