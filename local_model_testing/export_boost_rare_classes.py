# local_model_testing/export_boost_rare_classes.py
# Pull extra Alfalfa + Oats samples from 2023 AND 2022 to boost rare classes.
#
# WHY 2022 TOO:
#   Training on multiple years forces the model to learn phenological PATTERNS
#   rather than 2023-specific spectral values. Alfalfa and Oats have consistent
#   growth signatures across years — adding 2022 gives inter-annual robustness,
#   which directly helps generalize to 2024.
#
# STRATEGY:
#   2023 additional: 1500 raw -> expect ~600 survive (40% rate) -> +600 pts
#   2022 additional: 1500 raw -> expect ~600 survive (40% rate) -> +600 pts
#   Target: Alfalfa ~405+600+600 = ~1600 total, Oats ~356+600+600 = ~1550 total
#
# OUTPUTS (Google Drive -> gee_exports/):
#   boost_alfalfa_oats_2023.csv    extra 2023 samples
#   boost_alfalfa_oats_2022.csv    extra 2022 samples
#
# Run: ..\venv\Scripts\python export_boost_rare_classes.py

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth import init_ee
init_ee(verbose=True)

import ee

DRIVE_FOLDER = 'gee_exports'
DATA_DIR     = os.path.join(os.path.dirname(__file__), 'data')

# ── Study region: Iowa + Illinois (same as training) ──────────────────────────
study_region = (
    ee.FeatureCollection('TIGER/2018/States')
    .filter(ee.Filter.inList('NAME', ['Iowa', 'Illinois']))
    .geometry()
)

from core import build_landsat_collection, monthly_composite
from feature_registry import (build_feature_image, build_shape_image,
                               build_harmonic_image, build_robust_image)

BOOST_CLASSES = {
    36: 'Alfalfa',
    28: 'Oats',
}
N_RAW     = 1500   # per class per year — expect ~600 survive
YEARS     = [2023, 2022]
SEEDS     = {2023: 123, 2022: 456}   # different seeds from original (seed=42)

CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}


def build_sources(year, region):
    ls      = build_landsat_collection(f'{year}-01-01', f'{year}-12-31', region)
    monthly = {m: monthly_composite(m, ls) for m in range(1, 13)}
    return ls, {
        **{n: monthly[i+1] for i, n in enumerate(
            ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])},
        'shapes':    build_shape_image(monthly),
        'harm_NDVI': build_harmonic_image(ls, 'NDVI'),
        'harm_EVI':  build_harmonic_image(ls, 'EVI'),
        'harm_LSWI': build_harmonic_image(ls, 'LSWI'),
        'harm_GCVI': build_harmonic_image(ls, 'GCVI'),
        'robust':    build_robust_image(monthly),
    }


tasks = {}

for year in YEARS:
    print(f"\n{'='*55}")
    print(f"YEAR: {year}")
    print(f"{'='*55}")

    cdl = (ee.ImageCollection('USDA/NASS/CDL')
           .filter(ee.Filter.calendarRange(year, year, 'year'))
           .first().select('cropland').clip(study_region))

    print(f"  Building {year} Landsat feature sources...")
    ls, sources = build_sources(year, study_region)
    feature_image, feature_names = build_feature_image('v4', sources)
    print(f"  {len(feature_names)} feature bands ready")

    all_fcs = []
    for code, name in BOOST_CLASSES.items():
        print(f"  Sampling {name} (code={code}, n={N_RAW})...", end='', flush=True)

        mask     = cdl.eq(code)
        remapped = (cdl.updateMask(mask)
                    .remap(from_=ee.List([code]), to=ee.List([1]), defaultValue=0)
                    .rename('class_idx').selfMask())

        samples = remapped.stratifiedSample(
            numPoints=N_RAW, classBand='class_idx',
            region=study_region, scale=30,
            seed=SEEDS[year], geometries=True, dropNulls=True
        ).map(lambda f: f.set('cdl_code', code,
                               'cdl_group', name,
                               'label_year', year,
                               'image_year', year))

        n = samples.size().getInfo()
        print(f" got {n} points")
        all_fcs.append(samples)

    # Merge and extract features
    merged  = all_fcs[0]
    for fc in all_fcs[1:]:
        merged = merged.merge(fc)

    print(f"  Sampling v4 features (server-side)...")
    sampled = (feature_image.unmask(-9999)
               .sampleRegions(
                   collection=merged,
                   properties=['cdl_code', 'cdl_group', 'label_year', 'image_year'],
                   scale=30,
                   geometries=True
               ))

    # Export to Drive
    file_name = f'boost_alfalfa_oats_{year}'
    print(f"  Exporting to Drive: {DRIVE_FOLDER}/{file_name}.csv ...")
    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=file_name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_name,
        fileFormat='CSV',
        selectors=feature_names + ['cdl_code','cdl_group','label_year','image_year','.geo'],
    )
    task.start()
    tasks[year] = {'task': task, 'file_name': file_name}
    print(f"  Task ID: {task.id}")

# ── Poll ──────────────────────────────────────────────────────────────────────
print(f"\nWaiting for {len(tasks)} export task(s)...")
pending = dict(tasks)
while pending:
    still = {}
    for yr, t in pending.items():
        state = t['task'].status()['state']
        if state == 'COMPLETED':
            print(f"  [DONE] {yr}")
        elif state in ('FAILED','CANCELLED'):
            print(f"  [FAIL] {yr}: {t['task'].status().get('error_message','?')}")
        else:
            still[yr] = t
    pending = still
    if pending:
        print(f"  Still running: {list(pending.keys())}...", end='\r')
        time.sleep(20)

# ── Instructions ──────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("EXPORT COMPLETE")
print(f"{'='*55}")
print("\nDownload from Google Drive -> 'gee_exports':")
for yr, t in tasks.items():
    fname = t['file_name']
    print(f"  {fname}.csv")
    print(f"    -> Save to: {os.path.join(DATA_DIR, fname + '.csv')}")
print("\nThen run: merge_boost_samples.py")
