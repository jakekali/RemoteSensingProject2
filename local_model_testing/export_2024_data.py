# local_model_testing/export_2024_data.py
# Export 2024 feature data for generalization testing.
#
# WHY: Training was on 2023 Iowa+Illinois. Now we test how well models
# generalize to a NEW YEAR (2024) and NEW COUNTIES:
#   McLean County, IL  (FIPS 17113) — same state as training (in-distribution)
#   Renville County, MN (FIPS 27129) — different state (out-of-distribution)
#
# WHAT:
#   - Builds 2024 Landsat monthly composites for each county
#   - Samples 2024 CDL labels (same 7-class structure as training)
#   - Extracts v4 features at sample points
#   - Exports to Google Drive as CSV (batch task, no timeout risk)
#
# OUTPUTS (Google Drive -> gee_exports/):
#   test_2024_mclean.csv      McLean IL, 2024 imagery + 2024 CDL labels
#   test_2024_renville.csv    Renville MN, 2024 imagery + 2024 CDL labels
#
# After download, save to data/ and run evaluate_2024.py
#
# Run: ..\venv\Scripts\python export_2024_data.py

import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth import init_ee
init_ee(verbose=True)

import ee

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
DRIVE_FOLDER = 'gee_exports'
os.makedirs(DATA_DIR, exist_ok=True)

# ── County definitions ────────────────────────────────────────────────────────
COUNTIES = {
    'mclean':   {'fips': '17113', 'name': 'McLean County, IL',
                 'state': 'Illinois',  'note': 'in-distribution (trained on IL)'},
    'renville': {'fips': '27129', 'name': 'Renville County, MN',
                 'state': 'Minnesota', 'note': 'out-of-distribution (no MN training data)'},
}

# ── Class structure (must match training) ─────────────────────────────────────
TARGET_CROP_CODES = [1, 5, 24, 28, 36, 26]   # 5 crops + dbl_crop
OTHER_CODES = [176, 37, 141, 142, 143, 121, 122, 123, 124, 111, 190, 195, 61, 63, 64]
CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat', 28:'Oats',
             36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}

# Samples per class (smaller than training — this is test data)
N_PER_CROP  = 200   # per target crop class
N_OTHER     = 200   # total Other (across all sub-groups)
N_PER_OTHER_GROUP = 50

SEED = 2024

# ── Shared helpers ────────────────────────────────────────────────────────────
from core import (build_landsat_collection, monthly_composite,
                  mask_landsat_clouds, add_indices)
from feature_registry import (build_feature_image, build_shape_image,
                               build_harmonic_image, build_robust_image)

def get_county_geom(fips):
    return (ee.FeatureCollection('TIGER/2018/Counties')
            .filter(ee.Filter.eq('GEOID', fips))
            .first().geometry())


def build_honest_cdl(year, region):
    """CDL with non-target codes remapped to 0."""
    cdl = (ee.ImageCollection('USDA/NASS/CDL')
           .filter(ee.Filter.calendarRange(year, year, 'year'))
           .first().select('cropland').clip(region))
    mask = cdl.eq(TARGET_CROP_CODES[0])
    for c in TARGET_CROP_CODES[1:]:
        mask = mask.Or(cdl.eq(c))
    return cdl.where(mask.Not(), 0)


def sample_class(cdl_img, codes, out_code, n, region, seed, group_name=None):
    """Sample n points for given CDL codes, label as out_code."""
    mask = cdl_img.eq(codes[0])
    for c in codes[1:]:
        mask = mask.Or(cdl_img.eq(c))
    remapped = (cdl_img.updateMask(mask)
                .remap(from_=ee.List(codes),
                       to=ee.List([1]*len(codes)),
                       defaultValue=0)
                .rename('class_idx').selfMask())
    try:
        fc = remapped.stratifiedSample(
            numPoints=n, classBand='class_idx',
            region=region, scale=30, seed=seed,
            geometries=True, dropNulls=True
        ).map(lambda f: f.set('cdl_code', out_code,
                               'cdl_group', group_name or str(out_code)))
        return fc
    except Exception as e:
        print(f"    WARNING: sampling failed for codes={codes}: {e}")
        return None


def build_sources_2024(region):
    """Build all v4 feature sources from 2024 Landsat for a given region."""
    ls = build_landsat_collection('2024-01-01', '2024-12-31', region)
    monthly = {m: monthly_composite(m, ls) for m in range(1, 13)}
    return {
        **{n: monthly[i+1] for i, n in enumerate(
            ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])},
        'shapes':    build_shape_image(monthly),
        'harm_NDVI': build_harmonic_image(ls, 'NDVI'),
        'harm_EVI':  build_harmonic_image(ls, 'EVI'),
        'harm_LSWI': build_harmonic_image(ls, 'LSWI'),
        'harm_GCVI': build_harmonic_image(ls, 'GCVI'),
        'robust':    build_robust_image(monthly),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP — one Drive export per county
# ══════════════════════════════════════════════════════════════════════════════
tasks = {}

for slug, info in COUNTIES.items():
    print(f"\n{'='*55}")
    print(f"COUNTY: {info['name']}  [{info['note']}]")
    print(f"{'='*55}")

    geom    = get_county_geom(info['fips'])
    cdl     = build_honest_cdl(2024, geom)

    # ── Check CDL 2024 availability ───────────────────────────────────────────
    try:
        _ = cdl.reduceRegion(ee.Reducer.count(), geom, 1000, maxPixels=1e6).getInfo()
        print("  CDL 2024: available")
        label_year = 2024
    except Exception:
        print("  CDL 2024: not available — falling back to CDL 2023")
        cdl = build_honest_cdl(2023, geom)
        label_year = 2023

    # ── Sample points ─────────────────────────────────────────────────────────
    print("  Sampling points...")
    all_fcs = []

    # Target crops
    for code in TARGET_CROP_CODES:
        fc = sample_class(cdl, [code], code, N_PER_CROP, geom, SEED,
                          CLASS_MAP.get(code, str(code)))
        if fc is not None:
            n = fc.size().getInfo()
            print(f"    {CLASS_MAP.get(code,code):<15} {n} pts")
            all_fcs.append(fc)

    # Other sub-groups
    other_groups = {
        'other_pasture':   [176, 37],
        'other_forest':    [141, 142, 143],
        'other_developed': [121, 122, 123, 124],
        'other_water':     [111, 190, 195],
        'other_fallow':    [61, 63, 64],
    }
    for gname, codes in other_groups.items():
        fc = sample_class(cdl, codes, 0, N_PER_OTHER_GROUP, geom, SEED, gname)
        if fc is not None:
            n = fc.size().getInfo()
            print(f"    {gname:<15} {n} pts")
            all_fcs.append(fc)

    if not all_fcs:
        print("  ERROR: no samples collected, skipping county")
        continue

    merged = all_fcs[0]
    for fc in all_fcs[1:]:
        merged = merged.merge(fc)

    total = merged.size().getInfo()
    print(f"  Total sample points: {total}")

    # ── Build 2024 feature image ───────────────────────────────────────────────
    print("  Building 2024 feature sources...")
    sources = build_sources_2024(geom)
    feature_image, feature_names = build_feature_image('v4', sources)
    print(f"  {len(feature_names)} feature bands")

    # ── Sample features ────────────────────────────────────────────────────────
    print("  Sampling v4 features at points (server-side)...")
    sampled = (feature_image.unmask(-9999)
               .sampleRegions(
                   collection=merged,
                   properties=['cdl_code', 'cdl_group'],
                   scale=30,
                   geometries=True
               ))

    # ── Export to Drive ────────────────────────────────────────────────────────
    file_name = f'test_2024_{slug}'
    print(f"  Exporting to Drive: {DRIVE_FOLDER}/{file_name}.csv ...")

    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=file_name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_name,
        fileFormat='CSV',
        selectors=feature_names + ['cdl_code', 'cdl_group', '.geo'],
    )
    task.start()
    tasks[slug] = {'task': task, 'info': info,
                   'label_year': label_year, 'file_name': file_name}
    print(f"  Task ID: {task.id}")

# ── Poll all tasks ─────────────────────────────────────────────────────────────
print(f"\nWaiting for {len(tasks)} export task(s)...")
pending = dict(tasks)
while pending:
    still = {}
    for slug, t in pending.items():
        state = t['task'].status()['state']
        if state == 'COMPLETED':
            print(f"  [DONE] {slug}")
        elif state in ('FAILED', 'CANCELLED'):
            err = t['task'].status().get('error_message','?')
            print(f"  [FAIL] {slug}: {err}")
        else:
            still[slug] = t
    pending = still
    if pending:
        print(f"  Waiting ({', '.join(pending.keys())})...", end='\r')
        time.sleep(20)

# ── Instructions ───────────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print("EXPORT COMPLETE")
print(f"{'='*55}")
print(f"\nDownload from Google Drive -> '{DRIVE_FOLDER}':")
for slug, t in tasks.items():
    info = t['info']
    print(f"  {t['file_name']}.csv  ({info['name']}, {info['note']})")
    print(f"    -> Save to: {os.path.join(DATA_DIR, t['file_name']+'.csv')}")

print(f"\nThen run: evaluate_2024.py")
