# phase3_create.py — Unified Classification: Build, Train, Evaluate
# Supports both FAST (iteration) and FULL (production) modes via Config.
#
# Logic:
#   1. Sets up region and sample size based on core.Config.
#   2. Builds all feature source images (Monthly, Shape, Harmonic, Robust, Context).
#   3. Generates "Honest" CDL (Everything Else = 0).
#   4. Trains Hierarchical RF models for all requested versions in parallel.
#   5. Exports accuracy metrics to GEE assets.
#
# Run with: venv/Scripts/python phase3_create.py

from core import *
from feature_registry import (
    FEATURE_SETS, build_feature_image,
    build_shape_image, build_harmonic_image,
    build_context_image, build_robust_image
)
import json

# ── Setup ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"PHASE 3: CLASSIFICATION (Mode: {Config.MODE})")
print("=" * 60)

region = Config.get_main_region()
n_train = Config.N_TRAIN
VERSIONS = ['v1', 'v2', 'v3', 'v4', 'v5', 'v6']

print(f"Region: IA+IL")
print(f"Samples: {n_train}/class")

# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD FEATURE SOURCE IMAGES
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding feature source images...")
landsat_2023 = build_landsat_collection('2023-01-01', '2023-12-31', region)
monthly_2023 = {m: monthly_composite(m, landsat_2023) for m in range(1, 13)}

sources = {
    **{m_name: monthly_2023[i+1] for i, m_name in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])},
    'shapes':    build_shape_image(monthly_2023),
    'harm_NDVI': build_harmonic_image(landsat_2023, 'NDVI'),
    'harm_EVI':  build_harmonic_image(landsat_2023, 'EVI'),
    'harm_LSWI': build_harmonic_image(landsat_2023, 'LSWI'),
    'harm_GCVI': build_harmonic_image(landsat_2023, 'GCVI'),
    'robust':    build_robust_image(monthly_2023),
    'context':   build_context_image(monthly_2023, region),
}

# ══════════════════════════════════════════════════════════════════════════════
# 2. SAMPLING
# ══════════════════════════════════════════════════════════════════════════════
print(f"Drawing training samples from 2023 CDL...")
cdl_honest = build_honest_cdl(2023, region)

samples = make_stratified_samples(
    n_train, SAMP_CODES, 
    cdl_image=cdl_honest, 
    region=region, 
    seed=42
)

# ══════════════════════════════════════════════════════════════════════════════
# 3. MAIN EXECUTION — Tasks for each version
# ══════════════════════════════════════════════════════════════════════════════
print("\nLaunching GEE Export Tasks...")
all_tasks = []

for version in VERSIONS:
    asset_id = f'{ASSET_ROOT}/p3_results_{Config.MODE}_{version}'
    if asset_exists(asset_id):
        print(f"  Version {version}: Asset exists, skipping.")
        continue

    feature_image, feature_names = build_feature_image(version, sources)
    
    # Prep Sample Points
    sampled_fc = (
        feature_image.unmask(-9999)
        .select(feature_names)
        .sampleRegions(collection=samples, properties=['cdl_code'], scale=30, geometries=False)
        .randomColumn(seed=42)
    )
    
    train_fc = sampled_fc.filter(ee.Filter.lt('random', 0.8))
    test_fc  = sampled_fc.filter(ee.Filter.gte('random', 0.8))
    
    # Train & Classify (Hierarchical)
    test_final, rf_l1, rf_l2 = hierarchical_classify(train_fc, test_fc, feature_names)
    
    # Accuracy Result
    cm = test_final.errorMatrix('cdl_code', 'classification')
    results_fc = ee.FeatureCollection([
        ee.Feature(region.centroid(100), {
            'accuracy': cm.accuracy(),
            'kappa':    cm.kappa(),
            'version':  version,
            'mode':     Config.MODE
        })
    ])
    
    # 1. Export Metrics
    task_metrics = ee.batch.Export.table.toAsset(
        collection=results_fc,
        description=f'p3_metrics_{Config.MODE}_{version}',
        assetId=asset_id
    )
    task_metrics.start()
    all_tasks.append(task_metrics)

    # 2. Export Classifiers (The Robot's Brain)
    id_l1 = f'{ASSET_ROOT}/p3_model_l1_{Config.MODE}_{version}'
    id_l2 = f'{ASSET_ROOT}/p3_model_l2_{Config.MODE}_{version}'
    
    task_l1 = export_classifier_to_asset(rf_l1, f'p3_model_l1_{version}', id_l1)
    task_l2 = export_classifier_to_asset(rf_l2, f'p3_model_l2_{version}', id_l2)
    all_tasks.extend([task_l1, task_l2])

    print(f"  Tasks started for {version}: Metrics + 2 Classifiers")

if all_tasks:
    print(f"\nWaiting for {len(all_tasks)} Phase 3 tasks...")
    wait_for_tasks(all_tasks)

# ── Summary Report ────────────────────────────────────────────────────────────
print(f"\n=== PHASE 3 SUMMARY (Mode: {Config.MODE}) ===")
for version in VERSIONS:
    asset_id = f'{ASSET_ROOT}/p3_results_{Config.MODE}_{version}'
    if asset_exists(asset_id):
        res = ee.FeatureCollection(asset_id).first().toDictionary().getInfo()
        print(f"  {version.upper()}: Accuracy={res['accuracy']:.1%}  Kappa={res['kappa']:.3f}")

print("\nPhase 3 complete.")
