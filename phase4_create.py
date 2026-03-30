# phase4_create.py — Unified Generalization: Evaluate Across Years (Multi-County)
# Supports both FAST and FULL modes via Config.
#
# Logic:
#   1. Training Data (2023): Drawn from the WHOLE regional study area (IA+IL).
#   2. Testing Data (2024): Drawn from SPECIFIC counties (McLean, Story).
#   3. For each Version + County:
#       a. Trains Hierarchical RF on 2023 Regional data.
#       b. Evaluates on 2024 County data.
#       c. Exports metrics to GEE assets.
#
# Run with: venv/Scripts/python phase4_create.py

from core import *
from feature_registry import (
    FEATURE_SETS, build_feature_image,
    build_shape_image, build_harmonic_image,
    build_context_image, build_robust_image
)
import json

# ── Setup ─────────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"PHASE 4: GENERALIZATION (Mode: {Config.MODE})")
print("=" * 60)

main_region = Config.get_main_region()
county_regions = Config.get_county_regions()
n_train = Config.N_TRAIN
n_test  = Config.N_TEST
VERSIONS = ['v1', 'v2', 'v3', 'v4', 'v7']
YEAR = 2024

print(f"Training Region: IA+IL (2023)")
print(f"Testing Counties: {list(county_regions.keys())} (2024)")
print(f"Samples: {n_train}/class train, {n_test}/class test")

# ══════════════════════════════════════════════════════════════════════════════
# 1. BUILD REGIONAL TRAINING SOURCES (2023)
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding Regional Training Sources (2023)...")
landsat_2023 = build_landsat_collection('2023-01-01', '2023-12-31', main_region)
monthly_2023 = {m: monthly_composite(m, landsat_2023) for m in range(1, 13)}
sources_2023 = {
    **{m_name: monthly_2023[i+1] for i, m_name in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])},
    'shapes':    build_shape_image(monthly_2023),
    'harm_NDVI': build_harmonic_image(landsat_2023, 'NDVI'),
    'harm_EVI':  build_harmonic_image(landsat_2023, 'EVI'),
    'harm_LSWI': build_harmonic_image(landsat_2023, 'LSWI'),
    'harm_GCVI': build_harmonic_image(landsat_2023, 'GCVI'),
    'robust':    build_robust_image(monthly_2023),
    'context':   build_context_image(monthly_2023, main_region),
}

cdl_2023_honest = build_honest_cdl(2023, main_region)
train_samples = make_stratified_samples(n_train, SAMP_CODES, cdl_image=cdl_2023_honest, region=main_region, seed=42)

def get_available_codes(cdl_img, geom, candidate_codes):
    """Returns only those codes that actually exist in the clipped CDL."""
    counts = cdl_img.reduceRegion(
        reducer=ee.Reducer.frequencyHistogram(),
        geometry=geom,
        scale=30,
        maxPixels=1e9
    ).get('cropland').getInfo()
    found_codes = [int(float(c)) for c in counts.keys()]
    return [c for c in candidate_codes if c in found_codes]

# ══════════════════════════════════════════════════════════════════════════════
# 2. MAIN LOOP — Per County
# ══════════════════════════════════════════════════════════════════════════════
all_tasks = []

for county_slug, county_geom in county_regions.items():
    print(f"\nProcessing County: {county_slug.upper()}")
    
    # Build 2024 Sources for THIS county
    landsat_2024 = build_landsat_collection('2024-01-01', '2024-12-31', county_geom)
    monthly_2024 = {m: monthly_composite(m, landsat_2024) for m in range(1, 13)}
    sources_2024 = {
        **{m_name: monthly_2024[i+1] for i, m_name in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])},
        'shapes':    build_shape_image(monthly_2024),
        'harm_NDVI': build_harmonic_image(landsat_2024, 'NDVI'),
        'harm_EVI':  build_harmonic_image(landsat_2024, 'EVI'),
        'harm_LSWI': build_harmonic_image(landsat_2024, 'LSWI'),
        'harm_GCVI': build_harmonic_image(landsat_2024, 'GCVI'),
        'robust':    build_robust_image(monthly_2024),
        'context':   build_context_image(monthly_2024, county_geom),
    }
    
    cdl_2024_honest = build_honest_cdl(2024, county_geom)
    
    # ── Export CDL Reference Map ──────────────────────────────────────────
    print(f"  Generating 2024 CDL reference map...")
    palette = ['E8A020', '2D6A4F', '4A7C59', 'C9A84C', '06D6A0', '9B59B6']
    vis_params = {'min': 1, 'max': 36, 'palette': ['808080'] * 36}
    for i, code in enumerate([1, 5, 36, 24, 28, 26]):
        vis_params['palette'][code-1] = palette[i]

    cdl_url = cdl_2024_honest.getThumbURL({'min': 0, 'max': 36, 'palette': vis_params['palette'], 'dimensions': 256, 'region': county_geom})
    import requests
    with open(f'{OUT}/phase4_{county_slug}_cdl_map.png', 'wb') as f:
        f.write(requests.get(cdl_url).content)

    # Dynamically filter codes that exist in this county
    available_codes = get_available_codes(cdl_2024_honest, county_geom, SAMP_CODES)
    print(f"  Available classes: {available_codes}")
    
    acc_samples = make_stratified_samples(n_test, available_codes, cdl_image=cdl_2024_honest, region=county_geom, seed=99)

    for version in VERSIONS:
        # Asset IDs for the pre-trained models from Phase 3
        id_l1 = f'{ASSET_ROOT}/p3_model_l1_{Config.MODE}_{version}'
        id_l2 = f'{ASSET_ROOT}/p3_model_l2_{Config.MODE}_{version}'
        
        if not (asset_exists(id_l1) and asset_exists(id_l2)):
            print(f"  Version {version}: Trained models not found in assets. Run Phase 3 first.")
            continue

        asset_id = f'{ASSET_ROOT}/p4_res_{Config.MODE}_{county_slug}_{version}'
        if asset_exists(asset_id):
            print(f"  Version {version}: Metrics already exist, skipping.")
            continue

        feat_img_test, feature_names = build_feature_image(version, sources_2024)
        
        # 1. Load the "Robot's Brain" from Phase 3
        print(f"  Version {version}: Loading pre-trained Phase 3 model...")
        rf_l1 = ee.Classifier.load(id_l1)
        rf_l2 = ee.Classifier.load(id_l2)

        # 2. Deploy to 2024 County (Points only for Metrics)
        # Sampling 2024 points
        test_fc = feat_img_test.unmask(-9999).select(feature_names).sampleRegions(
            collection=acc_samples, properties=['cdl_code'], scale=30, geometries=False
        )
        
        # Filter out cases where satellite clues are missing (clouds/masking)
        test_fc = test_fc.filter(ee.Filter.notNull(feature_names))
        print(f"  DEBUG: test_fc size after null filter: {test_fc.size().getInfo()}")

        # Apply the pre-trained hierarchical logic
        # L1: Crop vs Not-Crop
        classified_l1 = test_fc.classify(rf_l1, 'l1_pred')
        # L2: Specific Crops
        classified_both = classified_l1.classify(rf_l2, 'l2_pred')
        
        # Mapping to final classification
        # L1 is 1/0. L2 is CropCode. L1*L2 = CropCode if field, 0 if not.
        final_fc = classified_both.filter(ee.Filter.notNull(['l1_pred', 'l2_pred'])).map(lambda f: f.set(
            'classification_final',
            ee.Number(f.get('l1_pred')).multiply(ee.Number(f.get('l2_pred')))
        ))
        
        # 3. Accuracy Result (Calculate GEE-side for speed and stability)
        # REMAP for ErrorMatrix (which hates 0 and non-contiguous codes)
        clean_codes = [int(c) for c in SAMP_CODES]
        remap_from = ee.List(clean_codes)
        remap_to   = ee.List(list(range(1, len(clean_codes) + 1)))
        
        # REMAP direct on final_fc
        eval_remap = final_fc.map(lambda f: f.set({
            'true_idx': ee.Number(ee.Algorithms.If(remap_from.contains(ee.Number(f.get('cdl_code')).int()), 
                                                   remap_from.indexOf(ee.Number(f.get('cdl_code')).int()).add(1), 
                                                   0)),
            'pred_idx': ee.Number(ee.Algorithms.If(remap_from.contains(ee.Number(f.get('classification_final')).int()), 
                                                   remap_from.indexOf(ee.Number(f.get('classification_final')).int()).add(1), 
                                                   0))
        }))
        
        # FINAL FILTER: Ensure both indices are > 0 (found in list)
        eval_remap = eval_remap.filter(ee.Filter.gt('true_idx', 0)).filter(ee.Filter.gt('pred_idx', 0))
        
        cm = eval_remap.errorMatrix('true_idx', 'pred_idx', remap_to)
        
        results_fc = ee.FeatureCollection([
            ee.Feature(county_geom.centroid(100), {
                'accuracy': cm.accuracy(),
                'kappa':    cm.kappa(),
                'version':  version,
                'county':   county_slug,
                'mode':     Config.MODE,
                'n_features': len(feature_names)
            })
        ])
        
        task = ee.batch.Export.table.toAsset(
            collection=results_fc,
            description=f'p4_task_{county_slug}_{version}',
            assetId=asset_id
        )
        task.start()
        print(f"  Task started: {county_slug}/{version} (id: {task.id})")
        all_tasks.append(task)

if all_tasks:
    print(f"\nWaiting for {len(all_tasks)} Phase 4 tasks...")
    wait_for_tasks(all_tasks)

# ── Summary Report ────────────────────────────────────────────────────────────
print(f"\n=== PHASE 4 SUMMARY (Mode: {Config.MODE}) ===")
for county_slug in county_regions.keys():
    print(f"\n  County: {county_slug.upper()}")
    for version in VERSIONS:
        asset_id = f'{ASSET_ROOT}/p4_res_{Config.MODE}_{county_slug}_{version}'
        if asset_exists(asset_id):
            res = ee.FeatureCollection(asset_id).first().toDictionary().getInfo()
            print(f"    {version.upper()}: Accuracy={res['accuracy']:.1%}  Kappa={res['kappa']:.3f}")

print("\nPhase 4 complete.")
