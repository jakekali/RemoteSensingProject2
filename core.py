# core.py
# Shared GEE setup, Landsat helpers, CDL constants, sampling utils.
# Import with:  from core import *

import warnings
warnings.filterwarnings('ignore')

import os, math, time
import ee
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT    = 'heroic-goal-412401'
ASSET_ROOT = f'projects/{PROJECT}/assets/project2'
OUT        = 'D:/remote_project_2/outputs'
os.makedirs(OUT, exist_ok=True)

# ── Counties for Generalization ───────────────────────────────────────────────
COUNTIES = [
    ('17113', 'mclean',   'McLean County, IL', 'Illinois'),
    ('27129', 'renville', 'Renville County, MN', 'Minnesota'),
]

# ── GEE init ──────────────────────────────────────────────────────────────────
ee.Initialize(project=PROJECT)

# ── CDL class names ───────────────────────────────────────────────────────────
_cdl_df   = pd.read_excel(
    'https://www.nass.usda.gov/Research_and_Science/Cropland/docs/cdl_codes_names.xlsx'
)
CDL_NAMES = _cdl_df.set_index('MasterCat')['Crop'].to_dict()

NON_CROP_CODES = {
    63, 64, 65, 81, 82, 83, 87, 88, 111, 112,
    121, 122, 123, 124, 131, 141, 142, 143, 152, 176, 190, 195
}
WSP_HA = 3.95   # Washington Square Park in hectares

# ── Config / Mode Management ──────────────────────────────────────────────────
class Config:
    MODE = 'FULL'  # 'FAST' (small n) or 'FULL' (large n)
    
    # Sample sizes
    N_TRAIN = 5000 if MODE == 'FULL' else 100
    N_TEST  = 1000 if MODE == 'FULL' else 100
    
    @classmethod
    def get_main_region(cls):
        """Returns the primary study region (IA+IL)."""
        return study_region

    @classmethod
    def get_county_regions(cls):
        """Returns geometries for the generalization counties."""
        fc = ee.FeatureCollection('TIGER/2018/Counties')
        geoms = {}
        for fips, slug, name, state in COUNTIES:
            geoms[slug] = fc.filter(ee.Filter.eq('GEOID', fips)).first().geometry()
        return geoms

# ── Target classes ────────────────────────────────────────────────────────────
# 1, 5, 36, 24, 28 = Target Crops
# 26 = Double Crop
# 0  = Everything Else (Man-made, Forest, Water, etc.)
TARGET_CROP_CODES = [1, 5, 36, 24, 28, 26]
SAMP_CODES = TARGET_CROP_CODES + [0]
SAMP_NAMES = ['Corn', 'Soybeans', 'Alfalfa', 'Winter_Wheat', 'Oats', 'Dbl_Crop', 'Other']

TARGET_CODES = SAMP_CODES
CLASS_NAMES  = SAMP_NAMES

# ══════════════════════════════════════════════════════════════════════════════
# CDL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_honest_cdl(year, region):
    """
    Returns a CDL image where target crops are kept and everything else is 0.
    This implements the 'Everything Else' logic for the Other class.
    """
    cdl = (ee.ImageCollection('USDA/NASS/CDL')
           .filter(ee.Filter.calendarRange(year, year, 'year'))
           .first().select('cropland').clip(region))
    
    # Create mask for target crops
    mask = cdl.eq(TARGET_CROP_CODES[0])
    for code in TARGET_CROP_CODES[1:]:
        mask = mask.Or(cdl.eq(code))
    
    return cdl.where(mask.Not(), 0)

# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def hierarchical_classify(sample_fc, feature_image, feature_names, trees=100, prob_threshold=0.6):
    """
    Trains and applies a two-layered hierarchical classifier.
    Layer 1: Binary (Crop vs Non-Crop) with Probability Threshold
    Layer 2: Multi-class (Specific Crop)
    """
    # 1. Prepare Training Data
    # L1: is_crop property (1 if not 0, else 0)
    train_l1 = sample_fc.map(lambda f: f.set('is_crop', ee.Number(f.get('cdl_code')).neq(0).int()))
    # L2: Crop-only (filter out 0)
    train_l2 = sample_fc.filter(ee.Filter.neq('cdl_code', 0))

    # 2. Train Models
    rf_l1 = ee.Classifier.smileRandomForest(trees, seed=42).train(train_l1, 'is_crop', feature_names)
    rf_l2 = ee.Classifier.smileRandomForest(trees, seed=42).train(train_l2, 'cdl_code', feature_names)

    # 3. Classify Image or FeatureCollection
    is_img = isinstance(feature_image, ee.Image)
    
    if is_img:
        # Probabilistic Gatekeeper on Image
        prob_img = feature_image.classify(rf_l1.setOutputMode('PROBABILITY'))
        crop_img = feature_image.classify(rf_l2)
        # Apply threshold: if prob < threshold, result is 0
        final = crop_img.where(prob_img.lt(prob_threshold), 0)
        return final, rf_l1, rf_l2
    else:
        # Logic for FeatureCollection (used in Accuracy scripts)
        classified_l1 = feature_image.classify(rf_l1.setOutputMode('PROBABILITY'), 'crop_prob')
        classified_both = classified_l1.classify(rf_l2, 'l2_pred')
        
        final_fc = classified_both.map(lambda f: f.set(
            'classification',
            ee.Number(ee.Algorithms.If(
                ee.Number(f.get('crop_prob')).gte(prob_threshold),
                f.get('l2_pred'),
                0
            ))
        ))
        return final_fc, rf_l1, rf_l2

CROP_COLORS = {
    'Corn':                     '#E8A020',
    'Soybeans':                 '#2D6A4F',
    'Alfalfa':                  '#4A7C59',
    'Winter_Wheat':             '#C9A84C',
    'Oats':                     '#06D6A0',
    'Dbl_Crop_WinWht/Soybeans': '#9B59B6',
}

MONTH_NAMES = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

# ── Study region ──────────────────────────────────────────────────────────────
iowa_illinois = (ee.FeatureCollection('TIGER/2018/States')
                 .filter(ee.Filter.inList('NAME', ['Iowa', 'Illinois'])))
study_region  = iowa_illinois.geometry()

# ── 2023 CDL ──────────────────────────────────────────────────────────────────
cdl_2023 = (
    ee.ImageCollection('USDA/NASS/CDL')
    .filter(ee.Filter.calendarRange(2023, 2023, 'year'))
    .first()
    .select('cropland')
    .clip(study_region)
)

# ══════════════════════════════════════════════════════════════════════════════
# LANDSAT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def mask_landsat_clouds(image):
    qa = image.select('QA_PIXEL')
    cloud_mask = (
        qa.bitwiseAnd(1 << 1).eq(0)
        .And(qa.bitwiseAnd(1 << 3).eq(0))
        .And(qa.bitwiseAnd(1 << 4).eq(0))
        .And(qa.bitwiseAnd(1 << 5).eq(0))
    )
    scaled = image.select('SR_B.').multiply(0.0000275).add(-0.2)
    return image.addBands(scaled, overwrite=True).updateMask(cloud_mask)


def add_indices(image):
    nir, red, green, blue, swir1 = (
        image.select('SR_B5'), image.select('SR_B4'),
        image.select('SR_B3'), image.select('SR_B2'),
        image.select('SR_B6'),
    )
    ndvi = nir.subtract(red).divide(nir.add(red)).rename('NDVI')
    evi  = image.expression(
        '2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))',
        {'NIR': nir, 'RED': red, 'BLUE': blue}
    ).rename('EVI')
    lswi = nir.subtract(swir1).divide(nir.add(swir1)).rename('LSWI')
    gcvi = nir.divide(green).subtract(1).rename('GCVI')
    return image.addBands([ndvi, evi, lswi, gcvi])


def build_landsat_collection(start, end, region):
    """Merged, cloud-masked, indexed L8+L9 collection for a date range."""
    kwargs = dict(filterBounds=region)
    l8 = (ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
          .filterDate(start, end).filterBounds(region)
          .map(mask_landsat_clouds).map(add_indices))
    l9 = (ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
          .filterDate(start, end).filterBounds(region)
          .map(mask_landsat_clouds).map(add_indices))
    return l8.merge(l9).sort('system:time_start')


def monthly_composite(month, collection):
    """Cloud-free median composite for a single calendar month."""
    return collection.filter(ee.Filter.calendarRange(month, month, 'month')).median()


# ── 2023 collection + monthly composites ──────────────────────────────────────
landsat_2023 = build_landsat_collection('2023-01-01', '2023-12-31', study_region)

MONTHLY_2023 = {m: monthly_composite(m, landsat_2023) for m in range(1, 13)}
jan, feb, mar, apr = [MONTHLY_2023[m] for m in [1,2,3,4]]
may, jun, jul, aug = [MONTHLY_2023[m] for m in [5,6,7,8]]
sep, oct, nov, dec = [MONTHLY_2023[m] for m in [9,10,11,12]]

# ══════════════════════════════════════════════════════════════════════════════
# STRATIFIED SAMPLING
# ══════════════════════════════════════════════════════════════════════════════

def make_stratified_samples(n_per_class, codes, cdl_image=None, region=None, seed=42):
    """Stratified random sample from the CDL. Returns FC with 'cdl_code' property."""
    img = cdl_image if cdl_image is not None else cdl_2023
    rgn = region    if region    is not None else study_region
    remap_from = ee.List(codes)
    remap_to   = ee.List(list(range(1, len(codes) + 1)))
    code_list  = ee.List(codes)
    remapped   = (img.remap(from_=remap_from, to=remap_to, defaultValue=0)
                  .rename('class_idx').selfMask())
    return remapped.stratifiedSample(
        numPoints=n_per_class, classBand='class_idx',
        region=rgn, scale=30, seed=seed, geometries=True,
        dropNulls=True
    ).map(lambda f: f.set(
        'cdl_code', code_list.get(f.getNumber('class_idx').subtract(1).int())
    ))

# ══════════════════════════════════════════════════════════════════════════════
# GEE ASSET EXPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def export_classifier_to_asset(classifier, description, asset_id):
    """Start a Classifier -> Asset export task."""
    task = ee.batch.Export.classifier.toAsset(
        classifier=classifier,
        description=description,
        assetId=asset_id
    )
    task.start()
    print(f'  Classifier export started: {description}  (task {task.id})')
    return task


def wait_for_tasks(tasks, poll_sec=30):
    """Block until all tasks complete. Raises on any failure."""
    pending = list(tasks)
    while pending:
        still_running = []
        for task in pending:
            status = task.status()
            state  = status['state']
            if state in ('COMPLETED',):
                print(f'  [done]  {status["description"]}')
            elif state in ('FAILED', 'CANCELLED'):
                raise RuntimeError(
                    f'Task failed: {status["description"]}  '
                    f'error: {status.get("error_message","unknown")}'
                )
            else:
                still_running.append(task)
        pending = still_running
        if pending:
            print(f'  Waiting — {len(pending)} task(s) still running...')
            time.sleep(poll_sec)


def asset_exists(asset_id):
    try:
        ee.data.getAsset(asset_id)
        return True
    except Exception:
        return False


if __name__ == '__main__':
    print('core.py OK')
    print(f'  TARGET_CODES : {TARGET_CODES}')
    print(f'  ASSET_ROOT   : {ASSET_ROOT}')
