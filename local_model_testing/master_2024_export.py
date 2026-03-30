# local_model_testing/master_2024_export.py
# Unified 2024 Export: Landsat + SAR + Robust Stats + Fixed Sampling.
#
# WHY: Fixes the 'Other' class bug in Renville and adds SAR to solve the McLean cloud problem.
#
# Run: ..\venv\Scripts\python local_model_testing/master_2024_export.py

import sys, os, time, json
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth import init_ee
init_ee()

import ee

from core import (build_landsat_collection, monthly_composite, add_indices)
from feature_registry import (build_feature_image, build_shape_image, 
                               build_harmonic_image, build_robust_image)

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
DRIVE_FOLDER = 'gee_exports'

# County definitions
COUNTIES = {
    'mclean':   {'fips': '17113', 'name': 'McLean County, IL'},
    'renville': {'fips': '27129', 'name': 'Renville County, MN'},
}

TARGET_CROP_CODES = [1, 5, 24, 28, 36, 26]
OTHER_GROUPS = {
    'other_pasture':   [176, 37],
    'other_forest':    [141, 142, 143],
    'other_developed': [121, 122, 123, 124],
    'other_water':     [111, 190, 195],
    'other_fallow':    [61, 63, 64],
}

def get_county_geom(fips):
    return (ee.FeatureCollection('TIGER/2018/Counties')
            .filter(ee.Filter.eq('GEOID', fips))
            .first().geometry())

def build_sar_image(year, region):
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(region)
          .filterDate(f'{year}-01-01', f'{year}-12-31')
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    
    m_jul = s1.filter(ee.Filter.calendarRange(7, 7, 'month')).mean()
    m_aug = s1.filter(ee.Filter.calendarRange(8, 8, 'month')).mean()
    s1_mean = s1.mean()
    s1_std  = s1.reduce(ee.Reducer.stdDev())
    
    return ee.Image.cat([
        m_jul.select('VH').rename('SAR_VH_jul'),
        m_aug.select('VH').rename('SAR_VH_aug'),
        m_jul.select('VV').rename('SAR_VV_jul'),
        m_aug.select('VV').rename('SAR_VV_aug'),
        s1_mean.select('VH').rename('SAR_VH_mean'),
        s1_mean.select('VV').rename('SAR_VV_mean'),
        s1_std.select('VH_stdDev').rename('SAR_VH_std'),
        s1_std.select('VV_stdDev').rename('SAR_VV_std'),
    ]).float()

def sample_class_fixed(cdl_img, codes, out_code, n, region, seed, group_name):
    # Fixed: No remapping to 0 before sampling
    mask = cdl_img.eq(codes[0])
    for c in codes[1:]: mask = mask.Or(cdl_img.eq(c))
    
    remapped = (cdl_img.updateMask(mask)
                .remap(from_=ee.List(codes), to=ee.List([1]*len(codes)), defaultValue=0)
                .rename('class_idx').selfMask())
    try:
        fc = remapped.stratifiedSample(
            numPoints=n, classBand='class_idx',
            region=region, scale=30, seed=seed, geometries=True, dropNulls=True
        ).map(lambda f: f.set('cdl_code', out_code, 'cdl_group', group_name))
        return fc
    except: return None

# Main Loop
for slug, info in COUNTIES.items():
    print(f"\nProcessing {info['name']} (2024)...")
    geom = get_county_geom(info['fips'])
    
    # CDL - Raw (No early remapping!)
    cdl_raw = (ee.ImageCollection('USDA/NASS/CDL')
               .filter(ee.Filter.calendarRange(2024, 2024, 'year'))
               .first().select('cropland').clip(geom))
    
    # Sampling
    all_fcs = []
    for code in TARGET_CROP_CODES:
        fc = sample_class_fixed(cdl_raw, [code], code, 200, geom, 2024, str(code))
        if fc: all_fcs.append(fc)
    
    for gname, codes in OTHER_GROUPS.items():
        fc = sample_class_fixed(cdl_raw, codes, 0, 50, geom, 2024, gname)
        if fc: all_fcs.append(fc)
        
    merged_pts = all_fcs[0]
    for fc in all_fcs[1:]: merged_pts = merged_pts.merge(fc)
    
    # Features
    ls = build_landsat_collection('2024-01-01', '2024-12-31', geom)
    m = {mo: monthly_composite(mo, ls) for mo in range(1, 13)}
    
    sources = {
        **{n: m[i+1] for i, n in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])},
        'shapes':    build_shape_image(m),
        'harm_NDVI': build_harmonic_image(ls, 'NDVI'),
        'harm_EVI':  build_harmonic_image(ls, 'EVI'),
        'harm_LSWI': build_harmonic_image(ls, 'LSWI'),
        'harm_GCVI': build_harmonic_image(ls, 'GCVI'),
        'robust':    build_robust_image(m),
    }
    
    landsat_img, landsat_names = build_feature_image('v4', sources)
    sar_img = build_sar_image(2024, geom)
    
    master_img = ee.Image.cat([landsat_img, sar_img]).unmask(-9999)
    
    # Export
    file_name = f'master_test_2024_{slug}'
    print(f"  Starting Export: {file_name}")
    task = ee.batch.Export.table.toDrive(
        collection=master_img.sampleRegions(collection=merged_pts, scale=30, properties=['cdl_code','cdl_group']),
        description=file_name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_name,
        fileFormat='CSV'
    )
    task.start()
    print(f"  Task ID: {task.id}")
