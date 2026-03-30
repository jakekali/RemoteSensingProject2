# local_model_testing/export_sar_to_drive.py
# Export 2022, 2023, and 2024 SAR features (Sentinel-1) for training and test points.
#
# WHY: SAR is weather-independent (sees through clouds). 
#      This should solve the 44% accuracy problem in McLean IL 2024.
#
# WHAT:
#   - Build SAR monthly features (VV, VH) for each year.
#   - Load existing point locations (lat/lon) from training and test CSVs.
#   - Extract SAR values at these points.
#   - Export to Drive as:
#       sar_train_boosted.csv    (for 2022+2023 training points)
#       sar_test_mclean_2024.csv (for 2024 McLean points)
#       sar_test_renville_2024.csv (for 2024 Renville points)
#
# Run: ..\venv\Scripts\python local_model_testing/export_sar_to_drive.py

import sys, os, time, json
import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from auth import init_ee
# Force User OAuth to bypass service account permission issues
init_ee(verbose=True, force_oauth=True)

import ee

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
DRIVE_FOLDER = 'gee_exports'
os.makedirs(DATA_DIR, exist_ok=True)

def build_sar_image(year, region):
    """
    Build a multi-band SAR image for a given year.
    Features: VV/VH means for Jun, Jul, Aug, Sep + Annual stats.
    """
    start = f'{year}-01-01'
    end   = f'{year}-12-31'
    
    # Filter Sentinel-1
    s1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
          .filterBounds(region)
          .filterDate(start, end)
          .filter(ee.Filter.eq('instrumentMode', 'IW'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
          .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
    
    # Pre-process: mean for each target month
    def get_month_mean(month, coll):
        return coll.filter(ee.Filter.calendarRange(month, month, 'month')).mean()

    # snapshots
    m_jun = get_month_mean(6, s1)
    m_jul = get_month_mean(7, s1)
    m_aug = get_month_mean(8, s1)
    m_sep = get_month_mean(9, s1)
    
    # annual stats
    s1_mean = s1.mean()
    s1_std  = s1.reduce(ee.Reducer.stdDev())
    
    bands = [
        m_jun.select('VH').rename('SAR_VH_jun'),
        m_jul.select('VH').rename('SAR_VH_jul'),
        m_aug.select('VH').rename('SAR_VH_aug'),
        m_sep.select('VH').rename('SAR_VH_sep'),
        m_jun.select('VV').rename('SAR_VV_jun'),
        m_jul.select('VV').rename('SAR_VV_jul'),
        m_aug.select('VV').rename('SAR_VV_aug'),
        m_sep.select('VV').rename('SAR_VV_sep'),
        s1_mean.select('VH').rename('SAR_VH_mean'),
        s1_mean.select('VV').rename('SAR_VV_mean'),
        s1_std.select('VH_stdDev').rename('SAR_VH_std'),
        s1_std.select('VV_stdDev').rename('SAR_VV_std'),
    ]
    
    return ee.Image.cat(bands).clip(region).float()

def csv_to_fc(df):
    """Convert pandas df (with lat/lon) to ee.FeatureCollection."""
    features = []
    # Use index to keep unique IDs
    for idx, row in df.iterrows():
        # Minimal properties to keep CSV small
        props = {'point_id': int(idx), 'image_year': int(row['image_year'])}
        # Add cdl_code if present
        if 'cdl_code' in row: props['cdl_code'] = int(row['cdl_code'])
            
        geom = ee.Geometry.Point([row['longitude'], row['latitude']])
        features.append(ee.Feature(geom, props))
    
    # If list is too large, chunk it
    if len(features) > 2000:
        return ee.FeatureCollection(features)
    return ee.FeatureCollection(features)

def export_points(points_df, name_prefix, year_override=None):
    """
    Export SAR features for a set of points.
    If year_override is set, use that for SAR imagery.
    Otherwise, use the 'image_year' column from the points.
    """
    # Parse .geo if longitude is missing
    if 'longitude' not in points_df.columns and '.geo' in points_df.columns:
        print(f"    Parsing .geo for {name_prefix}...")
        def _parse(g):
            import json as _j
            try:
                c = _j.loads(str(g))['coordinates']
                return round(float(c[0]),6), round(float(c[1]),6)
            except: return None, None
        points_df[['longitude','latitude']] = points_df['.geo'].apply(
            lambda g: pd.Series(_parse(g)))
        points_df = points_df.dropna(subset=['longitude', 'latitude'])

    # 1. Define region as bounding box of points + buffer
    lons = points_df['longitude']
    lats = points_df['latitude']
    region = ee.Geometry.Rectangle([lons.min()-0.1, lats.min()-0.1, lons.max()+0.1, lats.max()+0.1])
    
    # 2. Extract features
    # If points have mixed years (training), we need to split and join or do a map.
    # To keep it simple, we'll assume a specific year if passed, or split by year.
    
    years = [year_override] if year_override else points_df['image_year'].unique().tolist()
    
    all_sampled = []
    for yr in sorted(years):
        print(f"    Building SAR {yr} for {name_prefix}...")
        sar_img = build_sar_image(yr, region)
        
        # Filter points for this year
        yr_df = points_df[points_df['image_year'] == yr] if not year_override else points_df
        if yr_df.empty: continue
            
        fc = csv_to_fc(yr_df)
        sampled = sar_img.sampleRegions(
            collection=fc,
            properties=['point_id', 'cdl_code', 'image_year'],
            scale=10, # Sentinel-1 is 10m
            geometries=False
        )
        all_sampled.append(sampled)
    
    merged = all_sampled[0]
    for s in all_sampled[1:]:
        merged = merged.merge(s)
        
    # 3. Export
    file_name = f'sar_{name_prefix}'
    print(f"    Exporting {file_name} to Drive...")
    
    task = ee.batch.Export.table.toDrive(
        collection=merged,
        description=file_name,
        folder=DRIVE_FOLDER,
        fileNamePrefix=file_name,
        fileFormat='CSV'
    )
    task.start()
    return task

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

# 1. Training Points (2022 + 2023)
print("\nProcessing Training Points (raw_full_boosted.csv)...")
df_train = pd.read_csv(os.path.join(DATA_DIR, 'raw_full_boosted.csv'))
task_train = export_points(df_train, 'train_boosted')
print(f"  Task ID: {task_train.id}")

# 2. McLean 2024
print("\nProcessing McLean 2024 Points...")
df_mclean = pd.read_csv(os.path.join(DATA_DIR, 'test_2024_mclean.csv'))
# We need to add image_year if not there
if 'image_year' not in df_mclean.columns: df_mclean['image_year'] = 2024
task_mclean = export_points(df_mclean, 'test_mclean_2024', year_override=2024)
print(f"  Task ID: {task_mclean.id}")

# 3. Renville 2024
print("\nProcessing Renville 2024 Points...")
df_renville = pd.read_csv(os.path.join(DATA_DIR, 'test_2024_renville.csv'))
if 'image_year' not in df_renville.columns: df_renville['image_year'] = 2024
task_renville = export_points(df_renville, 'test_renville_2024', year_override=2024)
print(f"  Task ID: {task_renville.id}")

print("\n--- Summary ---")
print("Background export tasks started for 2022, 2023, and 2024 SAR data.")
print("The files will appear in Google Drive folder: gee_exports/")
print("Once done, download them to data/ to merge with Landsat features.")
