import pandas as pd
import numpy as np
import os

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')

def create_final_dataset():
    print("Creating final dataset (Landsat + SAR)...")
    
    # 1. Load original boosted data (Landsat features)
    df_train = pd.read_csv(os.path.join(DATA_DIR, 'raw_full_boosted.csv'))
    df_train['point_id'] = df_train.index
    
    # 2. Load SAR training data
    df_sar = pd.read_csv(os.path.join(DATA_DIR, 'sar_train_boosted.csv'))
    
    # Identify SAR feature columns
    sar_cols = [c for c in df_sar.columns if c.startswith('SAR_')]
    
    # 3. Merge
    # We use a left join to keep all original points, even if SAR is missing
    df_final = pd.merge(df_train, df_sar[['point_id'] + sar_cols], on='point_id', how='left')
    
    # Fill missing SAR values with -9999 (to keep it consistent with our mask logic)
    df_final[sar_cols] = df_final[sar_cols].fillna(-9999)
    
    # 4. Save
    out_path = os.path.join(DATA_DIR, 'raw_full_final.csv')
    df_final.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    print(f"  Total Rows: {len(df_final)}")
    print(f"  Features: {len([c for c in df_final.columns if not any(m in c for m in ['cdl', 'split', 'year', 'point_id', 'geo', 'dataset', 'region', 'crop_name', 'longitude', 'latitude'])])}")

if __name__ == "__main__":
    create_final_dataset()
