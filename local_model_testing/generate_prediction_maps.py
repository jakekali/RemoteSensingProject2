import pandas as pd
import numpy as np
import os
import json
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
OUT_DIR = os.path.join(HERE, 'outputs')
MODEL_PATH = os.path.join(HERE, 'models', 'xgb_boosted.json') # Or final boss logic

# Classes and Colors
CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}
CLASS_CODES = [1, 5, 24, 28, 36, 26, 0]
idx_to_code = {i: c for i, c in enumerate(CLASS_CODES)}
code_to_idx = {c: i for i, c in enumerate(CLASS_CODES)}

# Hex colors matching core.py conventions
COLORS = {
    'Corn': '#E8A020',
    'Soybeans': '#2D6A4F',
    'Alfalfa': '#4A7C59',
    'Winter_Wheat': '#C9A84C',
    'Oats': '#06D6A0',
    'Dbl_Crop': '#9B59B6',
    'Other': '#CCCCCC' # Light grey for background
}

def generate_spatial_maps():
    print("Generating spatial prediction maps...")
    
    # 1. Re-train/Load the Final Boss Logic
    # (Since we didn't save the final boss .json, let's retrain it quickly in memory)
    df_train_full = pd.read_csv(os.path.join(DATA_DIR, 'raw_full_final.csv'))
    df_test_mclean = pd.read_csv(os.path.join(DATA_DIR, 'master_test_2024_mclean.csv'))
    
    meta = ['cdl_code', 'cdl_group', 'longitude', 'latitude', 'crop_name', 'dataset', 'region', 'label_year', 'image_year', 'split', 'point_id', '.geo', 'system:index']
    train_feats = [c for c in df_train_full.columns if c not in meta]
    test_feats = [c for c in df_test_mclean.columns if c not in meta]
    features = sorted(list(set(train_feats) & set(test_feats)))
    
    train_df = df_train_full[df_train_full['split'] == 'train'].copy()
    X_train = train_df[features].replace(-9999, np.nan).values
    y_train = train_df['cdl_code'].values.astype(int)
    y_train_idx = np.array([code_to_idx[c] for c in y_train])
    
    from sklearn.utils.class_weight import compute_sample_weight
    sw = compute_sample_weight('balanced', y=y_train)
    
    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train_idx, sample_weight=sw)
    
    # 2. Process Counties
    for slug, name in [('mclean', 'McLean County, IL'), ('renville', 'Renville County, MN')]:
        print(f"  Processing {name}...")
        df = pd.read_csv(os.path.join(DATA_DIR, f'master_test_2024_{slug}.csv'))
        
        # Parse coordinates
        def _parse(g):
            try:
                c = json.loads(str(g))['coordinates']
                return round(float(c[0]),6), round(float(c[1]),6)
            except: return None, None
        df[['lon','lat']] = df['.geo'].apply(lambda g: pd.Series(_parse(g)))
        
        X = df[features].replace(-9999, np.nan).values
        y_true = df['cdl_code'].values.astype(int)
        
        preds_idx = model.predict(X)
        y_pred = np.array([idx_to_code[i] for i in preds_idx])
        
        # Add names
        df['True_Name'] = [CLASS_MAP.get(c, 'Other') for c in y_true]
        df['Pred_Name'] = [CLASS_MAP.get(c, 'Other') for c in y_pred]
        df['Is_Correct'] = (y_true == y_pred)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        fig.suptitle(f"Spatial Prediction Map: {name} (2024)\nFinal Boss Model (Landsat + SAR + Stats)", fontsize=16, fontweight='bold')
        
        # Plot settings
        scatter_size = 30
        
        for i, (col, title) in enumerate([('True_Name', 'Ground Truth (USDA CDL)'), ('Pred_Name', 'Model Predictions')]):
            ax = axes[i]
            for class_name, color in COLORS.items():
                mask = df[col] == class_name
                ax.scatter(df.loc[mask, 'lon'], df.loc[mask, 'lat'], 
                           c=color, label=class_name, s=scatter_size, alpha=0.8, edgecolors='none')
            
            ax.set_title(title, fontsize=14)
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.grid(True, alpha=0.3)
            if i == 1: # Legend on the second plot
                ax.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        out_path = os.path.join(OUT_DIR, f"spatial_map_{slug}.png")
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        print(f"    Saved: {out_path}")
        
        # 3. Error Map (Optional but helpful)
        plt.figure(figsize=(10, 8))
        plt.scatter(df.loc[df['Is_Correct'], 'lon'], df.loc[df['Is_Correct'], 'lat'], 
                    c='green', label='Correct', s=15, alpha=0.4)
        plt.scatter(df.loc[~df['Is_Correct'], 'lon'], df.loc[~df['Is_Correct'], 'lat'], 
                    c='red', label='Error', s=35, alpha=0.8)
        plt.title(f"Prediction Errors: {name}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        err_path = os.path.join(OUT_DIR, f"spatial_errors_{slug}.png")
        plt.savefig(err_path, dpi=150)
        plt.close('all')
        print(f"    Saved: {err_path}")

if __name__ == "__main__":
    generate_spatial_maps()
