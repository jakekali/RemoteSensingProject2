import sys, os, json
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
MODEL_PATH = os.path.join(HERE, 'models', 'xgb_boosted.json')

# Map
CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}
CLASS_CODES = [1, 5, 24, 28, 36, 26, 0]
idx_to_code = {i: c for i, c in enumerate(CLASS_CODES)}

# Load model
print(f"Loading model: {MODEL_PATH}")
model = XGBClassifier()
model.load_model(MODEL_PATH)

def evaluate(county_name, csv_file):
    path = os.path.join(DATA_DIR, csv_file)
    if not os.path.exists(path):
        print(f"  {county_name} data not found at {path}")
        return
    
    df = pd.read_csv(path)
    # Parse .geo
    if '.geo' in df.columns:
        def _parse(g):
            try:
                import json as _j
                c = _j.loads(str(g))['coordinates']
                return round(float(c[0]),6), round(float(c[1]),6)
            except: return None, None
        df[['longitude','latitude']] = df['.geo'].apply(lambda g: pd.Series(_parse(g)))
        df = df.drop(columns=['.geo'])

    # Filter invalid
    meta = ['cdl_code','crop_name','dataset','region','label_year','image_year','split','longitude','latitude','cdl_group']
    feat_cols = [c for c in df.columns if c not in meta and c != 'split']
    # Replace -9999 with NaN (XGBoost handles this)
    X = df[feat_cols].replace(-9999, np.nan).values
    y_true = df['cdl_code'].values.astype(int)

    # Predict
    preds_idx = model.predict(X)
    y_pred = np.array([idx_to_code[i] for i in preds_idx])

    # Stats
    acc = accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    
    print(f"\n--- {county_name} ---")
    print(f"  Acc: {acc:.1%} | Kappa: {kappa:.3f}")
    
    # Per-class recall
    present = sorted(set(y_true))
    names = [CLASS_MAP.get(c, str(c)) for c in present]
    print(classification_report(y_true, y_pred, labels=present, target_names=names, zero_division=0))

evaluate("McLean IL (2024)", "test_2024_mclean.csv")
evaluate("Renville MN (2024)", "test_2024_renville.csv")
