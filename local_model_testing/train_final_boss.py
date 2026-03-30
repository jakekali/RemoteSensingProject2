import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
OUT_DIR = os.path.join(HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}
CLASS_CODES = [1, 5, 24, 28, 36, 26, 0]
idx_to_code = {i: c for i, c in enumerate(CLASS_CODES)}
code_to_idx = {c: i for i, c in enumerate(CLASS_CODES)}

def train_and_evaluate():
    print("Training Final Boss Model (Landsat + SAR + Robust Stats)...")
    
    # 1. Load Training Data
    df_train_full = pd.read_csv(os.path.join(DATA_DIR, 'raw_full_final.csv'))
    
    # 2. Load one test set to find the feature intersection
    df_test_mclean = pd.read_csv(os.path.join(DATA_DIR, 'master_test_2024_mclean.csv'))
    
    # Identify common feature columns
    meta = ['cdl_code', 'cdl_group', 'longitude', 'latitude', 'crop_name', 'dataset', 'region', 'label_year', 'image_year', 'split', 'point_id', '.geo', 'system:index']
    train_feats = [c for c in df_train_full.columns if c not in meta]
    test_feats = [c for c in df_test_mclean.columns if c not in meta]
    
    features = sorted(list(set(train_feats) & set(test_feats)))
    print(f"  Using {len(features)} intersecting features.")
    
    # Prepare Training arrays
    train_df = df_train_full[df_train_full['split'] == 'train'].copy()
    X_train = train_df[features].replace(-9999, np.nan).values
    y_train = train_df['cdl_code'].values.astype(int)
    y_train_idx = np.array([code_to_idx[c] for c in y_train])
    
    sample_weights = compute_sample_weight('balanced', y=y_train)
    
    # 3. Train XGBoost
    model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train_idx, sample_weight=sample_weights)
    print("  Model trained.")
    
    # 4. Evaluate on 2024 Counties
    for slug, name in [('mclean', 'McLean IL (2024)'), ('renville', 'Renville MN (2024)')]:
        df_test = pd.read_csv(os.path.join(DATA_DIR, f'master_test_2024_{slug}.csv'))
        
        X_test = df_test[features].replace(-9999, np.nan).values
        y_test = df_test['cdl_code'].values.astype(int)
        
        preds_idx = model.predict(X_test)
        y_pred = np.array([idx_to_code[i] for i in preds_idx])
        
        acc = accuracy_score(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)
        
        print(f"\n--- {name} ---")
        print(f"  Acc: {acc:.1%} | Kappa: {kappa:.3f}")
        
        present = sorted(set(y_test))
        names = [CLASS_MAP.get(c, str(c)) for c in present]
        print(classification_report(y_test, y_pred, labels=present, target_names=names, zero_division=0))
        
        # Plot CM
        cm = confusion_matrix(y_test, y_pred, labels=present)
        cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens', xticklabels=names, yticklabels=names)
        plt.title(f"Final Boss CM: {name}")
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"final_cm_{slug}.png"))
        plt.close()

if __name__ == "__main__":
    train_and_evaluate()
