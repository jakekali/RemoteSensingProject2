import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

# Paths
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, 'data')
OUT_DIR = os.path.join(HERE, 'outputs')
MODEL_PATH = os.path.join(HERE, 'models', 'xgb_boosted.json')

# Map
CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}
CLASS_CODES = [1, 5, 24, 28, 36, 26, 0]
idx_to_code = {i: c for i, c in enumerate(CLASS_CODES)}

def full_cloud_and_error_analysis():
    # 1. Cloud Analysis
    def get_cloud_stats(csv_file):
        df = pd.read_csv(os.path.join(DATA_DIR, csv_file))
        # Find all feature columns
        feat_cols = [c for c in df.columns if any(m in c for m in ['NDVI', 'EVI', 'GCVI', 'LSWI'])]
        # Count -9999 (masked) in each column
        stats = (df[feat_cols] == -9999).mean() * 100
        return stats

    mclean_clouds = get_cloud_stats("test_2024_mclean.csv")
    renville_clouds = get_cloud_stats("test_2024_renville.csv")

    # Plot Cloud Stats
    plt.figure(figsize=(14, 6))
    mclean_clouds.sort_index().plot(kind='bar', color='red', alpha=0.6, label='McLean (Broken)', position=0, width=0.4)
    renville_clouds.sort_index().plot(kind='bar', color='blue', alpha=0.6, label='Renville (Good)', position=1, width=0.4)
    plt.title("Percentage of Masked (Cloudy) Data per Feature (2024)")
    plt.ylabel("% Masked")
    plt.legend()
    plt.xticks(rotation=90, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cloud_comparison.png"))
    print(f"Saved: {os.path.join(OUT_DIR, 'cloud_comparison.png')}")

    # 2. Confusion Matrices
    model = XGBClassifier()
    model.load_model(MODEL_PATH)

    for slug, name in [('mclean', 'McLean IL'), ('renville', 'Renville MN')]:
        df = pd.read_csv(os.path.join(DATA_DIR, f"test_2024_{slug}.csv"))
        meta = ['cdl_code','crop_name','dataset','region','label_year','image_year','split','longitude','latitude','cdl_group']
        feat_cols = [c for c in df.columns if c not in meta and c != 'split' and c != '.geo']
        
        X = df[feat_cols].replace(-9999, np.nan).values
        y_true = df['cdl_code'].values.astype(int)
        
        preds_idx = model.predict(X)
        y_pred = np.array([idx_to_code[i] for i in preds_idx])
        
        present_codes = sorted(set(y_true))
        present_names = [CLASS_MAP.get(c, str(c)) for c in present_codes]
        
        cm = confusion_matrix(y_true, y_pred, labels=present_codes)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                    xticklabels=present_names, yticklabels=present_names)
        plt.title(f"Normalized Confusion Matrix: {name} (2024)")
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"cm_2024_{slug}.png"))
        print(f"Saved: {os.path.join(OUT_DIR, f'cm_2024_{slug}.png')}")

if __name__ == "__main__":
    full_cloud_and_error_analysis()
