# plot_spatial_maps.py
# Create spatial scatter maps of 2024 predictions for McLean IL + Renville MN.
# Loads fresh test CSVs, applies best model, plots predicted vs CDL reference.
#
# Run: ..\venv\Scripts\python plot_spatial_maps.py

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, cohen_kappa_score
from xgboost import XGBClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs')
ROOT_OUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

CLASS_MAP = {0:'Other', 1:'Corn', 5:'Soybeans', 24:'Winter Wheat',
             26:'Dbl Crop', 28:'Oats', 36:'Alfalfa'}
COLORS    = {0:'#9E9E9E', 1:'#E8A020', 5:'#2D6A4F',
             24:'#C9A84C', 26:'#8B4513', 28:'#A8D5BA', 36:'#9B59B6'}

COUNTIES = {
    'mclean':   'McLean County, IL  (in-distribution)',
    'renville': 'Renville County, MN  (out-of-distribution)',
}

# ── Load training data ────────────────────────────────────────────────────────
print("Loading training data...")
df_tr_full = pd.read_csv(os.path.join(DATA_DIR, 'raw_full_boosted.csv'))
meta = ['cdl_code','crop_name','dataset','region','label_year',
        'image_year','split','longitude','latitude','cdl_group']
feat_cols = [c for c in df_tr_full.columns if c not in meta]

df_tr = df_tr_full[df_tr_full['split']=='train'].dropna(subset=feat_cols)
X_tr = df_tr[feat_cols].values
y_tr = df_tr['cdl_code'].values.astype(int)

classes     = np.sort(np.unique(y_tr))
code_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_code = {i: c for c, i in code_to_idx.items()}
sw_tr       = compute_sample_weight('balanced', y=y_tr)
y_tr_idx    = np.array([code_to_idx[c] for c in y_tr])

# Load best Optuna params
metrics_path = os.path.join(OUT_DIR, 'local_metrics.json')
with open(metrics_path) as f:
    saved = json.load(f)
tuned_params = saved.get('XGB_tuned', {}).get('best_params', None)

print("Training XGB_tuned...")
if tuned_params:
    model = XGBClassifier(**tuned_params, eval_metric='mlogloss',
                          n_jobs=-1, random_state=42, verbosity=0)
else:
    model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='mlogloss', n_jobs=-1, random_state=42, verbosity=0)
model.fit(X_tr, y_tr_idx, sample_weight=sw_tr)

# ── Process each county ───────────────────────────────────────────────────────
for slug, county_name in COUNTIES.items():
    csv_path = os.path.join(DATA_DIR, f'test_2024_{slug}.csv')
    if not os.path.exists(csv_path):
        print(f"SKIP {slug}: file not found")
        continue

    print(f"\n{county_name}")
    df = pd.read_csv(csv_path)

    # Parse geometry
    if '.geo' in df.columns:
        def _parse(g):
            try:
                import json as _j
                c = _j.loads(str(g))['coordinates']
                return float(c[0]), float(c[1])
            except: return None, None
        coords = df['.geo'].apply(lambda g: pd.Series(_parse(g), index=['lon','lat']))
        df = pd.concat([df.drop(columns=['.geo']), coords], axis=1)

    test_feat = [c for c in feat_cols if c in df.columns]
    df[test_feat] = df[test_feat].replace(-9999, float('nan'))
    df = df.dropna(subset=['lon','lat'])

    X_te = df[test_feat].values
    preds_idx = model.predict(X_te)
    preds = np.array([idx_to_code.get(i, -1) for i in preds_idx])
    y_true = df['cdl_code'].values.astype(int)

    acc = accuracy_score(y_true, preds)
    kap = cohen_kappa_score(y_true, preds)
    print(f"  Acc={acc:.1%}  Kappa={kap:.3f}  n={len(df)}")

    # ── Figure: CDL reference | Predicted ────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'2024 Crop Classification — {county_name}\n'
                 f'XGB_tuned (2023 Iowa+IL → 2024)  |  '
                 f'Acc={acc:.1%}  κ={kap:.3f}  n={len(df)}',
                 fontsize=11, fontweight='bold')

    for ax, labels, title in [
        (axes[0], y_true, 'CDL 2024 (reference)'),
        (axes[1], preds,  'XGB_tuned predicted'),
    ]:
        for code in sorted(set(labels)):
            mask = labels == code
            ax.scatter(df['lon'].values[mask], df['lat'].values[mask],
                       c=COLORS.get(code, '#000000'), s=12, alpha=0.7,
                       label=CLASS_MAP.get(code, str(code)), linewidths=0)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=0.2)

    # Shared legend
    present_codes = sorted(set(y_true) | set(preds))
    patches = [mpatches.Patch(facecolor=COLORS.get(c,'#000'), label=CLASS_MAP.get(c,str(c)))
               for c in present_codes if c >= 0]
    fig.legend(handles=patches, loc='lower center', ncol=len(patches),
               fontsize=9, bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    out_path = os.path.join(ROOT_OUT, f'phase4_{slug}_spatial_map.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # ── Error map: correct (green) vs wrong (red) ─────────────────────────────
    correct = preds == y_true
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(df['lon'].values[~correct], df['lat'].values[~correct],
               c='#E74C3C', s=12, alpha=0.6, label='Error', linewidths=0)
    ax.scatter(df['lon'].values[correct], df['lat'].values[correct],
               c='#2ECC71', s=12, alpha=0.6, label='Correct', linewidths=0)
    ax.set_title(f'Prediction Errors — {county_name}\n'
                 f'Acc={acc:.1%}  ({correct.sum()}/{len(correct)} correct)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    ax.set_aspect('equal', adjustable='datalim')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.2)
    plt.tight_layout()
    err_path = os.path.join(ROOT_OUT, f'phase4_{slug}_error_map.png')
    plt.savefig(err_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {err_path}")

print("\nDone.")
