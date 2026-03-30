# local_model_testing/retrain_boosted.py
# Train XGB_weighted + RF_500 on raw_full_boosted.csv only.
# Compares recall vs the previously saved metrics from raw_full.csv.
#
# Outputs -> outputs/
#   boosted_cm_XGB_weighted.png
#   boosted_cm_RF_500.png
#   boosted_recall_comparison.png   vs old metrics from local_metrics.json
#   boosted_metrics.json
#
# Run:
#   cd D:\remote_project_2
#   venv\Scripts\python local_model_testing/retrain_boosted.py

import os, sys, json
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _ROOT)
sys.path.insert(0, _HERE)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              classification_report, confusion_matrix)
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

DATA_DIR = os.path.join(_HERE, 'data')
OUT_DIR  = os.path.join(_HERE, 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_MAP   = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
               28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}
CLASS_CODES = [1, 5, 24, 28, 36, 26, 0]
CLASS_NAMES = [CLASS_MAP[c] for c in CLASS_CODES]

META_COLS = ['cdl_code','crop_name','dataset','region','label_year',
             'image_year','split','longitude','latitude','cdl_group']

code_to_idx = {c: i for i, c in enumerate(CLASS_CODES)}
idx_to_code = {i: c for c, i in code_to_idx.items()}

# ── Load boosted dataset ──────────────────────────────────────────────────────
print("Loading raw_full_boosted.csv...")
df   = pd.read_csv(os.path.join(DATA_DIR, 'raw_full_boosted.csv'))
feat = [c for c in df.columns if c not in META_COLS]

train = df[df['split'] == 'train']
test  = df[df['split'] == 'test']

X_tr = train[feat].values;  y_tr = train['cdl_code'].values
X_te = test[feat].values;   y_te = test['cdl_code'].values

print(f"  {len(train)} train / {len(test)} test | {len(feat)} features")
print("  Class distribution (train):")
print(train['crop_name'].value_counts().to_string())

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining XGB_weighted (boosted)...")
sw    = compute_sample_weight('balanced', y=y_tr)
y_idx = np.array([code_to_idx[c] for c in y_tr])
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, num_class=len(CLASS_CODES),
)
xgb_model.fit(X_tr, y_idx, sample_weight=sw)

print("Training RF_500 (boosted)...")
rf_model = RandomForestClassifier(
    n_estimators=500, class_weight='balanced',
    random_state=42, n_jobs=-1,
)
rf_model.fit(X_tr, y_tr)


# ── Evaluate ─────────────────────────────────────────────────────────────────
def evaluate(name, model, is_xgb=False):
    if is_xgb:
        preds_idx = model.predict(X_te)
        y_pred    = np.array([idx_to_code[i] for i in preds_idx])
    else:
        y_pred = model.predict(X_te)

    acc   = accuracy_score(y_te, y_pred)
    kappa = cohen_kappa_score(y_te, y_pred)
    rep   = classification_report(y_te, y_pred,
                                   labels=CLASS_CODES, target_names=CLASS_NAMES,
                                   output_dict=True, zero_division=0)
    print(f"\n  [{name}]  Acc={acc:.4f}  kappa={kappa:.4f}")
    for cn in CLASS_NAMES:
        r = rep[cn]['recall']
        f = rep[cn]['f1-score']
        marker = ' <--' if cn in ('Alfalfa','Oats') else ''
        print(f"    {cn:<15} recall={r:.3f}  f1={f:.3f}{marker}")
    return acc, kappa, rep, y_pred


results = {}
preds   = {}

acc, kappa, rep, yp = evaluate('XGB_weighted_boosted', xgb_model, is_xgb=True)
results['XGB_weighted_boosted'] = {'acc': acc, 'kappa': kappa, 'report': rep}
preds['XGB_weighted_boosted']   = yp

acc, kappa, rep, yp = evaluate('RF_500_boosted', rf_model, is_xgb=False)
results['RF_500_boosted'] = {'acc': acc, 'kappa': kappa, 'report': rep}
preds['RF_500_boosted']   = yp


# ── Confusion matrices ────────────────────────────────────────────────────────
def plot_cm(y_true, y_pred, title, fname):
    cm      = confusion_matrix(y_true, y_pred, labels=CLASS_CODES)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                ax=ax, vmin=0, vmax=1)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, fname)
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  Saved: {p}")

for tag, yp in preds.items():
    r = results[tag]
    plot_cm(y_te, yp,
            title=f"{tag}  (acc={r['acc']:.3f}, kappa={r['kappa']:.3f})",
            fname=f"boosted_cm_{tag}.png")


# ── Recall comparison vs old metrics ─────────────────────────────────────────
old_json = os.path.join(OUT_DIR, 'local_metrics.json')
if os.path.exists(old_json):
    with open(old_json) as f:
        old_metrics = json.load(f)

    print("\nGenerating recall comparison (boosted vs original)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    x     = np.arange(len(CLASS_NAMES))
    width = 0.35

    pairs = [
        ('XGB_weighted', 'XGB_weighted_boosted', axes[0]),
        ('RF_500',        'RF_500_boosted',        axes[1]),
    ]

    for old_key, new_key, ax in pairs:
        if old_key not in old_metrics or 'report' not in old_metrics[old_key]:
            ax.set_title(f'{old_key}: no old data (missing report)'); continue

        old_rec  = [old_metrics[old_key]['report'].get(n, {}).get('recall', 0)
                    for n in CLASS_NAMES]
        new_rec  = [results[new_key]['report'][n]['recall']
                    for n in CLASS_NAMES]

        ax.bar(x - width/2, old_rec, width, label='Original (2023 only)',
               color='steelblue', alpha=0.8)
        ax.bar(x + width/2, new_rec, width, label='Boosted (+2022)',
               color='darkorange', alpha=0.8)

        ax.set_xticks(x); ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
        ax.set_ylim(0, 1.1); ax.set_ylabel('Recall')
        ax.set_title(f'{old_key}')
        ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, label='0.70 target')
        ax.legend()

        for i, cn in enumerate(CLASS_NAMES):
            if cn in ('Alfalfa', 'Oats'):
                delta = new_rec[i] - old_rec[i]
                col   = 'green' if delta >= 0 else 'red'
                ax.text(i + width/2, new_rec[i] + 0.03,
                        f'{delta:+.2f}', ha='center', fontsize=9,
                        color=col, fontweight='bold')

    plt.suptitle('Recall Improvement from 2022 Boost Samples\n(Alfalfa & Oats annotated)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    p = os.path.join(OUT_DIR, 'boosted_recall_comparison.png')
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {p}")
else:
    print(f"\n  (no {old_json} found — skipping comparison chart)")


# ── Save metrics ──────────────────────────────────────────────────────────────
def clean(obj):
    if isinstance(obj, (np.floating, float)): return round(float(obj), 4)
    if isinstance(obj, dict):                 return {k: clean(v) for k, v in obj.items()}
    return obj

out_json = os.path.join(OUT_DIR, 'boosted_metrics.json')
with open(out_json, 'w') as f:
    json.dump(clean(results), f, indent=2)
print(f"\nMetrics saved: {out_json}")

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("FINAL SUMMARY (boosted dataset)")
print(f"{'='*60}")
for tag, res in results.items():
    print(f"  {tag:<35} acc={res['acc']:.4f}  kappa={res['kappa']:.4f}")

print("\nAlfalfa & Oats recall (boosted):")
for tag in results:
    for cls in ['Alfalfa', 'Oats']:
        r = results[tag]['report'][cls]['recall']
        f = results[tag]['report'][cls]['f1-score']
        print(f"  {tag:<35} {cls:<10} recall={r:.3f}  f1={f:.3f}")
