# local_model_testing/evaluate_2024.py
# Apply all trained local models to 2024 county test data.
# Measures generalization: same model, new year, new location.
#
# Inputs:
#   data/test_2024_mclean.csv      McLean IL (in-distribution)
#   data/test_2024_renville.csv    Renville MN (out-of-distribution)
#   outputs/local_metrics.json     trained model params
#
# Models applied:
#   XGB_tuned  (best overall)
#   XGB_base   (baseline comparison)
#   RF_200     (forest comparison)
#   GBT        (gradient boost comparison)
#
# Outputs -> outputs/
#   gen_metrics.json               accuracy/kappa per model x county
#   gen_cm_{county}_{model}.png    confusion matrices
#   gen_recall_heatmap.png         per-class recall: train vs mclean vs renville
#   gen_report.txt                 full text generalization report
#
# Run: ..\venv\Scripts\python evaluate_2024.py

import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              classification_report, confusion_matrix, f1_score)
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs')

CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}

COUNTIES = {
    'mclean':   'McLean County, IL  (in-distribution)',
    'renville': 'Renville County, MN  (out-of-distribution)',
}

# ── Load training data & retrain models ──────────────────────────────────────
print("Loading training data...")
df_train_full = pd.read_csv(os.path.join(DATA_DIR, 'raw_full_boosted.csv'))
meta = ['cdl_code','crop_name','dataset','region','label_year',
        'image_year','split','longitude','latitude','cdl_group']
feat_cols = [c for c in df_train_full.columns if c not in meta]

df_tr = df_train_full[df_train_full['split']=='train'].dropna(subset=feat_cols)
X_tr  = df_tr[feat_cols].values
y_tr  = df_tr['cdl_code'].values.astype(int)

# XGB label encoding
classes     = np.sort(np.unique(y_tr))
code_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_code = {i: c for c, i in code_to_idx.items()}
y_tr_idx    = np.array([code_to_idx[c] for c in y_tr])
sw_tr       = compute_sample_weight('balanced', y=y_tr)

# Load best XGB params from Optuna if available
metrics_path = os.path.join(OUT_DIR, 'local_metrics.json')
with open(metrics_path) as f:
    saved_metrics = json.load(f)

tuned_params = saved_metrics.get('XGB_tuned', {}).get('best_params', None)

# ── Build & train models ──────────────────────────────────────────────────────
print("Training models on 2023 Iowa+IL data...")

MODELS = {}

# XGB base
print("  XGB_base...")
xgb_base = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='mlogloss', n_jobs=-1,
                          random_state=42, verbosity=0)
xgb_base.fit(X_tr, y_tr_idx)
MODELS['XGB_base'] = ('xgb', xgb_base)

# XGB weighted
print("  XGB_weighted...")
xgb_w = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8,
                       eval_metric='mlogloss', n_jobs=-1,
                       random_state=42, verbosity=0)
xgb_w.fit(X_tr, y_tr_idx, sample_weight=sw_tr)
MODELS['XGB_weighted'] = ('xgb', xgb_w)

# XGB tuned
if tuned_params:
    print("  XGB_tuned (Optuna params)...")
    xgb_t = XGBClassifier(**tuned_params, eval_metric='mlogloss',
                           n_jobs=-1, random_state=42, verbosity=0)
    xgb_t.fit(X_tr, y_tr_idx, sample_weight=sw_tr)
    MODELS['XGB_tuned'] = ('xgb', xgb_t)
else:
    print("  XGB_tuned: Optuna params not found yet — skipping")

# RF_200
print("  RF_200...")
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                             n_jobs=-1, random_state=42)
rf.fit(X_tr, y_tr)
MODELS['RF_200'] = ('sklearn', rf)

# GBT
print("  GBT...")
gbt = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                  max_depth=5, subsample=0.8, random_state=42)
gbt.fit(X_tr, y_tr)
MODELS['GBT'] = ('sklearn', gbt)

print(f"  {len(MODELS)} models ready")

# ══════════════════════════════════════════════════════════════════════════════
# EVALUATE ON EACH COUNTY
# ══════════════════════════════════════════════════════════════════════════════
gen_metrics = {}
all_preds   = {}   # [county][model] -> (y_true, y_pred)

for slug, county_name in COUNTIES.items():
    csv_path = os.path.join(DATA_DIR, f'test_2024_{slug}.csv')
    if not os.path.exists(csv_path):
        print(f"\nSKIP {slug}: {csv_path} not found")
        print("  Download from Drive first.")
        continue

    print(f"\n{'='*55}")
    print(f"COUNTY: {county_name}")
    print(f"{'='*55}")

    df_test = pd.read_csv(csv_path)

    # Parse .geo for lat/lon if present
    if '.geo' in df_test.columns:
        def _parse(g):
            try:
                import json as _j
                c = _j.loads(str(g))['coordinates']
                return round(float(c[0]),6), round(float(c[1]),6)
            except: return None, None
        df_test[['longitude','latitude']] = df_test['.geo'].apply(
            lambda g: pd.Series(_parse(g)))
        df_test = df_test.drop(columns=['.geo'])

    # Keep only feature cols present in training
    test_feat_cols = [c for c in feat_cols if c in df_test.columns]
    missing = [c for c in feat_cols if c not in df_test.columns]
    if missing:
        print(f"  WARNING: {len(missing)} features missing from test data")

    df_test = df_test.dropna(subset=test_feat_cols)
    # Replace fill sentinel with NaN — XGBoost handles NaN natively via learned
    # default directions; sklearn models get -1 (out-of-range, tree-splittable)
    df_test[test_feat_cols] = df_test[test_feat_cols].replace(-9999, float('nan'))

    X_te   = df_test[test_feat_cols].values
    y_te   = df_test['cdl_code'].values.astype(int)
    present = sorted(set(y_te))
    names   = [CLASS_MAP.get(c,'?') for c in present]

    print(f"  {len(df_test)} samples after cleaning")
    print(f"  Classes present: {names}")
    print(f"  Distribution: {dict(zip(names, [int((y_te==c).sum()) for c in present]))}")

    gen_metrics[slug] = {}
    all_preds[slug]   = {}

    X_te_sklearn = np.where(np.isnan(X_te), -1.0, X_te)  # -1 for RF/GBT (out-of-range, splittable)

    for model_name, (mtype, model) in MODELS.items():
        if mtype == 'xgb':
            preds_idx = model.predict(X_te)  # XGB handles NaN natively
            preds = np.array([idx_to_code.get(i, -1) for i in preds_idx])
        else:
            preds = model.predict(X_te_sklearn)

        # Filter to classes present in test set
        valid = np.isin(y_te, present) & np.isin(preds, present + [0,1,5,24,26,28,36])
        y_v, p_v = y_te[valid], preds[valid]

        acc  = accuracy_score(y_v, p_v)
        kap  = cohen_kappa_score(y_v, p_v)
        f1m  = f1_score(y_v, p_v, average='macro',    zero_division=0, labels=present)
        f1w  = f1_score(y_v, p_v, average='weighted', zero_division=0, labels=present)

        gen_metrics[slug][model_name] = {
            'accuracy': acc, 'kappa': kap,
            'f1_macro': f1m, 'f1_weighted': f1w,
            'n_test': int(len(y_v)), 'county': county_name,
            'label_year': 2024, 'image_year': 2024,
        }
        all_preds[slug][model_name] = (y_v, p_v)

        # Generalization gap
        train_acc = saved_metrics.get(model_name, {}).get('accuracy', None)
        gap_str = f"  gap={train_acc - acc:+.1%}" if train_acc else ''
        print(f"  {model_name:<16} Acc={acc:.1%}  K={kap:.3f}  F1-mac={f1m:.3f}{gap_str}")

    # Best model confusion matrix
    best_model = 'XGB_tuned' if 'XGB_tuned' in MODELS else 'XGB_weighted'
    if best_model in all_preds[slug]:
        y_v, p_v = all_preds[slug][best_model]
        cm  = confusion_matrix(y_v, p_v, labels=present)
        cmn = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        short = [n.replace('_','\n') for n in names]
        m     = gen_metrics[slug][best_model]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle(f'{best_model} → {county_name}\n'
                     f'2024 Landsat + 2024 CDL  |  '
                     f'Acc={m["accuracy"]:.1%}  Kappa={m["kappa"]:.3f}',
                     fontsize=11, fontweight='bold')
        sns.heatmap(cm,  annot=True, fmt='d',   cmap='Blues', ax=axes[0],
                    xticklabels=short, yticklabels=short,
                    cbar=False, linewidths=0.4)
        sns.heatmap(cmn, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                    xticklabels=short, yticklabels=short, linewidths=0.4)
        for ax, t in zip(axes,['Counts','Normalised']):
            ax.set_title(t); ax.set_ylabel('True'); ax.set_xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'gen_cm_{slug}_{best_model}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY PLOT: Accuracy across train / McLean / Renville
# ══════════════════════════════════════════════════════════════════════════════
if gen_metrics:
    model_names = list(MODELS.keys())
    county_slugs = list(gen_metrics.keys())
    regions = ['Iowa+IL\n(2023 train)'] + [COUNTIES[s].split('(')[0].strip() for s in county_slugs]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Generalization: 2023 Training → 2024 Counties',
                 fontsize=12, fontweight='bold')

    x      = np.arange(len(model_names))
    n_reg  = len(regions)
    bar_w  = 0.7 / n_reg
    colors = ['#1A1A2E', '#E8A020', '#2D6A4F', '#9B59B6']

    for ri, (region, color) in enumerate(zip(regions, colors)):
        offset = x + (ri - n_reg/2 + 0.5) * bar_w
        if ri == 0:  # training region
            acc_vals = [saved_metrics.get(m, {}).get('accuracy', 0) for m in model_names]
            kap_vals = [saved_metrics.get(m, {}).get('kappa', 0)    for m in model_names]
        else:
            slug = county_slugs[ri - 1]
            acc_vals = [gen_metrics[slug].get(m, {}).get('accuracy', 0) for m in model_names]
            kap_vals = [gen_metrics[slug].get(m, {}).get('kappa', 0)    for m in model_names]

        ax1.bar(offset, acc_vals, bar_w, label=region, color=color, edgecolor='white')
        ax2.bar(offset, kap_vals, bar_w, label=region, color=color, edgecolor='white')

    for ax, ylabel, title in [
        (ax1, 'Overall Accuracy', 'Accuracy'),
        (ax2, "Cohen's Kappa",    'Kappa'),
    ]:
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_','\n') for m in model_names], fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.axhline(0.7, color='#ddd', linestyle='--', linewidth=0.8)
        ax.legend(fontsize=8, loc='lower right')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y', alpha=0.2)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'gen_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: gen_summary.png")

# ── Save metrics ──────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, 'gen_metrics.json'), 'w') as f:
    json.dump(gen_metrics, f, indent=2)

# ── Text report ───────────────────────────────────────────────────────────────
report_path = os.path.join(OUT_DIR, 'gen_report.txt')
with open(report_path, 'w') as f:
    f.write("GENERALIZATION REPORT\n")
    f.write("Model trained on: Iowa + Illinois, 2023 CDL + 2023 Landsat\n")
    f.write("Tested on: 2024 Landsat + 2024 CDL\n")
    f.write("="*70 + "\n\n")

    f.write(f"{'Model':<18} {'Train (2023)':<20}")
    for slug in county_slugs:
        f.write(f"  {slug.capitalize():<20}")
    f.write("\n" + "-"*70 + "\n")

    for mn in model_names:
        tr = saved_metrics.get(mn, {})
        line = f"{mn:<18} Acc={tr.get('accuracy',0):.1%} K={tr.get('kappa',0):.3f}  "
        for slug in county_slugs:
            m = gen_metrics.get(slug, {}).get(mn, {})
            if m:
                gap = tr.get('accuracy', 0) - m['accuracy']
                line += f"  Acc={m['accuracy']:.1%} K={m['kappa']:.3f} gap={gap:+.1%}  "
            else:
                line += "  N/A                  "
        f.write(line + "\n")

    for slug in county_slugs:
        f.write(f"\n{'='*70}\n{COUNTIES[slug]}\n{'='*70}\n")
        best = 'XGB_tuned' if 'XGB_tuned' in gen_metrics.get(slug,{}) else 'XGB_weighted'
        if best in all_preds.get(slug, {}):
            y_v, p_v = all_preds[slug][best]
            present  = sorted(set(y_v))
            names    = [CLASS_MAP.get(c,'?') for c in present]
            f.write(f"\nBest model: {best}\n")
            f.write(classification_report(y_v, p_v, labels=present,
                                          target_names=names, zero_division=0))

print(f"Saved: {report_path}")
print("\n=== GENERALIZATION SUMMARY ===")
print(f"{'Model':<18} {'Train':>8}", end='')
for slug in county_slugs:
    print(f"  {slug.capitalize():>10}", end='')
print()
print("-"*55)
for mn in model_names:
    tr_acc = saved_metrics.get(mn,{}).get('accuracy', 0)
    print(f"{mn:<18} {tr_acc:>7.1%}", end='')
    for slug in county_slugs:
        m = gen_metrics.get(slug,{}).get(mn,{})
        if m:
            gap = tr_acc - m['accuracy']
            print(f"  {m['accuracy']:>7.1%}({gap:+.1%})", end='')
        else:
            print(f"  {'N/A':>10}", end='')
    print()
