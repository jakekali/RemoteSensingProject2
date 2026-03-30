# local_model_testing/tune_xgb.py
# Improve XGB with:
#   1. Class weights (sample_weight in fit) — fixes Other dominance
#   2. Optuna hyperparameter search (50 trials, 3-fold CV)
#
# Compares: XGB_base -> XGB_weighted -> XGB_tuned
#
# Run: ..\venv\Scripts\python tune_xgb.py

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              classification_report, confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR  = os.path.join(os.path.dirname(__file__), 'outputs')

CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv(os.path.join(DATA_DIR, 'raw_full.csv'))
meta = ['cdl_code','crop_name','dataset','region','label_year',
        'image_year','split','longitude','latitude','cdl_group']
feat_cols = [c for c in df.columns if c not in meta]

df_train = df[df['split']=='train'].dropna(subset=feat_cols)
df_test  = df[df['split']=='test'].dropna(subset=feat_cols)

X_train = df_train[feat_cols].values
y_train = df_train['cdl_code'].values.astype(int)
X_test  = df_test[feat_cols].values
y_test  = df_test['cdl_code'].values.astype(int)

# XGB needs 0-indexed labels
classes     = np.sort(np.unique(y_train))
code_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_code = {i: c for c, i in code_to_idx.items()}
y_train_idx = np.array([code_to_idx[c] for c in y_train])
y_test_idx  = np.array([code_to_idx[c] for c in y_test])

# Class weights for sample_weight
sample_weights = compute_sample_weight('balanced', y=y_train)

print(f"  Train: {len(X_train)}  Test: {len(X_test)}  Features: {len(feat_cols)}")
print(f"  Classes: {[CLASS_MAP[c] for c in classes]}")
print(f"  Sample weight range: {sample_weights.min():.3f} – {sample_weights.max():.3f}")

results = {}

# ══════════════════════════════════════════════════════════════════════════════
# BASELINE XGB (no weights, default params) — for fair comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("XGB_base (no weights, default params)")
print("="*55)
xgb_base = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                          subsample=0.8, colsample_bytree=0.8,
                          eval_metric='mlogloss', n_jobs=-1,
                          random_state=42, verbosity=0)
xgb_base.fit(X_train, y_train_idx)
preds_base = np.array([idx_to_code[i] for i in xgb_base.predict(X_test)])
acc  = accuracy_score(y_test, preds_base)
kap  = cohen_kappa_score(y_test, preds_base)
f1m  = f1_score(y_test, preds_base, average='macro', zero_division=0)
print(f"  Accuracy={acc:.1%}  Kappa={kap:.3f}  F1-macro={f1m:.3f}")
results['XGB_base'] = {'accuracy': acc, 'kappa': kap, 'f1_macro': f1m,
                        'preds': preds_base}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: CLASS WEIGHTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("XGB_weighted (balanced sample weights, same params)")
print("="*55)
xgb_w = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8,
                       eval_metric='mlogloss', n_jobs=-1,
                       random_state=42, verbosity=0)
xgb_w.fit(X_train, y_train_idx, sample_weight=sample_weights)
preds_w = np.array([idx_to_code[i] for i in xgb_w.predict(X_test)])
acc  = accuracy_score(y_test, preds_w)
kap  = cohen_kappa_score(y_test, preds_w)
f1m  = f1_score(y_test, preds_w, average='macro', zero_division=0)
print(f"  Accuracy={acc:.1%}  Kappa={kap:.3f}  F1-macro={f1m:.3f}")
print(f"\n  Per-class recall comparison (base -> weighted):")
present = sorted(set(y_test))
cm_b = confusion_matrix(y_test, preds_base, labels=present)
cm_w = confusion_matrix(y_test, preds_w, labels=present)
rec_b = cm_b.diagonal() / cm_b.sum(axis=1)
rec_w = cm_w.diagonal() / cm_w.sum(axis=1)
for i, code in enumerate(present):
    name = CLASS_MAP.get(code, str(code))
    delta = rec_w[i] - rec_b[i]
    sign  = '+' if delta >= 0 else ''
    print(f"    {name:<15} {rec_b[i]:.1%} -> {rec_w[i]:.1%}  ({sign}{delta:.1%})")
results['XGB_weighted'] = {'accuracy': acc, 'kappa': kap, 'f1_macro': f1m,
                            'preds': preds_w}

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: OPTUNA HYPERPARAMETER SEARCH
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*55)
print("XGB_tuned (Optuna, 60 trials, 3-fold CV, with class weights)")
print("="*55)

N_TRIALS = 60
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

def objective(trial):
    params = {
        'n_estimators':      trial.suggest_int('n_estimators', 100, 600),
        'max_depth':         trial.suggest_int('max_depth', 3, 10),
        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.4, 1.0),
        'min_child_weight':  trial.suggest_int('min_child_weight', 1, 10),
        'gamma':             trial.suggest_float('gamma', 0.0, 5.0),
        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
        'eval_metric': 'mlogloss', 'n_jobs': -1,
        'random_state': 42, 'verbosity': 0,
    }
    model = XGBClassifier(**params)
    # CV with sample weights — macro F1 as objective (cares about minority classes)
    scores = []
    for train_idx, val_idx in cv.split(X_train, y_train_idx):
        Xtr, Xval = X_train[train_idx], X_train[val_idx]
        ytr, yval = y_train_idx[train_idx], y_train_idx[val_idx]
        sw_tr     = sample_weights[train_idx]
        model.fit(Xtr, ytr, sample_weight=sw_tr,
                  eval_set=[(Xval, yval)], verbose=False)
        preds_val = model.predict(Xval)
        preds_orig = np.array([idx_to_code[i] for i in preds_val])
        yval_orig  = np.array([idx_to_code[i] for i in yval])
        scores.append(f1_score(yval_orig, preds_orig, average='macro', zero_division=0))
    return np.mean(scores)

study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))

# Progress callback
def progress_cb(study, trial):
    if trial.number % 10 == 0:
        print(f"  Trial {trial.number:3d}/60  best F1-macro={study.best_value:.4f}  "
              f"(this={trial.value:.4f})", flush=True)

print(f"  Running {N_TRIALS} trials (optimising macro F1 with class weights)...")
study.optimize(objective, n_trials=N_TRIALS, callbacks=[progress_cb])

best = study.best_params
print(f"\n  Best params:")
for k, v in best.items():
    print(f"    {k:<22} = {v}")

# Final model with best params
xgb_tuned = XGBClassifier(**best, eval_metric='mlogloss',
                            n_jobs=-1, random_state=42, verbosity=0)
xgb_tuned.fit(X_train, y_train_idx, sample_weight=sample_weights)
preds_t = np.array([idx_to_code[i] for i in xgb_tuned.predict(X_test)])
acc  = accuracy_score(y_test, preds_t)
kap  = cohen_kappa_score(y_test, preds_t)
f1m  = f1_score(y_test, preds_t, average='macro', zero_division=0)
f1w  = f1_score(y_test, preds_t, average='weighted', zero_division=0)
print(f"\n  Accuracy={acc:.1%}  Kappa={kap:.3f}  F1-macro={f1m:.3f}  F1-wtd={f1w:.3f}")
results['XGB_tuned'] = {'accuracy': acc, 'kappa': kap, 'f1_macro': f1m,
                         'f1_weighted': f1w, 'best_params': best, 'preds': preds_t}

print(f"\n  Per-class report:")
names_present = [CLASS_MAP.get(c,'?') for c in present]
print(classification_report(y_test, preds_t, labels=present,
                             target_names=names_present, zero_division=0))

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# 1. Confusion matrix for tuned model
short = [CLASS_MAP.get(c,'?').replace('_','\n') for c in present]
cm_t  = confusion_matrix(y_test, preds_t, labels=present)
cm_tn = cm_t.astype(float) / (cm_t.sum(axis=1, keepdims=True) + 1e-9)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle(f'XGB_tuned — Acc={acc:.1%}  Kappa={kap:.3f}  F1-macro={f1m:.3f}',
             fontsize=12, fontweight='bold')
sns.heatmap(cm_t,  annot=True, fmt='d',   cmap='Blues', ax=axes[0],
            xticklabels=short, yticklabels=short, cbar=False, linewidths=0.4)
sns.heatmap(cm_tn, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            xticklabels=short, yticklabels=short, linewidths=0.4)
for ax, t in zip(axes,['Counts','Normalised']):
    ax.set_title(t); ax.set_ylabel('True'); ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'xgb_tuned_cm.png'), dpi=150, bbox_inches='tight')
plt.close()

# 2. Recall comparison: base vs weighted vs tuned
fig, ax = plt.subplots(figsize=(10, 5))
x     = np.arange(len(present))
w     = 0.25
cm_t2 = confusion_matrix(y_test, preds_t, labels=present)
rec_t = cm_t2.diagonal() / cm_t2.sum(axis=1)
names_short = [CLASS_MAP.get(c,'?') for c in present]

bars_b = ax.bar(x - w, rec_b, w, label='XGB_base',     color='#CCCCCC', edgecolor='white')
bars_w = ax.bar(x,     rec_w, w, label='XGB_weighted', color='#2D6A4F', edgecolor='white')
bars_t = ax.bar(x + w, rec_t, w, label='XGB_tuned',    color='#E8A020', edgecolor='white')

ax.set_xticks(x); ax.set_xticklabels(names_short, rotation=15)
ax.set_ylabel('Recall (per-class)', fontsize=11)
ax.set_title('Per-class Recall: Base vs Weighted vs Tuned', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.axhline(0.7, color='#ccc', linestyle='--', linewidth=0.8)
ax.legend(fontsize=10)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'xgb_recall_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# 3. Optuna optimization history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))
trials_df = study.trials_dataframe()
ax1.plot(trials_df['number'], trials_df['value'], alpha=0.4, color='#2D6A4F', linewidth=0.8)
ax1.plot(trials_df['number'],
         trials_df['value'].cummax(), color='#E8A020', linewidth=2, label='Best so far')
ax1.set_xlabel('Trial'); ax1.set_ylabel('CV F1-macro')
ax1.set_title('Optuna Optimization History'); ax1.legend()
ax1.spines[['top','right']].set_visible(False)

# Param importances
try:
    imp = optuna.importance.get_param_importances(study)
    names_i = list(imp.keys())[:8]; vals_i = [imp[n] for n in names_i]
    ax2.barh(names_i[::-1], vals_i[::-1], color='#9B59B6', edgecolor='white')
    ax2.set_xlabel('Importance'); ax2.set_title('Hyperparameter Importance')
    ax2.spines[['top','right']].set_visible(False)
except Exception:
    ax2.text(0.5, 0.5, 'Importance\nnot available', ha='center', va='center',
             transform=ax2.transAxes)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,'xgb_optuna.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Save updated metrics ───────────────────────────────────────────────────────
metrics_path = os.path.join(OUT_DIR, 'local_metrics.json')
with open(metrics_path) as f:
    all_metrics = json.load(f)

for name in ('XGB_base', 'XGB_weighted', 'XGB_tuned'):
    r = results[name]
    all_metrics[name] = {
        'accuracy': r['accuracy'], 'kappa': r['kappa'],
        'f1_macro': r['f1_macro'],
        'f1_weighted': r.get('f1_weighted', None),
        'n_train': len(X_train), 'n_test': len(X_test),
        'n_features': len(feat_cols),
        'best_params': r.get('best_params'),
    }
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)

# ── Final summary ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("SUMMARY")
print("="*55)
print(f"{'Model':<16} {'Accuracy':>10} {'Kappa':>8} {'F1-macro':>10}")
print("-"*46)
for name in ('XGB_base','XGB_weighted','XGB_tuned'):
    r = results[name]
    print(f"{name:<16} {r['accuracy']:>9.1%} {r['kappa']:>8.3f} {r['f1_macro']:>10.3f}")

print(f"\nOutputs saved to: {OUT_DIR}")
print("  xgb_tuned_cm.png         confusion matrix")
print("  xgb_recall_comparison.png per-class recall bar chart")
print("  xgb_optuna.png            optimization history + param importance")
