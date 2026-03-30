# local_model_testing/train_local_models.py
# Train multiple sklearn models on raw_full.csv (7 classes, 42 v4 features).
# Uses class_weight='balanced' throughout to handle the Other class imbalance.
#
# Models:
#   RF_200    RandomForest 200 trees
#   RF_500    RandomForest 500 trees
#   GBT       GradientBoosting 200 trees
#   SVM       SVM RBF kernel (scaled)
#   LR        Logistic Regression baseline
#   XGB       XGBoost 300 trees
#   MLP_S     TensorFlow MLP: small  (256-128-64)
#   MLP_L     TensorFlow MLP: large  (512-256-128-64) + dropout + batch norm
#
# Outputs -> outputs/
#   local_metrics.json
#   local_cm_{model}.png
#   local_importance_{model}.png    (RF + GBT only)
#   local_{model}_preds.csv
#
# Run: ..\venv\Scripts\python train_local_models.py

import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, cohen_kappa_score,
                              classification_report, confusion_matrix, f1_score)

DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')
OUT_DIR    = os.path.join(os.path.dirname(__file__), 'outputs')
os.makedirs(OUT_DIR, exist_ok=True)

CLASS_MAP = {1:'Corn', 5:'Soybeans', 24:'Winter_Wheat',
             28:'Oats', 36:'Alfalfa', 26:'Dbl_Crop', 0:'Other'}
CLASS_CODES  = [1, 5, 24, 28, 36, 26, 0]
CLASS_NAMES  = [CLASS_MAP[c] for c in CLASS_CODES]

# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading raw_full.csv...")
df = pd.read_csv(os.path.join(DATA_DIR, 'raw_full.csv'))
print(f"  {len(df)} rows | {df['cdl_code'].nunique()} classes")
print(df['crop_name'].value_counts().to_string())

meta_cols    = ['cdl_code','crop_name','dataset','region','label_year',
                'image_year','split','longitude','latitude','cdl_group']
feature_cols = [c for c in df.columns if c not in meta_cols]
print(f"\n  {len(feature_cols)} feature columns")

df_train = df[df['split'] == 'train'].dropna(subset=feature_cols)
df_test  = df[df['split'] == 'test'].dropna(subset=feature_cols)

X_train = df_train[feature_cols].values
y_train = df_train['cdl_code'].values.astype(int)
X_test  = df_test[feature_cols].values
y_test  = df_test['cdl_code'].values.astype(int)

print(f"\n  Train: {len(X_train)}  Test: {len(X_test)}")

# ── TensorFlow MLP wrapper (sklearn-compatible interface) ─────────────────────
class TFMLPClassifier:
    """
    Keras MLP wrapped to match sklearn's fit/predict interface.
    Architecture: Dense -> BatchNorm -> Dropout (repeated), then softmax out.
    Uses class weights to handle imbalance, early stopping to avoid overfit.
    """
    def __init__(self, hidden_layers, dropout=0.3, epochs=100,
                 batch_size=64, name='MLP'):
        self.hidden_layers  = hidden_layers
        self.dropout        = dropout
        self.epochs         = epochs
        self.batch_size     = batch_size
        self.name           = name
        self.model_         = None
        self.classes_       = None
        self.scaler_        = StandardScaler()

    def fit(self, X, y):
        import tensorflow as tf
        from tensorflow import keras

        # Scale
        X_sc = self.scaler_.fit_transform(X)

        # Label encode to 0-indexed
        self.classes_ = np.sort(np.unique(y))
        self._code_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([self._code_to_idx[c] for c in y])
        n_classes = len(self.classes_)

        # Class weights
        from sklearn.utils.class_weight import compute_class_weight
        weights = compute_class_weight('balanced', classes=self.classes_, y=y)
        cw = {self._code_to_idx[c]: w for c, w in zip(self.classes_, weights)}

        # Build model
        inp = keras.Input(shape=(X_sc.shape[1],))
        x   = inp
        for units in self.hidden_layers:
            x = keras.layers.Dense(units, activation='relu',
                                   kernel_regularizer=keras.regularizers.l2(1e-4))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Dropout(self.dropout)(x)
        out = keras.layers.Dense(n_classes, activation='softmax')(x)

        self.model_ = keras.Model(inp, out)
        self.model_.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=7, min_lr=1e-5),
        ]

        self.model_.fit(
            X_sc, y_idx,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.15,
            class_weight=cw,
            callbacks=callbacks,
            verbose=0
        )
        return self

    def predict(self, X):
        X_sc    = self.scaler_.transform(X)
        probs   = self.model_.predict(X_sc, verbose=0)
        idx     = np.argmax(probs, axis=1)
        return np.array([self.classes_[i] for i in idx])

    def predict_proba(self, X):
        X_sc = self.scaler_.transform(X)
        return self.model_.predict(X_sc, verbose=0)

    @property
    def feature_importances_(self):
        return None   # not applicable for NNs


# ── Model zoo ─────────────────────────────────────────────────────────────────
def make_models():
    models = {
        'RF_200': RandomForestClassifier(
            n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42),
        'RF_500': RandomForestClassifier(
            n_estimators=500, class_weight='balanced', n_jobs=-1, random_state=42),
        'GBT': GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5,
            subsample=0.8, random_state=42),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(kernel='rbf', C=10, gamma='scale',
                        class_weight='balanced', random_state=42))]),
        'LR': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(
                max_iter=2000, class_weight='balanced', random_state=42))]),
    }
    # XGBoost
    try:
        from xgboost import XGBClassifier
        models['XGB'] = XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            n_jobs=-1, random_state=42, verbosity=0)
        print("  XGBoost included")
    except ImportError:
        print("  XGBoost not installed")

    # TensorFlow MLPs — wrapped in a sklearn-compatible class
    try:
        models['MLP_S'] = TFMLPClassifier(hidden_layers=[256, 128, 64],
                                           dropout=0.3, epochs=100, batch_size=64,
                                           name='MLP_S')
        models['MLP_L'] = TFMLPClassifier(hidden_layers=[512, 256, 128, 64],
                                           dropout=0.4, epochs=150, batch_size=64,
                                           name='MLP_L')
        print("  TensorFlow MLP included")
    except Exception as e:
        print(f"  TensorFlow not available: {e}")

    return models

# ── Helpers ───────────────────────────────────────────────────────────────────
def feat_color(name):
    if name.startswith('shape_'):   return '#E8A020'
    if name in ('NDVI_max','NDVI_amp','NDVI_mean','NDVI_std'): return '#06D6A0'
    if any(name.endswith(s) for s in ('_c','_cos1','_sin1','_cos2','_sin2',
                                       '_amp1','_amp2','_phase1')): return '#9B59B6'
    return '#2D6A4F'

def plot_cm(y_true, y_pred, title, path):
    present = sorted(set(y_true) | set(y_pred))
    names   = [CLASS_MAP.get(c, str(c)).replace('_','\n') for c in present]
    cm      = confusion_matrix(y_true, y_pred, labels=present)
    cm_n    = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=11, fontweight='bold')
    sns.heatmap(cm,   annot=True, fmt='d',    cmap='Blues', ax=axes[0],
                xticklabels=names, yticklabels=names, cbar=False, linewidths=0.4)
    sns.heatmap(cm_n, annot=True, fmt='.2f',  cmap='Blues', ax=axes[1],
                xticklabels=names, yticklabels=names, linewidths=0.4)
    for ax, t in zip(axes, ['Counts','Normalised']):
        ax.set_title(t); ax.set_ylabel('True'); ax.set_xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

def plot_importance(importances, names, title, path):
    df_imp = pd.DataFrame({'feature': names, 'importance': importances})
    df_imp = df_imp.sort_values('importance', ascending=True).tail(25)
    colors = [feat_color(n) for n in df_imp['feature']]
    fig, ax = plt.subplots(figsize=(8, max(5, len(df_imp)*0.32)))
    ax.barh(df_imp['feature'], df_imp['importance'], color=colors, edgecolor='white')
    ax.set_xlabel('Importance', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#2D6A4F', label='Monthly snapshot'),
        Patch(facecolor='#E8A020', label='Seasonal shape'),
        Patch(facecolor='#9B59B6', label='Harmonic'),
        Patch(facecolor='#06D6A0', label='Robust stats'),
    ], fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight'); plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN
# ══════════════════════════════════════════════════════════════════════════════
all_metrics = {}
all_models = make_models()
# Resume from XGB (RF/GBT/SVM/LR already completed)
skip_done = {'RF_200', 'RF_500', 'GBT', 'SVM', 'LR', 'XGB', 'MLP_S'}
models = {k: v for k, v in all_models.items() if k not in skip_done}

# Load previous results
metrics_path = os.path.join(OUT_DIR, 'local_metrics.json')
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        import json; all_metrics = json.load(f)


for model_name, model in models.items():
    print(f"\n{'='*55}")
    print(f"  {model_name}")
    print(f"{'='*55}")

    if model_name == 'XGB':
        # XGB needs 0-indexed integer labels
        code_to_idx = {c: i for i, c in enumerate(sorted(set(y_train)))}
        idx_to_code = {i: c for c, i in code_to_idx.items()}
        model.fit(X_train, np.array([code_to_idx[c] for c in y_train]))
        preds = np.array([idx_to_code[i] for i in model.predict(X_test)])
    elif isinstance(model, TFMLPClassifier):
        print(f"  Training TF model ({model.hidden_layers}, dropout={model.dropout})...")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

    acc   = accuracy_score(y_test, preds)
    kappa = cohen_kappa_score(y_test, preds)
    f1    = f1_score(y_test, preds, average='weighted', zero_division=0)
    f1_mac= f1_score(y_test, preds, average='macro',    zero_division=0)

    print(f"  Accuracy : {acc:.1%}")
    print(f"  Kappa    : {kappa:.3f}")
    print(f"  F1 (wtd) : {f1:.3f}    F1 (macro): {f1_mac:.3f}")

    all_metrics[model_name] = {
        'accuracy': acc, 'kappa': kappa,
        'f1_weighted': f1, 'f1_macro': f1_mac,
        'n_train': len(X_train), 'n_test': len(X_test),
        'n_features': len(feature_cols),
    }

    # Confusion matrix
    plot_cm(y_test, preds,
            f'{model_name} — 7 classes | Acc={acc:.1%} Kappa={kappa:.3f}',
            os.path.join(OUT_DIR, f'local_cm_{model_name}.png'))

    # Feature importance (tree models only)
    raw_model = model.named_steps.get('svm') or model.named_steps.get('lr') \
                if hasattr(model, 'named_steps') else model
    imp = getattr(raw_model, 'feature_importances_', None)
    if imp is not None:
        plot_importance(imp, feature_cols,
                        f'Feature Importance — {model_name}',
                        os.path.join(OUT_DIR, f'local_importance_{model_name}.png'))

    # Save predictions
    pd.DataFrame({'true_code': y_test, 'pred_code': preds}).to_csv(
        os.path.join(OUT_DIR, f'local_{model_name}_preds.csv'), index=False)

    # Per-class report
    present = sorted(set(y_test))
    names   = [CLASS_MAP.get(c, str(c)) for c in present]
    print(f"\n{classification_report(y_test, preds, labels=present, target_names=names, zero_division=0)}")

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics_path = os.path.join(OUT_DIR, 'local_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(all_metrics, f, indent=2)
print(f"\nSaved: {metrics_path}")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("FINAL SUMMARY")
print(f"{'='*65}")
print(f"{'Model':<10} {'Accuracy':>10} {'Kappa':>8} {'F1-wtd':>8} {'F1-mac':>8}")
print("-" * 50)
for mn, m in sorted(all_metrics.items(), key=lambda x: -x[1]['kappa']):
    print(f"{mn:<10} {m['accuracy']:>9.1%} {m['kappa']:>8.3f} "
          f"{m['f1_weighted']:>8.3f} {m['f1_macro']:>8.3f}")

print(f"\nAll outputs saved to: {OUT_DIR}")
print("Run compare_to_gee.py to compare against Phase 3 GEE RF.")
