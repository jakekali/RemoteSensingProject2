# phase3_analysis.py — Classification: Evaluation & Plots
# Reads the CSVs/JSON saved by phase3_create.py and generates all figures.
# Re-run as many times as needed — touches no GEE, no training.
#
# Inputs  (from outputs/):
#   phase3_metrics.json
#   phase3_v{n}_preds.csv
#   phase3_v{n}_importance.csv
#
# Outputs (to outputs/):
#   phase3_summary_table.png
#   phase3_cm_{v}.png              confusion matrix (counts + normalised)
#   phase3_importance_{v}.png      feature importance bar chart
#   phase3_error_analysis.txt      biggest off-diagonal errors per version
#
# Run with: venv/Scripts/python phase3_analysis.py

from core import *
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

print("=" * 60)
print("PHASE 3 ANALYSIS: EVALUATION & PLOTS")
print("=" * 60)

# ── Load metrics ──────────────────────────────────────────────────────────────
metrics_path = f'{OUT}/phase3_metrics.json'
if not os.path.exists(metrics_path):
    raise FileNotFoundError(
        f"Not found: {metrics_path}\n"
        "Run phase3_create.py first."
    )
with open(metrics_path) as f:
    all_metrics = json.load(f)

versions = list(all_metrics.keys())
print(f"Versions loaded: {versions}")

# ── Load predictions ──────────────────────────────────────────────────────────
preds = {}
for v in versions:
    path = f'{OUT}/phase3_{v}_preds.csv'
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found, skipping {v}")
        continue
    preds[v] = pd.read_csv(path)
    print(f"  {v}: {len(preds[v])} test samples loaded")

# ── Load importances ──────────────────────────────────────────────────────────
importances = {}
for v in versions:
    path = f'{OUT}/phase3_{v}_importance.csv'
    if os.path.exists(path):
        importances[v] = pd.read_csv(path)

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: ACCURACY SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

rows = []
for v, m in all_metrics.items():
    rows.append([
        v.upper(),
        str(m['n_features']),
        f"{m['accuracy']:.1%}",
        f"{m['kappa']:.3f}",
        str(m.get('n_train', '—')),
    ])

table = ax.table(
    cellText=rows,
    colLabels=['Version', 'Features', 'Accuracy', 'Kappa', 'Train samples'],
    cellLoc='center', loc='center',
    bbox=[0, 0, 1, 1]
)
table.auto_set_font_size(False)
table.set_fontsize(11)

# Style header
for j in range(5):
    table[0, j].set_facecolor('#2D6A4F')
    table[0, j].set_text_props(color='white', fontweight='bold')

# Highlight best accuracy row
best_v = max(all_metrics, key=lambda v: all_metrics[v]['accuracy'])
best_row = versions.index(best_v) + 1
for j in range(5):
    table[best_row, j].set_facecolor('#eae2b7')

fig.suptitle('Phase 3 — Model Accuracy Summary', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{OUT}/phase3_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUT}/phase3_summary_table.png")

print("\n=== ACCURACY SUMMARY ===")
for v, m in all_metrics.items():
    marker = '  <-- best' if v == best_v else ''
    print(f"  {v.upper()}  {m['n_features']:2d} features  "
          f"Accuracy={m['accuracy']:.1%}  Kappa={m['kappa']:.3f}{marker}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: CONFUSION MATRICES (one per version)
# ══════════════════════════════════════════════════════════════════════════════
class_labels = [n.replace('_', '\n') for n in CLASS_NAMES]

for v, df in preds.items():
    true_codes = df['true_code'].tolist()
    pred_codes = df['pred_code'].tolist()

    cm_counts = confusion_matrix(true_codes, pred_codes, labels=TARGET_CODES)
    cm_norm   = cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True)

    m = all_metrics[v]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f'Phase 3 {v.upper()} — {m["n_features"]} features  '
        f'Accuracy={m["accuracy"]:.1%}  Kappa={m["kappa"]:.3f}',
        fontsize=13, fontweight='bold'
    )

    # Raw counts
    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[0], cbar=False, linewidths=0.4)
    axes[0].set_title('Counts', fontsize=11)
    axes[0].set_ylabel('True class', fontsize=10)
    axes[0].set_xlabel('Predicted class', fontsize=10)

    # Normalised by true class
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[1], linewidths=0.4,
                cbar_kws={'label': 'Recall (fraction of true class)'})
    axes[1].set_title('Normalised by true class', fontsize=11)
    axes[1].set_ylabel('True class', fontsize=10)
    axes[1].set_xlabel('Predicted class', fontsize=10)

    plt.tight_layout()
    path = f'{OUT}/phase3_cm_{v}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: FEATURE IMPORTANCE (one per version that has importances)
# ══════════════════════════════════════════════════════════════════════════════
for v, imp_df in importances.items():
    top_n = min(20, len(imp_df))
    plot_df = imp_df.head(top_n).sort_values('importance')

    # Colour bars by feature type
    def bar_color(name):
        if name.startswith('harm_') or any(
            name.endswith(s) for s in ['_c','_cos1','_sin1','_cos2','_sin2',
                                        '_amp1','_amp2','_phase1']):
            return '#9B59B6'   # harmonic — purple
        if name.startswith('shape_'):
            return '#E8A020'   # shape — amber
        return '#2D6A4F'       # monthly snapshot — green

    colors = [bar_color(n) for n in plot_df['feature']]

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.38)))
    ax.barh(plot_df['feature'], plot_df['importance'], color=colors, edgecolor='white')
    ax.set_xlabel('Gini importance', fontsize=11)
    ax.set_title(f'Feature Importance — {v.upper()} '
                 f'(top {top_n} of {len(imp_df)})', fontsize=12, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor='#2D6A4F', label='Monthly snapshot'),
        Patch(facecolor='#E8A020', label='Shape / curve'),
        Patch(facecolor='#9B59B6', label='Harmonic (V4)'),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc='lower right')

    plt.tight_layout()
    path = f'{OUT}/phase3_importance_{v}.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# TEXT: CLASSIFICATION REPORTS + ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
report_path = f'{OUT}/phase3_error_analysis.txt'
with open(report_path, 'w') as out_f:
    for v, df in preds.items():
        true_codes = df['true_code'].tolist()
        pred_codes = df['pred_code'].tolist()
        m = all_metrics[v]

        out_f.write(f"\n{'='*60}\n")
        out_f.write(f"VERSION {v.upper()}  "
                    f"{m['n_features']} features  "
                    f"Accuracy={m['accuracy']:.1%}  "
                    f"Kappa={m['kappa']:.3f}\n")
        out_f.write(f"{'='*60}\n\n")

        out_f.write(classification_report(
            true_codes, pred_codes,
            labels=TARGET_CODES, target_names=CLASS_NAMES
        ))

        # Biggest off-diagonal errors
        cm_counts = confusion_matrix(true_codes, pred_codes, labels=TARGET_CODES)
        cm_norm   = cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True)
        out_f.write("\nBiggest confusion errors (>5% of true class):\n")
        found_any = False
        for i, true_name in enumerate(CLASS_NAMES):
            for j, pred_name in enumerate(CLASS_NAMES):
                if i != j and cm_norm[i, j] > 0.05:
                    out_f.write(
                        f"  {true_name} -> {pred_name}: "
                        f"{cm_counts[i,j]} samples ({cm_norm[i,j]:.1%})\n"
                    )
                    found_any = True
        if not found_any:
            out_f.write("  None above 5%\n")
        out_f.write("\n")

print(f"Saved: {report_path}")

# Print summary to console too
print("\n=== PER-CLASS REPORT (best version) ===")
best_df = preds[best_v]
print(classification_report(
    best_df['true_code'].tolist(),
    best_df['pred_code'].tolist(),
    labels=TARGET_CODES, target_names=CLASS_NAMES
))

print("\nPhase 3 analysis complete.")
