# local_model_testing/compare_to_gee.py
# Step 3: Side-by-side comparison of local sklearn models vs GEE RF (Phase 3).
# Zero GEE calls — reads local JSONs only.
#
# Outputs -> outputs/
#   compare_accuracy.png     grouped bar: GEE RF vs local models across versions
#   compare_kappa.png
#   compare_summary.txt      full text table

import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR    = os.path.join(os.path.dirname(__file__), 'outputs')
ROOT_OUT   = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

# ── Load metrics ──────────────────────────────────────────────────────────────
local_path = os.path.join(OUT_DIR, 'local_metrics.json')
gee_path   = os.path.join(ROOT_OUT, 'phase3_metrics.json')

if not os.path.exists(local_path):
    raise FileNotFoundError("Run train_local_models.py first.")
if not os.path.exists(gee_path):
    raise FileNotFoundError("phase3_metrics.json not found — run phase3_create.py first.")

with open(local_path)  as f: local_metrics = json.load(f)
with open(gee_path)    as f: gee_metrics   = json.load(f)

versions = sorted(set(local_metrics) & set(gee_metrics))
local_model_names = ['RF_200', 'RF_500', 'GBT', 'SVM', 'LR']

# ── Plot: Accuracy comparison ──────────────────────────────────────────────────
fig, (ax_acc, ax_kap) = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle('Local sklearn Models vs GEE Random Forest',
             fontsize=13, fontweight='bold')

# One group per version, bars = [GEE_RF] + local models
colors = {
    'GEE_RF': '#1A1A2E',
    'RF_200': '#16213E',
    'RF_500': '#2D6A4F',
    'GBT':    '#E8A020',
    'SVM':    '#9B59B6',
    'LR':     '#06D6A0',
}

all_names = ['GEE_RF'] + local_model_names
n_groups  = len(all_names)
x         = np.arange(len(versions))
bar_w     = 0.7 / n_groups

for gi, name in enumerate(all_names):
    offset = x + (gi - n_groups/2 + 0.5) * bar_w
    if name == 'GEE_RF':
        acc_vals = [gee_metrics.get(v, {}).get('accuracy', 0) for v in versions]
        kap_vals = [gee_metrics.get(v, {}).get('kappa',    0) for v in versions]
    else:
        acc_vals = [local_metrics.get(v, {}).get(name, {}).get('accuracy', 0) for v in versions]
        kap_vals = [local_metrics.get(v, {}).get(name, {}).get('kappa',    0) for v in versions]

    ax_acc.bar(offset, acc_vals, bar_w, label=name, color=colors[name],
               edgecolor='white', linewidth=0.5)
    ax_kap.bar(offset, kap_vals, bar_w, label=name, color=colors[name],
               edgecolor='white', linewidth=0.5)

for ax, ylabel, title in [
    (ax_acc, 'Overall Accuracy', 'Accuracy: Local vs GEE'),
    (ax_kap, "Cohen's Kappa",   'Kappa: Local vs GEE'),
]:
    ax.set_xticks(x)
    ax.set_xticklabels([v.upper() for v in versions])
    ax.set_xlabel('Feature Version', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color='#ddd', linestyle='--', linewidth=0.8, zorder=0)
    ax.grid(axis='y', alpha=0.2, linestyle='--')
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
acc_path = os.path.join(OUT_DIR, 'compare_accuracy.png')
plt.savefig(acc_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {acc_path}")

# ── Text summary ──────────────────────────────────────────────────────────────
summary_path = os.path.join(OUT_DIR, 'compare_summary.txt')
with open(summary_path, 'w') as f:
    col_w = 14
    header = f"{'Version':<8} {'GEE_RF':<{col_w}}" + \
             ''.join(f"{n:<{col_w}}" for n in local_model_names)
    f.write(header + "\n")
    f.write("-" * len(header) + "\n")
    for v in versions:
        gee_acc = gee_metrics.get(v, {}).get('accuracy', 0)
        gee_kap = gee_metrics.get(v, {}).get('kappa', 0)
        line = f"{v.upper():<8} {gee_acc:.1%}/K{gee_kap:.2f}  "
        for mn in local_model_names:
            m = local_metrics.get(v, {}).get(mn, {})
            if m:
                line += f"{m['accuracy']:.1%}/K{m['kappa']:.2f}  "
            else:
                line += f"{'N/A':<{col_w}}"
        f.write(line + "\n")

print(f"Saved: {summary_path}")

# ── Console print ──────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print("COMPARISON SUMMARY  (Accuracy / Kappa)")
print(f"{'='*70}")
print(f"{'Version':<8} {'GEE RF':<14}", end='')
for mn in local_model_names:
    print(f"  {mn:<14}", end='')
print()
print("-" * 70)
for v in versions:
    gee = gee_metrics.get(v, {})
    print(f"{v.upper():<8} {gee.get('accuracy',0):.1%} K={gee.get('kappa',0):.3f} ", end='')
    for mn in local_model_names:
        m = local_metrics.get(v, {}).get(mn, {})
        if m:
            print(f"  {m['accuracy']:.1%} K={m['kappa']:.3f} ", end='')
        else:
            print(f"  {'N/A':<14}", end='')
    print()
