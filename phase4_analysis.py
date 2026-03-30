# phase4_analysis.py — Generalization: Evaluation & Comparison Plots
# Reads the CSVs/JSON saved by phase4_create.py and generates all figures.
# Re-run as many times as needed — touches no GEE, no training.
#
# Inputs (from outputs/):
#   phase4_metrics.json
#   phase3_metrics.json           (for in-training-region baseline)
#   phase4_{county}_{v}_preds.csv
#   phase4_{county}_class_map.png  (thumbnails, already saved by create)
#   phase4_{county}_cdl_map.png
#
# Outputs (to outputs/):
#   phase4_accuracy_comparison.png   grouped bar: Phase3 vs county vs county
#   phase4_cm_{county}_{v}.png       confusion matrix (best version per county)
#   phase4_generalization_report.txt per-class breakdown + biggest errors
#
# Run with: venv/Scripts/python phase4_analysis.py

from core import *
import json
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

print("=" * 60)
print("PHASE 4 ANALYSIS: GENERALIZATION EVALUATION")
print("=" * 60)

# ── Load Phase 3 baseline ─────────────────────────────────────────────────────
p3_path = f'{OUT}/phase3_metrics.json'
if not os.path.exists(p3_path):
    raise FileNotFoundError(
        f"Not found: {p3_path}\n"
        "Run phase3_create.py (and phase3_analysis.py) first."
    )
with open(p3_path) as f:
    phase3_metrics = json.load(f)

# ── Load Phase 4 metrics ───────────────────────────────────────────────────────
p4_path = f'{OUT}/phase4_metrics.json'
if not os.path.exists(p4_path):
    raise FileNotFoundError(
        f"Not found: {p4_path}\n"
        "Run phase4_create.py first."
    )
with open(p4_path) as f:
    phase4_metrics = json.load(f)

# Load county metadata if available
counties_path = f'{OUT}/phase4_counties.json'
if os.path.exists(counties_path):
    with open(counties_path) as f:
        county_meta = json.load(f)
else:
    county_meta = {slug: {'display_name': slug} for slug in phase4_metrics}

counties = list(phase4_metrics.keys())
versions = list(phase3_metrics.keys())  # v1-v4

print(f"Counties: {counties}")
print(f"Versions: {versions}")

# ── Load predictions ──────────────────────────────────────────────────────────
preds = {}   # preds[slug][version] = DataFrame
for slug in counties:
    preds[slug] = {}
    for v in versions:
        path = f'{OUT}/phase4_{slug}_{v}_preds.csv'
        if os.path.exists(path):
            preds[slug][v] = pd.read_csv(path)
            print(f"  {slug} {v}: {len(preds[slug][v])} samples")
        else:
            print(f"  WARNING: {path} not found, skipping")

class_labels = [n.replace('_', '\n') for n in CLASS_NAMES]

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1: ACCURACY COMPARISON — Phase 3 (train region) vs Phase 4 counties
# Grouped bar chart: x = version, groups = [Phase3 Iowa/IL, McLean IL, Renville MN]
# ══════════════════════════════════════════════════════════════════════════════

fig, (ax_acc, ax_kap) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Phase 4 — Generalization: Accuracy Across Regions',
             fontsize=13, fontweight='bold')

n_versions = len(versions)
n_groups   = 1 + len(counties)   # Phase3 + one bar per county
bar_w      = 0.7 / n_groups
x          = np.arange(n_versions)

# Colors: first bar = training region (dark), subsequent = counties
group_colors = ['#2D6A4F', '#E8A020', '#9B59B6', '#E63946']
group_labels = ['Iowa+IL (train region)'] + [
    county_meta.get(s, {}).get('display_name', s) for s in counties
]

for gi, (label, color) in enumerate(zip(group_labels, group_colors)):
    offsets = x + (gi - n_groups / 2 + 0.5) * bar_w

    if gi == 0:
        # Phase 3 training-region accuracy
        acc_vals = [phase3_metrics.get(v, {}).get('accuracy', 0) for v in versions]
        kap_vals = [phase3_metrics.get(v, {}).get('kappa', 0)    for v in versions]
    else:
        slug = counties[gi - 1]
        acc_vals = [phase4_metrics.get(slug, {}).get(v, {}).get('accuracy', 0) for v in versions]
        kap_vals = [phase4_metrics.get(slug, {}).get(v, {}).get('kappa', 0)    for v in versions]

    ax_acc.bar(offsets, acc_vals, bar_w, label=label, color=color,
               edgecolor='white', linewidth=0.5)
    ax_kap.bar(offsets, kap_vals, bar_w, label=label, color=color,
               edgecolor='white', linewidth=0.5)

for ax, ylabel, title in [
    (ax_acc, 'Overall Accuracy', 'Overall Accuracy by Version & Region'),
    (ax_kap, "Cohen's Kappa",    "Kappa by Version & Region"),
]:
    ax.set_xticks(x)
    ax.set_xticklabels([v.upper() for v in versions])
    ax.set_xlabel('Feature Version', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.8, color='#ccc', linestyle='--', linewidth=0.8, zorder=0)
    ax.grid(axis='y', alpha=0.25, linestyle='--')

plt.tight_layout()
path = f'{OUT}/phase4_accuracy_comparison.png'
plt.savefig(path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {path}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: CONFUSION MATRICES — best version per county
# ══════════════════════════════════════════════════════════════════════════════

for slug in counties:
    county_name = county_meta.get(slug, {}).get('display_name', slug)

    # Find best version for this county
    available = {v: phase4_metrics.get(slug, {}).get(v, {})
                 for v in versions if v in phase4_metrics.get(slug, {})}
    if not available:
        print(f"  No metrics for {slug}, skipping CM")
        continue

    best_v = max(available, key=lambda v: available[v].get('kappa', 0))
    m = available[best_v]

    if best_v not in preds.get(slug, {}):
        print(f"  No predictions for {slug} {best_v}, skipping CM")
        continue

    df = preds[slug][best_v]
    true_codes = df['true_code'].tolist()
    pred_codes = df['pred_code'].tolist()

    cm_counts = confusion_matrix(true_codes, pred_codes, labels=TARGET_CODES)
    cm_norm   = cm_counts.astype(float) / (cm_counts.sum(axis=1, keepdims=True) + 1e-9)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    fig.suptitle(
        f'Phase 4 — {county_name}  ({best_v.upper()}, {m["n_features"]} features)\n'
        f'Accuracy={m["accuracy"]:.1%}  Kappa={m["kappa"]:.3f}  '
        f'[2024 Landsat → 2024 CDL reference]',
        fontsize=12, fontweight='bold'
    )

    sns.heatmap(cm_counts, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[0], cbar=False, linewidths=0.4)
    axes[0].set_title('Counts', fontsize=11)
    axes[0].set_ylabel('True class (CDL 2024)', fontsize=10)
    axes[0].set_xlabel('Predicted class', fontsize=10)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels,
                ax=axes[1], linewidths=0.4,
                cbar_kws={'label': 'Recall (fraction of true class)'})
    axes[1].set_title('Normalised by true class', fontsize=11)
    axes[1].set_ylabel('True class (CDL 2024)', fontsize=10)
    axes[1].set_xlabel('Predicted class', fontsize=10)

    plt.tight_layout()
    cm_path = f'{OUT}/phase4_cm_{slug}_{best_v}.png'
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: MAP COMPARISON PANELS
# Tile: [CDL reference] [V1 map] [V4 map] side-by-side for each county
# (Uses thumbnails already saved by phase4_create.py)
# ══════════════════════════════════════════════════════════════════════════════

for slug in counties:
    county_name = county_meta.get(slug, {}).get('display_name', slug)
    thumbnail_versions = [v for v in versions
                          if os.path.exists(f'{OUT}/phase4_{slug}_{v}_class_map.png')]
    cdl_thumb = f'{OUT}/phase4_{slug}_cdl_map.png'

    if not thumbnail_versions or not os.path.exists(cdl_thumb):
        print(f"  Map panel skipped for {slug} (thumbnails missing)")
        continue

    # Show CDL + up to 4 versions
    show_v = thumbnail_versions[:4]
    n_cols  = 1 + len(show_v)
    fig, axes = plt.subplots(1, n_cols, figsize=(4.5 * n_cols, 5))
    fig.suptitle(f'Classification Maps — {county_name} (2024)',
                 fontsize=12, fontweight='bold')

    # CDL panel
    cdl_img = plt.imread(cdl_thumb)
    axes[0].imshow(cdl_img)
    axes[0].set_title('CDL 2024\n(reference)', fontsize=10)
    axes[0].axis('off')

    for ax, v in zip(axes[1:], show_v):
        thumb_path = f'{OUT}/phase4_{slug}_{v}_class_map.png'
        img = plt.imread(thumb_path)
        ax.imshow(img)
        m = phase4_metrics.get(slug, {}).get(v, {})
        acc_str = f"Acc={m['accuracy']:.1%}" if 'accuracy' in m else ''
        ax.set_title(f'{v.upper()}\n{acc_str}', fontsize=10)
        ax.axis('off')

    # Legend
    legend_patches = [
        mpatches.Patch(facecolor='#' + hex_col, label=CLASS_NAMES[i])
        for i, hex_col in enumerate(['E8A020', '2D6A4F', '4A7C59', 'C9A84C', '06D6A0'])
    ]
    legend_patches.append(mpatches.Patch(facecolor='#808080', label='Other / unclassified'))
    fig.legend(handles=legend_patches, loc='lower center', ncol=3,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout()
    panel_path = f'{OUT}/phase4_{slug}_map_panel.png'
    plt.savefig(panel_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {panel_path}")

# ══════════════════════════════════════════════════════════════════════════════
# TEXT: GENERALIZATION REPORT
# ══════════════════════════════════════════════════════════════════════════════

report_path = f'{OUT}/phase4_generalization_report.txt'
with open(report_path, 'w') as out_f:
    out_f.write("PHASE 4 — GENERALIZATION ANALYSIS\n")
    out_f.write("Model trained on Iowa + Illinois (2023 CDL)\n")
    out_f.write("Applied to 2024 Landsat over new counties\n")
    out_f.write("=" * 70 + "\n\n")

    # Summary table
    out_f.write(f"{'Version':<8} {'Phase3 (train)':<18}")
    for slug in counties:
        name = county_meta.get(slug, {}).get('display_name', slug)
        out_f.write(f"{name:<22}")
    out_f.write("\n")
    out_f.write("-" * 70 + "\n")

    for v in versions:
        p3 = phase3_metrics.get(v, {})
        line = f"{v.upper():<8} Acc={p3.get('accuracy',0):.1%} K={p3.get('kappa',0):.3f}  "
        for slug in counties:
            m = phase4_metrics.get(slug, {}).get(v, {})
            if m:
                line += f"Acc={m['accuracy']:.1%} K={m['kappa']:.3f}       "
            else:
                line += f"{'N/A':<22}"
        out_f.write(line + "\n")

    out_f.write("\n")

    # Per-county detailed reports
    for slug in counties:
        county_name = county_meta.get(slug, {}).get('display_name', slug)
        state       = county_meta.get(slug, {}).get('state', '')

        out_f.write(f"\n{'='*70}\n")
        out_f.write(f"COUNTY: {county_name}  ({state})\n")
        out_f.write(f"{'='*70}\n")

        # Generalisation gap for each version
        out_f.write("\nGeneralization gap (Phase3 accuracy - Phase4 accuracy):\n")
        for v in versions:
            p3_acc = phase3_metrics.get(v, {}).get('accuracy', None)
            p4_acc = phase4_metrics.get(slug, {}).get(v, {}).get('accuracy', None)
            if p3_acc is not None and p4_acc is not None:
                gap = p3_acc - p4_acc
                sign = '+' if gap >= 0 else ''
                out_f.write(f"  {v.upper()}  Phase3={p3_acc:.1%}  "
                            f"Phase4={p4_acc:.1%}  gap={sign}{gap:.1%}\n")

        # Per-class breakdown for best version
        best_v = max(
            (v for v in versions if v in phase4_metrics.get(slug, {})),
            key=lambda v: phase4_metrics.get(slug, {}).get(v, {}).get('kappa', 0),
            default=None
        )
        if best_v and best_v in preds.get(slug, {}):
            df = preds[slug][best_v]
            out_f.write(f"\nPer-class report (best version: {best_v.upper()}):\n")
            out_f.write(classification_report(
                df['true_code'].tolist(),
                df['pred_code'].tolist(),
                labels=TARGET_CODES, target_names=CLASS_NAMES
            ))

            # Biggest confusions
            cm_counts = confusion_matrix(df['true_code'], df['pred_code'], labels=TARGET_CODES)
            cm_norm   = cm_counts.astype(float) / (cm_counts.sum(axis=1, keepdims=True) + 1e-9)
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

print(f"Saved: {report_path}")

# ── Console summary ────────────────────────────────────────────────────────────
print("\n=== GENERALIZATION SUMMARY ===")
print(f"\n{'Version':<8} {'Iowa+IL (train)':<20}", end='')
for slug in counties:
    name = county_meta.get(slug, {}).get('display_name', slug)
    print(f"  {name[:18]:<20}", end='')
print()
print("-" * (8 + 20 + 22 * len(counties)))

for v in versions:
    p3 = phase3_metrics.get(v, {})
    print(f"{v.upper():<8} {p3.get('accuracy',0):.1%} (K={p3.get('kappa',0):.3f})  ", end='')
    for slug in counties:
        m = phase4_metrics.get(slug, {}).get(v, {})
        if m:
            gap = p3.get('accuracy', 0) - m['accuracy']
            print(f"  {m['accuracy']:.1%} (K={m['kappa']:.3f}) gap={gap:+.1%}  ", end='')
        else:
            print(f"  {'N/A':<20}", end='')
    print()

print("\nPhase 4 analysis complete.")
