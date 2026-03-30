# phase2.py — Crop Phenology Exploration
# Characterises how the 5 target crops + Dbl_Crop behave spectrally across 2023.
# Uses monthly composites (not per-scene) to avoid GEE timeout.
# Outputs: phase2_stats.csv, phase2_phenology.png, phase2_phenology_clean.png,
#          phase2_separability.png
# Run with: venv/Scripts/python phase2.py

from core import *
import itertools
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

print("=" * 60)
print("PHASE 2: CROP PHENOLOGY")
print("=" * 60)
print(f"L8+L9 scenes in 2023: {landsat_2023.size().getInfo()}")

# ── Sample constants (re-defined since not in core.py) ────────────────────────
SAMPLE_CODES  = [1, 5, 36, 24, 28, 26, 0]
SAMPLE_LABELS = {
    1:  'Corn', 5: 'Soybeans', 36: 'Alfalfa',
    24: 'Winter_Wheat', 28: 'Oats', 26: 'Dbl_Crop_WinWht/Soybeans',
    0:  'Other'
}

# ── Stratified samples (25 pts/class, phenology only) ────────────────────────
print("Drawing stratified samples...")
samples = make_stratified_samples(
    n_per_class=25,
    codes=SAMPLE_CODES,
    seed=42
)
print(f"Phenology samples: {samples.size().getInfo()} pts")

# ── Sample each monthly composite ────────────────────────────────────────────
# Monthly composites are already built in core.py (MONTHLY_2023).
# Sampling 12 composites × 150 pts is fast — no GEE timeout risk.
print("Sampling monthly composites (12 months)...")

from feature_registry import analyze_phenology, build_feature_image
df_stats, monthly_agg, shape_summary = analyze_phenology(MONTHLY_2023, samples, SAMPLE_LABELS)

monthly_agg['date'] = pd.to_datetime(
    monthly_agg['month'].apply(lambda m: f'2023-{m:02d}-15')
)

monthly_agg.to_csv(f'{OUT}/phase2_stats.csv', index=False)
print(f"\nSaved: {OUT}/phase2_stats.csv  ({len(monthly_agg)} rows)")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n=== GCVI vs NDVI: CORN/SOY SEPARATION (JULY) ===")
july_stats = df_stats[df_stats['month'] == 7]
cs_july = july_stats[july_stats['crop_name'].isin(['Corn', 'Soybeans'])]
sep_report = cs_july.groupby('crop_name')[['NDVI', 'GCVI']].mean()
print(sep_report)
print(f"  NDVI Diff (C-S): {sep_report.loc['Corn','NDVI'] - sep_report.loc['Soybeans','NDVI']:.3f}")
print(f"  GCVI Diff (C-S): {sep_report.loc['Corn','GCVI'] - sep_report.loc['Soybeans','GCVI']:.3f}")

print("\n=== TEMPORAL REDUCTION (SHAPE) METRICS SUMMARY ===")
print(shape_summary.round(3))

# ── Hierarchical Classifier Map Preview (McLean County) ───────────────────────
print("\n" + "=" * 40)
print("CLASSIFIER PREVIEW: MAP SEPARABILITY")
print("=" * 40)
print("Testing hierarchical separation on McLean County (2023)...")

try:
    # 1. Quick Model: Train on phenology samples (which now include 'Other')
    mclean_geom = Config.get_county_regions()['mclean']
    
    # Feature registry needs a dictionary for source keys
    sources = {
        **{m_name: MONTHLY_2023[i+1] for i, m_name in enumerate(['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])},
    }
    v1_img, v1_names = build_feature_image('v1', sources)
    # Train hierarchical model on current samples
    print("  Step 1: Sampling training features from 2023 composites...")
    train_fc = v1_img.sampleRegions(collection=samples, properties=['cdl_code'], scale=30)

    # DEBUG: Check if we have multiple classes
    counts = train_fc.aggregate_histogram('cdl_code').getInfo()
    print(f"  Step 2: Preview Training Class Distribution: {counts}")

    # Apply hierarchical logic
    print("  Step 3: Training hierarchical RF model (Layer 1 + Layer 2)...")
    final_img, rf_l1, rf_l2 = hierarchical_classify(train_fc, v1_img, v1_names)

    # 2. Export Visual Preview (Matching Phase 4 style)
    print("  Step 4: Requesting map preview thumbnails from Earth Engine...")
    palette = ['E8A020', '2D6A4F', '4A7C59', 'C9A84C', '06D6A0', '9B59B6']

    vis_params = {'min': 1, 'max': 36, 'palette': ['ffffff'] * 36}
    for i, code in enumerate([1, 5, 36, 24, 28, 26]):
        vis_params['palette'][code-1] = palette[i]

    # Use 2023 CDL as reference comparison
    cdl_vis = {'min': 0, 'max': 254} # Full CDL range
    url_cdl = cdl_2023.clip(mclean_geom).getThumbURL({'dimensions': 512, 'region': mclean_geom})
    url_pred = final_img.clip(mclean_geom).getThumbURL({'min': 0, 'max': 36, 'palette': vis_params['palette'], 'dimensions': 512, 'region': mclean_geom})
    
    import requests
    with open(f'{OUT}/phase2_mclean_cdl_ref.png', 'wb') as f: f.write(requests.get(url_cdl).content)
    with open(f'{OUT}/phase2_mclean_pred_preview.png', 'wb') as f: f.write(requests.get(url_pred).content)
    
    # Create Side-by-Side Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(plt.imread(f'{OUT}/phase2_mclean_cdl_ref.png'))
    axes[0].set_title('CDL 2023 (Reference)', fontsize=10)
    axes[1].imshow(plt.imread(f'{OUT}/phase2_mclean_pred_preview.png'))
    axes[1].set_title('Phase 2 Hierarchical Preview (V1 Features)', fontsize=10)
    for ax in axes: ax.axis('off')
    plt.savefig(f'{OUT}/phase2_separability_map.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {OUT}/phase2_separability_map.png")

except Exception as e:
    import traceback
    print(f"  [Skipped Map Preview: {e}]")
    traceback.print_exc()

# ── Phenology plot helper ─────────────────────────────────────────────────────
SEASONS = [
    ('2023-04-01', '2023-06-01', '#a8d5a2', 'Planting'),
    ('2023-07-01', '2023-08-31', '#ffe0a0', 'Peak'),
    ('2023-09-15', '2023-11-15', '#f4a460', 'Senescence'),
]

def make_phenology_plot(data, title, filename):
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight='bold')
    panels = [('NDVI_mean', 'NDVI_std', 'NDVI'),
              ('EVI_mean',  'EVI_std',  'EVI'),
              ('LSWI_mean', 'LSWI_std', 'LSWI'),
              ('GCVI_mean', 'GCVI_std', 'GCVI')]
    for ax, (mean_col, std_col, ylabel) in zip(axes, panels):
        for start, end, color, label in SEASONS:
            ax.axvspan(pd.Timestamp(start), pd.Timestamp(end),
                       alpha=0.12, color=color, zorder=1)
        for crop, group in data.groupby('crop_name'):
            color = CROP_COLORS.get(crop, '#999')
            group = group.sort_values('date')
            ls = '--' if 'Dbl' in crop else '-'
            lw = 1.6  if 'Dbl' in crop else 2.2
            ax.plot(group['date'], group[mean_col], color=color,
                    linewidth=lw, linestyle=ls,
                    label=crop.replace('_', ' '), zorder=3)
            ax.fill_between(group['date'],
                            group[mean_col] - group[std_col],
                            group[mean_col] + group[std_col],
                            color=color, alpha=0.15, zorder=2)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.axhline(0, color='#ccc', linewidth=0.5, linestyle=':')
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.spines[['top', 'right']].set_visible(False)
        if ylabel == 'NDVI':
            ax.legend(loc='upper left', fontsize=9, ncol=2, framealpha=0.8)
            for start, end, _, label in SEASONS:
                mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
                ax.text(mid, 0.98, label, ha='center', va='top', fontsize=8,
                        color='#666', transform=ax.get_xaxis_transform())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[-1].set_xlabel('2023', fontsize=11)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {filename}")

make_phenology_plot(
    monthly_agg,
    'Crop Phenology  Iowa and Illinois (2023)\n'
    'Monthly mean +/- 1 std across 25 sample points per class',
    f'{OUT}/phase2_phenology.png'
)

# ── Corn vs soy NDVI difference by month ─────────────────────────────────────
print("\n=== CORN vs SOY NDVI DIFFERENCE BY MONTH ===")
corn_m = monthly_agg[monthly_agg['crop_name'] == 'Corn'].set_index('month')['NDVI_mean']
soy_m  = monthly_agg[monthly_agg['crop_name'] == 'Soybeans'].set_index('month')['NDVI_mean']
print((corn_m - soy_m).round(3).to_string())

# ── Bhattacharyya separability matrices ───────────────────────────────────────
def bhattacharyya(x1, x2, n_bins=30):
    lo, hi = min(x1.min(), x2.min()), max(x1.max(), x2.max())
    if lo == hi: return 0.0
    bins = np.linspace(lo, hi, n_bins)
    h1, _ = np.histogram(x1, bins=bins, density=True)
    h2, _ = np.histogram(x2, bins=bins, density=True)
    h1 /= (h1.sum() + 1e-10)
    h2 /= (h2.sum() + 1e-10)
    return -np.log(np.sum(np.sqrt(h1 * h2)) + 1e-10)

TARGET_NAMES = [v for k, v in SAMPLE_LABELS.items() if k != 26]
MONTHS_SEP   = {'January': 1, 'May': 5, 'July': 7, 'October': 10}

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('Pairwise NDVI Separability by Month (Bhattacharyya Distance)\n'
             'Higher = easier to separate', fontsize=12, fontweight='bold')

for ax, (month_label, month_num) in zip(axes, MONTHS_SEP.items()):
    month_df = df_stats[df_stats['month'] == month_num]
    mat = pd.DataFrame(
        np.zeros((len(TARGET_NAMES), len(TARGET_NAMES))),
        index=TARGET_NAMES, columns=TARGET_NAMES
    )
    for c1, c2 in itertools.combinations(TARGET_NAMES, 2):
        x1 = month_df[month_df['crop_name'] == c1]['NDVI'].dropna().values
        x2 = month_df[month_df['crop_name'] == c2]['NDVI'].dropna().values
        if len(x1) > 3 and len(x2) > 3:
            d = bhattacharyya(x1, x2)
            mat.loc[c1, c2] = d; mat.loc[c2, c1] = d
    mat.index   = [n.replace('_', ' ') for n in TARGET_NAMES]
    mat.columns = [n.replace('_', ' ') for n in TARGET_NAMES]
    sns.heatmap(mat, annot=True, fmt='.2f', cmap='YlOrRd',
                mask=np.eye(len(TARGET_NAMES), dtype=bool),
                ax=ax, linewidths=0.3, vmin=0, vmax=8,
                cbar=(ax is axes[-1]),
                cbar_kws={'label': 'Bhattacharyya distance'})
    ax.set_title(month_label, fontsize=11)
    ax.tick_params(axis='x', rotation=35, labelsize=8)
    ax.tick_params(axis='y', rotation=0,  labelsize=8)

plt.tight_layout()
plt.savefig(f'{OUT}/phase2_separability.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/phase2_separability.png")
print("\nPhase 2 complete.")
