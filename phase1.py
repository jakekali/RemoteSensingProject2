# phase1.py — Landscape Assessment
# Determines which crops dominate Iowa + Illinois using the 2023 CDL.
# Outputs: phase1_crop_distribution.png, phase1_waffle.png
# Run with: venv/Scripts/python phase1.py

from core import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import numpy as np

print("=" * 60)
print("PHASE 1: LANDSCAPE ASSESSMENT")
print("=" * 60)

# ── Pixel count per class ─────────────────────────────────────────────────────
print("Computing CDL histogram (scale=300m)...")
hist = cdl_2023.reduceRegion(
    reducer=ee.Reducer.frequencyHistogram(),
    geometry=study_region,
    scale=300,
    maxPixels=1e9,
    tileScale=4
)
hist_dict = hist.getInfo()['cropland']
print(f"Unique CDL classes found: {len(hist_dict)}")

# ── Area dataframe ────────────────────────────────────────────────────────────
SCALE_USED  = 300
PIXEL_TO_HA = (SCALE_USED ** 2) / 10_000

records = []
for code_str, count in hist_dict.items():
    code    = int(code_str)
    area_ha = count * PIXEL_TO_HA
    records.append({
        'code':       code,
        'name':       CDL_NAMES.get(code, f'Class {code}'),
        'pixels':     count,
        'area_ha':    area_ha,
        'area_Mha':   area_ha / 1e6,
        'area_wsp_k': area_ha / WSP_HA / 1e3,
    })

df = (pd.DataFrame(records)
      .sort_values('area_ha', ascending=False)
      .reset_index(drop=True))
df['area_pct'] = df['area_ha'] / df['area_ha'].sum() * 100

print("\nTop 15 classes by area:")
print(
    df[['code', 'name', 'area_ha', 'area_pct', 'area_wsp_k']]
    .head(15)
    .rename(columns={'area_ha': 'Area (ha)', 'area_pct': 'Area (%)',
                     'area_wsp_k': 'Area (000s WSP)'})
    .to_string(index=False, float_format=lambda x: f'{x:,.1f}')
)

# ── Top-5 crop selection ──────────────────────────────────────────────────────
df_crops = df[~df['code'].isin(NON_CROP_CODES)].copy()
top5 = df_crops[df_crops['code'] != 26].head(5)   # exclude Dbl_Crop from targets

print("\nTop 5 target crop classes:")
for rank, (_, row) in enumerate(top5.iterrows(), 1):
    print(f"  {rank}. {row['name']} (code {row['code']}): "
          f"{row['area_Mha']:.3f} Mha  ({row['area_pct']:.1f}%)")
print(f"\nTop 5 cover {top5['area_pct'].sum():.1f}% of total mapped area")

# ── Figure 1: Horizontal bar + pie ───────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('2023 Cropland Distribution — Iowa and Illinois',
             fontsize=14, fontweight='bold', y=1.02)

# Bar chart
ax1 = axes[0]
plot_df    = df_crops.head(15).copy()
bar_colors = ['#e85d04' if i < 5 else '#adb5bd' for i in range(len(plot_df))]
bars = ax1.barh(plot_df['name'][::-1], plot_df['area_Mha'][::-1],
                color=bar_colors[::-1], edgecolor='white', linewidth=0.5)
ax1.set_xlabel('Area (Million hectares)', fontsize=11)
ax1.set_title('Top 15 Crop Classes by Area', fontsize=12)
for bar, (_, row) in zip(bars[::-1], plot_df.iterrows()):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
             f"{row['area_pct']:.1f}%", va='center', ha='left', fontsize=8, color='#333')
ax1.legend(handles=[Patch(facecolor='#e85d04', label='Top 5 targets'),
                    Patch(facecolor='#adb5bd', label='Other crops')],
           loc='lower right', fontsize=9)
ax1.spines[['top', 'right']].set_visible(False)

# Pie chart
ax2 = axes[1]
other_crop = df_crops.iloc[5:]['area_Mha'].sum()
non_crop   = df[df['code'].isin(NON_CROP_CODES)]['area_Mha'].sum()
sizes  = list(top5['area_Mha']) + [other_crop, non_crop]
labels = list(top5['name'])    + ['Other Crops', 'Non-Crop']
colors = ['#e85d04','#f77f00','#fcbf49','#eae2b7','#d62828','#6c757d','#dee2e6']
wedges, texts, autotexts = ax2.pie(
    sizes, labels=labels, colors=colors,
    autopct=lambda p: f'{p:.1f}%' if p > 3 else '',
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(edgecolor='white', linewidth=1.5)
)
for t in texts:      t.set_fontsize(9)
for t in autotexts:  t.set_fontsize(8)
ax2.set_title('Land Area Composition', fontsize=12)

plt.tight_layout()
plt.savefig(f'{OUT}/phase1_crop_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved: {OUT}/phase1_crop_distribution.png")

# ── Figure 2: Waffle chart ────────────────────────────────────────────────────
GRID_N, COLS, ROWS = 1000, 40, 25

top5_plot = top5[['name', 'area_pct']].rename(columns={'area_pct': 'pct'})

NON_CROP_NAMES = [
    'Grassland/Pasture', 'Deciduous Forest', 'Developed/Low Intensity',
    'Developed/Open Space', 'Developed/Open Space', 'Developed/Med Intensity', 'Woody Wetlands',
    'Open Water', 'Herbaceous Wetlands', 'Developed/High Intensity',
    'Evergreen Forest', 'Mixed Forest', 'Shrubland',
]
non_crop_df = (
    df[df['name'].isin(NON_CROP_NAMES)][['name', 'area_pct']]
    .sort_values('area_pct', ascending=False)
    .head(5)
    .rename(columns={'area_pct': 'pct'})
)
remainder = df[df['name'].isin(NON_CROP_NAMES)]['area_pct'].sum() - non_crop_df['pct'].sum()
non_crop_df = pd.concat(
    [non_crop_df, pd.DataFrame([{'name': 'Other Non-Crop', 'pct': remainder}])],
    ignore_index=True
)

plot_classes = []
for (_, row), color in zip(top5_plot.iterrows(),
                           ['#E8A020','#2D6A4F','#80B918','#4A7C59','#C9A84C']):
    plot_classes.append({'name': row['name'], 'pct': row['pct'],
                         'color': color, 'section': 'crop'})
for (_, row), color in zip(non_crop_df.iterrows(),
                           ['#95C47A','#5A7A3A','#A89880','#B0C4B1','#6AAFC4','#D0CFC8']):
    plot_classes.append({'name': row['name'], 'pct': row['pct'],
                         'color': color, 'section': 'noncrop'})

counts    = [round(c['pct'] * 10) for c in plot_classes]
counts[0] += GRID_N - sum(counts)
cells = []
for c, n in zip(plot_classes, counts):
    cells.extend([c['color']] * n)
grid = np.array(cells[:GRID_N]).reshape(ROWS, COLS)

TOTAL_HA = df['area_ha'].sum()
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(17, 7.5),
                               gridspec_kw={'width_ratios': [2.2, 1]},
                               facecolor='white')
fig.suptitle(
    f'Iowa + Illinois  Land Area by Cover Type  (2023 CDL)\n'
    f'Each cell = 1,000 Washington Square Parks  '
    f'Total ~ {TOTAL_HA/1e6:.1f} Mha  {TOTAL_HA/WSP_HA/1e6:.1f}M WSP',
    fontsize=12, fontweight='bold', y=1.01
)
ax.set_xlim(0, COLS); ax.set_ylim(0, ROWS); ax.set_aspect('equal'); ax.axis('off')

CELL, PAD = 0.82, 0.09
for r in range(ROWS):
    for c in range(COLS):
        color = grid[ROWS - 1 - r, c]
        ax.add_patch(mpatches.FancyBboxPatch(
            (c + PAD, r + PAD), CELL, CELL,
            boxstyle='round,pad=0', mutation_scale=CELL * 0.12 * 80,
            facecolor=color, edgecolor='white', linewidth=0.6
        ))
        ax.add_patch(mpatches.FancyBboxPatch(
            (c + PAD + 0.05, r + PAD + 0.55), CELL - 0.10, 0.20,
            boxstyle='round,pad=0', mutation_scale=6,
            facecolor='white', edgecolor='none', alpha=0.18, zorder=3
        ))

ax2.axis('off'); ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)
y, prev_section = 0.97, None
for c in plot_classes:
    if c['section'] != prev_section:
        label = 'Target crop classes' if c['section'] == 'crop' else 'Non-crop land cover'
        ax2.text(0.0, y + 0.01, label, transform=ax2.transAxes,
                 fontsize=9, color='#888', style='italic', va='bottom')
        y -= 0.045
        prev_section = c['section']
    wsps_k = round(c['pct'] / 100 * TOTAL_HA / WSP_HA / 1000)
    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.0, y - 0.022), 0.075, 0.038, boxstyle='round,pad=0.005',
        facecolor=c['color'], edgecolor='#ccc', linewidth=0.5,
        transform=ax2.transAxes, clip_on=False
    ))
    ax2.text(0.10, y, c['name'].replace('_', ' '),
             transform=ax2.transAxes, fontsize=10.5, va='center', color='#222')
    ax2.text(0.99, y, f"{c['pct']:.1f}%   {wsps_k:,}k WSP",
             transform=ax2.transAxes, fontsize=9.5, va='center', ha='right', color='#666')
    y -= 0.075
ax2.text(0.0, 0.01, '1 WSP = Washington Square Park, NYC (3.95 ha)',
         transform=ax2.transAxes, fontsize=8.5, color='#aaa', style='italic')

plt.tight_layout()
plt.savefig(f'{OUT}/phase1_waffle.png', dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f"Saved: {OUT}/phase1_waffle.png")

# ── SPECTRAL WISDOM (Preview for Phase 2) ────────────────────────────────────
print("\n" + "=" * 40)
print("SPECTRAL WISDOM: BEYOND RAW BANDS")
print("=" * 40)
print("Recognizing that raw bands aren't enough, we've implemented expert features.")

from feature_registry import get_spectral_wisdom
print("Computing statewide July separation (scale=3km)...")
try:
    df_wisdom = get_spectral_wisdom(MONTHLY_2023, cdl_2023, study_region, scale=3000)
    print(df_wisdom.round(3))
    print(f"  NDVI Difference: {df_wisdom.loc['Corn','NDVI'] - df_wisdom.loc['Soybeans','NDVI']:.3f}")
    print(f"  GCVI Difference: {df_wisdom.loc['Corn','GCVI'] - df_wisdom.loc['Soybeans','GCVI']:.3f}")
    print("\nCrucially, GCVI provides much stronger separation for Corn/Soybeans in July.")
except Exception as e:
    print(f"  [Skipped spectral preview: {e}]")

print("\nPhase 1 complete.")
