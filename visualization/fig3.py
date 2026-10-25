"""generate_fig3.py - System architecture diagram, 600+ DPI (fixed overlaps)"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(4.2, 9.5))
ax.set_xlim(0, 10)
ax.set_ylim(3.8, 24.3)
ax.axis('off')

def box(y, h, text, color='#E9F1F5', edge='#21295C', fontsize=8.3, textcolor='#1B2430', bold=False):
    rect = FancyBboxPatch((0.4, y), 9.2, h, boxstyle="round,pad=0.05,rounding_size=0.15",
                           facecolor=color, edgecolor=edge, linewidth=1.1)
    ax.add_patch(rect)
    ax.text(5, y + h/2, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold' if bold else 'normal')

def arrow(y_top, y_bottom):
    ax.annotate('', xy=(5, y_bottom), xytext=(5, y_top),
                arrowprops=dict(arrowstyle='-|>', color='#1C7293', lw=1.4))

# --- Verified non-overlapping layout: each box's top/bottom computed with a
# consistent 0.5 gap reserved for the arrow before the next box starts ---

# 1. DATA SOURCES header: top=23.8, bottom=22.6
box(22.6, 1.2, "DATA SOURCES", color='#21295C', textcolor='white', fontsize=9.5, bold=True)
arrow(22.6, 22.1)

# 2. Four data-source sub-boxes: top=22.1, bottom=20.8
y2 = 20.8
labels = ["GRACE\n(TWS anomaly)", "ERA5-Land\n(climate)", "Landsat 7/8\n(NDVI, NDBI)", "WorldPop\n(pop. density)"]
for i, lab in enumerate(labels):
    x0 = 0.5 + i*2.3
    rect = FancyBboxPatch((x0, y2), 2.1, 1.3, boxstyle="round,pad=0.04,rounding_size=0.1",
                           facecolor='#FFFFFF', edgecolor='#1C7293', linewidth=1)
    ax.add_patch(rect)
    ax.text(x0+1.05, y2+0.65, lab, ha='center', va='center', fontsize=6.6, color='#1B2430')
arrow(20.8, 20.3)

# 3. PREPROCESSING: top=20.3, bottom=19.0
box(19.0, 1.3, "PREPROCESSING\nMonthly resampling - seasonal gap-fill (GRACE)\nlinear interp. (NDVI/NDBI) - unit conversion (ERA5)", fontsize=7.2)
arrow(19.0, 18.5)

# 4. FEATURE ENGINEERING: top=18.5, bottom=17.2
box(17.2, 1.3, "FEATURE ENGINEERING\nLags: 1,2,3,6,12 mo - 3-mo rolling mean\nDistrict one-hot encoding", fontsize=7.2)
arrow(17.2, 16.7)

# 5. 6-DISTRICT PANEL: top=16.7, bottom=15.8
box(15.8, 0.9, "6-DISTRICT MONTHLY PANEL\n1,758 rows (2002-2026)", color='#C9DCE6', fontsize=7.6, bold=True)
arrow(15.8, 15.3)

# 6. MODEL TRAINING & VALIDATION header: top=15.3, bottom=14.1
box(14.1, 1.2, "MODEL TRAINING & VALIDATION", color='#21295C', textcolor='white', fontsize=8.6, bold=True)
arrow(14.1, 13.6)

# 7. Three model sub-boxes: top=13.6, bottom=12.4
y7 = 12.4
sub_labels = ["Random\nForest", "XGBoost", "Stacked\nEnsemble\n(RF+XGB+MLP)"]
for i, lab in enumerate(sub_labels):
    x0 = 0.6 + i*3.0
    rect = FancyBboxPatch((x0, y7), 2.7, 1.2, boxstyle="round,pad=0.04,rounding_size=0.1",
                           facecolor='#FFFFFF', edgecolor='#1C7293', linewidth=1)
    ax.add_patch(rect)
    ax.text(x0+1.35, y7+0.6, lab, ha='center', va='center', fontsize=6.8, color='#1B2430')
arrow(12.4, 11.9)

# 8. VALIDATION: top=11.9, bottom=10.8
box(10.8, 1.1, "VALIDATION\n1-step walk-forward + multi-step recursive\n(3 cutoffs, up to 24 mo)", fontsize=7.0)
arrow(10.8, 10.3)

# 9. BEST MODEL: top=10.3, bottom=9.4
box(9.4, 0.9, "BEST MODEL: XGBoost\n(R2 = 0.646, 1-mo-ahead)", color='#C9DCE6', fontsize=7.6, bold=True)
arrow(9.4, 8.9)

# 10. RECURSIVE FORECAST: top=8.9, bottom=7.8
box(7.8, 1.1, "RECURSIVE 24-MONTH FORECAST\nClimatology-estimated future covariates", fontsize=7.2)
arrow(7.8, 7.3)

# 11. CATEGORY MAPPING: top=7.3, bottom=6.2
box(6.2, 1.1, "CGWB-STYLE CATEGORY MAPPING\nSafe / Semi-Critical / Critical / Over-Exploited", fontsize=7.0)
arrow(6.2, 5.7)

# 12. OUTPUT: top=5.7, bottom=4.5
box(4.5, 1.2, "OUTPUT: FORECAST DASHBOARD\n& VISUALIZATION", color='#21295C', textcolor='white', fontsize=8.0, bold=True)

plt.tight_layout()
plt.savefig('fig3_architecture.png', dpi=600, bbox_inches='tight', facecolor='white')
print("Saved fig3_architecture.png at 600 DPI, no overlaps")