# dva_ica_combined.py
# Runs DVA and ICA pipelines sequentially then produces cross-reference overlays.
# Edit CONFIGURATION below; individual pipeline parameters are set in
# dva_analysis.py and ica_analysis.py.

from dva import run_dva
from ica import run_ica
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to raw discharge data — same file used for both DVA and ICA
# Expected columns: cycle_number, capacity_mAh, voltage_V
CSV_PATH = "checkup_discharge_curves.csv"

# Or point to previously saved processed CSVs to skip smoothing entirely
DVA_RESUME_CSV = "nmc532_dva_processed.csv"    # e.g. "nmc532_dva_processed.csv"
ICA_RESUME_CSV = None    # e.g. "nmc532_ica_processed.csv"

# Cycles to include in the overlay plots (None = all cycles)
CYCLES_TO_PLOT = [10, 30, 50, 70]    # e.g. [0, 20, 40, 60, 80]

# Y-axis limits for the overlay (None = autoscale)
DVA_YLIM = None          # e.g. (-0.001, 0)
ICA_YLIM = None          # e.g. (0, 500)

# Output filenames
OVERLAY_PLOT_OUT = "dva_ica_voltage_overlay.png"

# =============================================================================
# RUN PIPELINES
# =============================================================================

print(f"\n{'='*60}")
print("DVA PIPELINE")
print(f"{'='*60}")
(dva_df, dva_features_df, dva_tracked_df, dva_features,
 cycle_col, capacity_col, voltage_col, dva_csv_out) = run_dva(
    csv_path=CSV_PATH,
    resume_csv=DVA_RESUME_CSV,
    dva_ylim=DVA_YLIM,
    interactive=True
)

print(f"\n{'='*60}")
print("ICA PIPELINE")
print(f"{'='*60}")
(ica_df, ica_features_df, ica_tracked_df, ica_features,
 _, _, _, ica_csv_out) = run_ica(
    csv_path=CSV_PATH,
    resume_csv=ICA_RESUME_CSV,
    ica_ylim=ICA_YLIM,
    interactive=True
)

# =============================================================================
# DVA + ICA VOLTAGE-DOMAIN OVERLAY
# Both plotted against voltage so features can be cross-referenced directly.
# DVA troughs (solid) should bracket ICA peaks (dashed) — one ICA peak
# between each pair of adjacent DVA troughs, plus one before the first
# trough and one after the last.
# =============================================================================

cycles = sorted(dva_df[cycle_col].unique())
if CYCLES_TO_PLOT is not None:
    cycles = [c for c in cycles if int(c) in CYCLES_TO_PLOT]

n      = len(cycles)
colors = [cm.viridis(i / max(n - 1, 1)) for i in range(n)]

fig, ax1 = plt.subplots(figsize=(11, 5))
ax2 = ax1.twinx()

for i, cyc in enumerate(cycles):
    dva_sub = dva_df[dva_df[cycle_col] == cyc].dropna(
                  subset=[voltage_col, "dVdQ_smooth"])
    ica_sub = ica_df[ica_df[cycle_col] == cyc].dropna(
                  subset=[voltage_col, "dQdV_smooth"])

    ax1.plot(dva_sub[voltage_col], dva_sub["dVdQ_smooth"],
             color=colors[i], linewidth=0.9, alpha=0.85,
             label=f"DVA cyc {int(cyc)}" if i == 0 else "")
    ax2.plot(ica_sub[voltage_col], ica_sub["dQdV_smooth"],
             color=colors[i], linewidth=0.9, alpha=0.85,
             linestyle="--",
             label=f"ICA cyc {int(cyc)}" if i == 0 else "")

ax1.invert_xaxis()   # high voltage left → low voltage right (discharge direction)
ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax2.axhline(0, color="black", linewidth=0.5, linestyle=":")

ax1.set_xlabel("Voltage (V)")
ax1.set_ylabel("dV/dQ  (V/mAh)  — solid", color="tab:blue")
ax2.set_ylabel("dQ/dV  (mAh/V)  — dashed", color="tab:orange")

if DVA_YLIM: ax1.set_ylim(DVA_YLIM)
if ICA_YLIM: ax2.set_ylim(ICA_YLIM)

# colorbar for cycle number
sm = plt.cm.ScalarMappable(
         cmap=cm.viridis,
         norm=plt.Normalize(vmin=int(min(cycles)), vmax=int(max(cycles))))
sm.set_array([])
plt.colorbar(sm, ax=[ax1, ax2], label=f"Cycle ({cycle_col})")

# legend entries for line style only
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color="grey", linewidth=1.2, label="DVA  dV/dQ (solid)"),
    Line2D([0], [0], color="grey", linewidth=1.2, linestyle="--",
           label="ICA  dQ/dV (dashed)")
]
ax1.legend(handles=legend_elements, loc="upper left", fontsize=8)

plt.title("DVA and ICA — voltage domain cross-reference\n"
          "ICA peaks (dashed) should sit between adjacent DVA troughs (solid)")
plt.tight_layout()
plt.savefig(OVERLAY_PLOT_OUT, dpi=150)
plt.show()
plt.close("all")
print(f"\nSaved overlay: {OVERLAY_PLOT_OUT}")

# =============================================================================
# FEATURE CROSS-REFERENCE TABLE
# For each cycle, print DVA trough positions (V) and ICA peak positions (V)
# side by side so the bracketing relationship can be verified numerically.
# =============================================================================

print(f"\n{'='*60}")
print("DVA / ICA feature cross-reference — voltage positions by cycle")
print(f"{'='*60}")
print(f"  DVA troughs mark transitions between lithiation equilibria.")
print(f"  ICA peaks mark concurrent lithiation equilibria at both electrodes.")
print(f"  Expected: one ICA peak bracketed between each pair of DVA troughs.\n")

all_cycles = sorted(dva_features_df[cycle_col].unique())
for cyc in all_cycles:
    # DVA trough voltage positions — map capacity position to voltage
    dva_cyc = dva_features_df[dva_features_df[cycle_col] == cyc]
    dva_sub  = dva_df[dva_df[cycle_col] == cyc].dropna(
                   subset=[capacity_col, voltage_col])

    def capacity_to_voltage(cap_mah):
        """Nearest voltage for a given capacity in the discharge curve."""
        idx = (dva_sub[capacity_col] - cap_mah).abs().idxmin()
        return dva_sub.loc[idx, voltage_col]

    dva_voltages = [capacity_to_voltage(row["x_position"])
                    for _, row in dva_cyc.iterrows()]

    # ICA peak voltage positions (already on voltage axis)
    ica_cyc = ica_features_df[ica_features_df[cycle_col] == cyc]
    ica_voltages = sorted(ica_cyc["x_position"].tolist(), reverse=True)

    print(f"  Cycle {int(cyc):>4}:")
    print(f"    DVA troughs (V): "
          + "  ".join(f"{v:.4f}" for v in sorted(dva_voltages, reverse=True)))
    print(f"    ICA peaks   (V): "
          + "  ".join(f"{v:.4f}" for v in ica_voltages))
    # check n+1 rule
    n_dva = len(dva_voltages)
    n_ica = len(ica_voltages)
    status = "OK" if n_ica == n_dva + 1 else f"WARNING: expected {n_dva+1} ICA peaks, got {n_ica}"
    print(f"    n+1 check: {n_dva} DVA troughs, {n_ica} ICA peaks — {status}\n")