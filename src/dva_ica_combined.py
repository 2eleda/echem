# dva_ica_combined.py
# Runs ICA then DVA pipelines sequentially, then produces cross-reference overlays.
# ICA is run first — its peaks map directly to literature values and are easier
# to validate before proceeding to DVA. All settings come from config.py.

from ica import run_ica
from dva import run_dva
from config import (CSV_PATH, CYCLE_COL, CAPACITY_COL, VOLTAGE_COL,
                    DVA_RESUME_CSV, ICA_RESUME_CSV,
                    DVA_YLIM, ICA_YLIM,
                    CYCLES_TO_PLOT, OVERLAY_PLOT_OUT)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

# =============================================================================
# RUN PIPELINES — ICA first for easier literature validation
# =============================================================================

print(f"\n{'='*60}")
print("ICA PIPELINE")
print(f"{'='*60}")
(ica_df, ica_features_df, ica_tracked_df, ica_features,
 cycle_col, capacity_col, voltage_col, ica_csv_out) = run_ica(
    csv_path=CSV_PATH,
    resume_csv=ICA_RESUME_CSV,
    cycle_col=CYCLE_COL,
    capacity_col=CAPACITY_COL,
    voltage_col=VOLTAGE_COL,
    ica_ylim=ICA_YLIM,
    interactive=True
)

print(f"\n{'='*60}")
print("DVA PIPELINE")
print(f"{'='*60}")
(dva_df, dva_features_df, dva_tracked_df, dva_features,
 _, _, _, dva_csv_out) = run_dva(
    csv_path=CSV_PATH,
    resume_csv=DVA_RESUME_CSV,
    cycle_col=CYCLE_COL,
    capacity_col=CAPACITY_COL,
    voltage_col=VOLTAGE_COL,
    dva_ylim=DVA_YLIM,
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
             color=colors[i], linewidth=0.9, alpha=0.85)
    ax2.plot(ica_sub[voltage_col], ica_sub["dQdV_smooth"],
             color=colors[i], linewidth=0.9, alpha=0.85, linestyle="--")

ax1.invert_xaxis()   # high voltage left → low voltage right (discharge direction)
ax1.axhline(0, color="black", linewidth=0.5, linestyle=":")
ax2.axhline(0, color="black", linewidth=0.5, linestyle=":")

ax1.set_xlabel("Voltage (V)")
ax1.set_ylabel("dV/dQ  (V/mAh)  — solid",   color="tab:blue")
ax2.set_ylabel("dQ/dV  (mAh/V)  — dashed",  color="tab:orange")

if DVA_YLIM: ax1.set_ylim(DVA_YLIM)
if ICA_YLIM: ax2.set_ylim(ICA_YLIM)

sm = plt.cm.ScalarMappable(
         cmap=cm.viridis,
         norm=plt.Normalize(vmin=int(min(cycles)), vmax=int(max(cycles))))
sm.set_array([])
plt.colorbar(sm, ax=[ax1, ax2], label=f"Cycle ({cycle_col})")

legend_elements = [
    Line2D([0], [0], color="grey", linewidth=1.2,
           label="DVA  dV/dQ (solid)"),
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
# For each cycle, print ICA peak positions (V) and DVA trough positions (V)
# side by side — ICA listed first to match the run order and literature
# comparison workflow.
# capacity_to_voltage is defined once outside the loop to avoid scoping issues.
# =============================================================================

def capacity_to_voltage(dva_sub, cap_mah, capacity_col, voltage_col):
    """Return the nearest voltage for a given capacity in a cycle's discharge curve."""
    idx = (dva_sub[capacity_col] - cap_mah).abs().idxmin()
    return dva_sub.loc[idx, voltage_col]


print(f"\n{'='*60}")
print("ICA / DVA feature cross-reference — voltage positions by cycle")
print(f"{'='*60}")
print(f"  ICA peaks mark concurrent lithiation equilibria.")
print(f"  DVA troughs mark transitions between lithiation equilibria.")
print(f"  Expected: one ICA peak bracketed between each pair of DVA troughs.\n")

all_cycles = sorted(ica_features_df[cycle_col].unique())
for cyc in all_cycles:
    # ICA peak voltage positions (directly on voltage axis)
    ica_cyc      = ica_features_df[ica_features_df[cycle_col] == cyc]
    ica_voltages = sorted(ica_cyc["x_position"].tolist(), reverse=True)

    # DVA trough capacity positions — map to voltage via discharge curve
    dva_cyc  = dva_features_df[dva_features_df[cycle_col] == cyc]
    dva_sub  = dva_df[dva_df[cycle_col] == cyc].dropna(
                   subset=[capacity_col, voltage_col])
    dva_voltages = sorted(
        [capacity_to_voltage(dva_sub, row["x_position"], capacity_col, voltage_col)
         for _, row in dva_cyc.iterrows()],
        reverse=True
    )

    print(f"  Cycle {int(cyc):>4}:")
    print(f"    ICA peaks   (V): "
          + "  ".join(f"{v:.4f}" for v in ica_voltages))
    print(f"    DVA troughs (V): "
          + "  ".join(f"{v:.4f}" for v in dva_voltages))
    n_ica  = len(ica_voltages)
    n_dva  = len(dva_voltages)
    status = ("OK" if n_ica == n_dva + 1
              else f"WARNING: expected {n_dva+1} ICA peaks, got {n_ica}")
    print(f"    n+1 check: {n_ica} ICA peaks, {n_dva} DVA troughs — {status}\n")