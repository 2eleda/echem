from differential_functions import apply_smoothing, compute_dvdq, plot_cycles_overlay
import pandas as pd

# ==========================================splin===================================
# CONFIGURATION — edit here only
# =============================================================================
csv_path = "checkup_discharge_curves.csv"  # per-point V-Q discharge curves for selected cycles

# Column positions (0-indexed)
# Expected order: cycle number, discharge capacity, voltage
# =============================================================================

df = pd.read_csv(csv_path)

cycle_col    = df.columns[0]
capacity_col = df.columns[1]
voltage_col  = df.columns[2]

print("Columns detected:", df.columns.tolist())
print("Cycles present:  ", [int(c) for c in sorted(df[cycle_col].unique())])

# =============================================================================
# DVA-SPECIFIC PARAMETERS
# Threshold decisions owned here, not inside differential_functions.py
# =============================================================================

# dq_threshold: minimum |dQ| step to compute dV/dQ
# At 0.5 mAh/pt grid spacing, 0.01 mAh clips only truly anomalous near-zero steps
DQ_THRESHOLD = 0.01

# =============================================================================
# PIPELINE
# =============================================================================

# Step 1 — interactive voltage smoothing
# edge trimming applied here before fitting to exclude end-dropout artifacts
df, edge_trim_pct = apply_smoothing(
    df, voltage_col, cycle_col, out_col="Voltage_smooth",
    prompt_edge_trim=True
)

# Step 2 — compute raw dV/dQ for DVA
# Prompt user to confirm or adjust dq_threshold before differentiating
print(f"\n{'='*60}")
print(f"dV/dQ computation — threshold settings")
print(f"Current dQ threshold: {DQ_THRESHOLD} mAh")
print(f"  Points where |dQ| < threshold are set to NaN to suppress division artifacts.")
if input("Edit dQ threshold? (yes/no): ").strip().lower() == "yes":
    try:
        DQ_THRESHOLD = float(input(f"  New dQ threshold (current {DQ_THRESHOLD}): "))
    except ValueError:
        print("  Invalid — keeping current threshold.")

# Troughs in dV/dQ vs. capacity mark redox reactions:
#   LLI  → lateral shift of trough positions along capacity axis
#   LAM  → reduction in trough depth or collapse
# Edge-trimmed NaNs in Voltage_smooth propagate naturally into dVdQ_raw
df = compute_dvdq(
    df, capacity_col, "Voltage_smooth", cycle_col,
    out_col="dVdQ_raw",
    dq_threshold=DQ_THRESHOLD
)

# Step 3 — interactive dV/dQ smoothing
# edge trim not re-prompted — already applied in Step 1
df, _ = apply_smoothing(
    df, "dVdQ_raw", cycle_col, out_col="dVdQ_smooth",
    prompt_edge_trim=False
)

print("\nFinal dataframe columns:", df.columns.tolist())

# =============================================================================
# OUTPUT — user-named CSV and DVA overlay plot
# =============================================================================
# Name files first, then the plot displays — closing it triggers the save.

print(f"\n{'='*60}")
print("Output file naming")
csv_out  = input("  Enter filename for processed dataframe CSV (e.g. dva_processed.csv): ").strip()
plot_out = input("  Enter filename for DVA overlay plot (e.g. dva_overlay.png): ").strip()

if not csv_out.endswith(".csv"):
    csv_out += ".csv"
if not plot_out.endswith(".png"):
    plot_out += ".png"

df.to_csv(csv_out, index=False)
print(f"  Saved dataframe: {csv_out}")

# =============================================================================
# DVA OVERLAY PLOT
# =============================================================================
# Displays first (plt.show blocks until window is closed), then saves.
# ylim clips the y-axis so steep end-of-discharge features don't compress the
# bulk of the plot. Adjust or set to None to show the full scale.
DVA_YLIM = (-0.0007, 0)

plot_cycles_overlay(
    df, cycle_col,
    x_col=capacity_col,
    y_col="dVdQ_smooth",
    xlabel=capacity_col,
    ylabel="dV/dQ (V/mAh)",
    title="DVA — dV/dQ vs. Capacity by Cycle",
    save_path=plot_out,
    ylim=DVA_YLIM
)
print(f"  Saved plot:      {plot_out}")