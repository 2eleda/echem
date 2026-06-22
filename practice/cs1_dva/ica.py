from differential_functions import apply_smoothing, compute_dqdv, plot_cycles_overlay
import pandas as pd

# =============================================================================
# CONFIGURATION — edit here only
# =============================================================================
csv_path = "your_ica_data.csv"  # per-point V-Q discharge curves for selected cycles

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
# ICA-SPECIFIC PARAMETERS
# Threshold decisions owned here, not inside differential_functions.py
# =============================================================================

# dv_threshold: minimum |dV| step to compute dQ/dV
# At voltage plateaus dV approaches zero and without the threshold you get ±infinity spikes right at the ICA peaks
DV_THRESHOLD = None

# =============================================================================
# PIPELINE
# =============================================================================

# Step 1 — interactive voltage smoothing
# edge trimming applied here before fitting to exclude end-dropout artifacts
df, edge_trim_pct = apply_smoothing(
    df, voltage_col, cycle_col, out_col="Voltage_smooth",
    prompt_edge_trim=True
)

# Step 2 — compute raw dQ/dV for ICA
# Prompt user to confirm or adjust dv_threshold before differentiating
# dv_threshold actively matters here — at voltage plateaus dV approaches zero
# and without the threshold you get ±infinity spikes right at the ICA peaks
print(f"\n{'='*60}")
print(f"dQ/dV computation — threshold settings")
print(f"Current dV threshold: {DV_THRESHOLD} V")
print(f"  Points where |dV| < threshold are set to NaN to suppress plateau spikes.")
if input("Edit dV threshold? (yes/no): ").strip().lower() == "yes":
    try:
        DV_THRESHOLD = float(input(f"  New dV threshold (current {DV_THRESHOLD}): "))
    except ValueError:
        print("  Invalid — keeping current threshold.")

# Peaks in dQ/dV vs. voltage mark redox reactions:
#   LLI  → lateral shift of peak positions along voltage axis
#   LAM  → reduction in peak height or area
# Edge-trimmed NaNs in Voltage_smooth propagate naturally into dQdV_raw
df = compute_dqdv(
    df, capacity_col, "Voltage_smooth", cycle_col,
    out_col="dQdV_raw",
    dv_threshold=DV_THRESHOLD
)

# Step 3 — interactive dQ/dV smoothing
# edge trim not re-prompted — already applied in Step 1
df, _ = apply_smoothing(
    df, "dQdV_raw", cycle_col, out_col="dQdV_smooth",
    prompt_edge_trim=False
)

print("\nFinal dataframe columns:", df.columns.tolist())

# =============================================================================
# OUTPUT — user-named CSV and ICA overlay plot
# =============================================================================
# Name files first, then the plot displays — closing it triggers the save.

print(f"\n{'='*60}")
print("Output file naming")
csv_out  = input("  Enter filename for processed dataframe CSV (e.g. ica_processed.csv): ").strip()
plot_out = input("  Enter filename for ICA overlay plot (e.g. ica_overlay.png): ").strip()

if not csv_out.endswith(".csv"):
    csv_out += ".csv"
if not plot_out.endswith(".png"):
    plot_out += ".png"

df.to_csv(csv_out, index=False)
print(f"  Saved dataframe: {csv_out}")

# =============================================================================
# ICA OVERLAY PLOT
# =============================================================================
# Displays first (plt.show blocks until window is closed), then saves.
# ICA plots dQ/dV vs. voltage — x-axis is voltage, not capacity.
# ylim: None lets matplotlib autoscale since dQ/dV peaks are the signal of
# interest and we don't want to clip them. Adjust if needed.
ICA_YLIM = None

plot_cycles_overlay(
    df, cycle_col,
    x_col=voltage_col,
    y_col="dQdV_smooth",
    xlabel=voltage_col,
    ylabel="dQ/dV (mAh/V)",
    title="ICA — dQ/dV vs. Voltage by Cycle",
    save_path=plot_out,
    ylim=ICA_YLIM
)
print(f"  Saved plot:      {plot_out}")