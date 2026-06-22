from differential_functions import compute_lean, plot_cycles_overlay
import pandas as pd

# =============================================================================
# CONFIGURATION — edit here only
# =============================================================================
csv_path = "checkup_discharge_curves.csv"  # per-point V-Q discharge curves for selected cycles

# Column positions (0-indexed)
# Expected order: cycle number, discharge capacity, voltage
# If your CSV also has current and time columns, set their names below
# =============================================================================

df = pd.read_csv(csv_path)

cycle_col    = df.columns[0]
capacity_col = df.columns[1]
voltage_col  = df.columns[2]

# Optional: set to column name strings if present in your CSV, else leave None
CURRENT_COL = None   # e.g. "current_A"
TIME_COL    = None   # e.g. "time_s"

print("Columns detected:", df.columns.tolist())
print("Cycles present:  ", [int(c) for c in sorted(df[cycle_col].unique())])

# =============================================================================
# LEAN-SPECIFIC PARAMETERS
# =============================================================================

# dq_step: fixed capacity increment per sample (mAh)
# Only used if CURRENT_COL and TIME_COL are both None
DQ_STEP = 0.5

# bin_width: voltage bin width ΔV in volts
# Must be ≥ voltage noise scale (~0.1 mV = 0.0001 V for this dataset)
# Rule of thumb: ≥ 5 points across each peak half-width
# Start at ~5-10x noise floor and increase for smoother output
# Larger ΔV → smoother but lower voltage resolution
BIN_WIDTH = 0.001

# =============================================================================
# PARAMETER PROMPT
# =============================================================================

print(f"\n{'='*60}")
print(f"LEAN computation — bin width settings")
print(f"Current bin width: {BIN_WIDTH} V ({BIN_WIDTH*1000:.1f} mV)")
print(f"  Must be ≥ voltage noise scale).")
print(f"  Larger bin = smoother output, lower voltage resolution.")
print(f"  Rule of thumb: ≥ 5 samples per peak half-width.")
if input("Edit bin width? (yes/no): ").strip().lower() == "yes":
    try:
        BIN_WIDTH = float(input(f"  New bin width in V (current {BIN_WIDTH}): "))
    except ValueError:
        print("  Invalid — keeping current bin width.")

if CURRENT_COL is None:
    print(f"\nNo current/time columns set — using fixed dQ step: {DQ_STEP} mAh/pt")
    if input("Edit dQ step? (yes/no): ").strip().lower() == "yes":
        try:
            DQ_STEP = float(input(f"  New dQ step in mAh (current {DQ_STEP}): "))
        except ValueError:
            print("  Invalid — keeping current dQ step.")

# =============================================================================
# PIPELINE
# =============================================================================

# LEAN operates directly on raw voltage — no pre-smoothing needed.
# The binning absorbs noise structurally rather than filtering it out.
print(f"\nComputing LEAN dQ/dV: bin_width={BIN_WIDTH} V, dQ_step={DQ_STEP} mAh...")

df = compute_lean(
    df, voltage_col, cycle_col,
    out_col="dQdV_lean",
    bin_width=BIN_WIDTH,
    current_col=CURRENT_COL,
    time_col=TIME_COL,
    dq_step=DQ_STEP
)

print("LEAN computation complete.")
print("\nFinal dataframe columns:", df.columns.tolist())

# =============================================================================
# OUTPUT — user-named CSV and LEAN overlay plot
# =============================================================================

print(f"\n{'='*60}")
print("Output file naming")
csv_out  = input("  Enter filename for processed dataframe CSV (e.g. lean_processed.csv): ").strip()
plot_out = input("  Enter filename for LEAN overlay plot (e.g. lean_overlay.png): ").strip()

if not csv_out.endswith(".csv"):
    csv_out += ".csv"
if not plot_out.endswith(".png"):
    plot_out += ".png"

df.to_csv(csv_out, index=False)
print(f"  Saved dataframe: {csv_out}")

# =============================================================================
# LEAN OVERLAY PLOT
# =============================================================================
# LEAN produces dQ/dV — plotted against voltage like ICA.
# ylim: None to autoscale; adjust if end-of-discharge features dominate.
LEAN_YLIM = None

plot_cycles_overlay(
    df, cycle_col,
    x_col=voltage_col,
    y_col="dQdV_lean",
    xlabel=voltage_col,
    ylabel="dQ/dV — LEAN (mAh/V)",
    title="LEAN — dQ/dV vs. Voltage by Cycle",
    save_path=plot_out,
    ylim=LEAN_YLIM
)
print(f"  Saved plot:      {plot_out}")