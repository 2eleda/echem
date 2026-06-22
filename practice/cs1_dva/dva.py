from differential_functions import (apply_smoothing, compute_dvdq,
                                     plot_cycles_overlay, find_features,
                                     track_features, DifferentialFeature)
import pandas as pd
import os

# =============================================================================
# CONFIGURATION — edit here only
# =============================================================================
csv_path = "checkup_discharge_curves.csv"  # per-point V-Q discharge curves for selected cycles

# Column positions (0-indexed)
# Expected order: cycle number, discharge capacity, voltage
# =============================================================================

# =============================================================================
# RESUME FROM PREVIOUS RUN?
# If you have already run the smoothing pipeline and saved the processed
# dataframe, you can skip straight to find_features by loading it here.
# Set RESUME_CSV to the path of your previously saved processed dataframe,
# or leave as None to run the full pipeline from raw data.
RESUME_CSV = "NMC532_dva_processed.csv"   # e.g. "dva_processed.csv"
# =============================================================================

# =============================================================================
# DVA-SPECIFIC PARAMETERS — set for your dataset
# =============================================================================
DQ_THRESHOLD      = None   # mAh — None for uniform grids; set for GITT/raw data
MAX_FEATURE_SHIFT = 300    # mAh — max cross-cycle feature shift for tracking
DVA_YLIM          = None   # e.g. (-0.001, 0); None = autoscale

# =============================================================================
# PIPELINE
# =============================================================================

if RESUME_CSV is not None:
    # --- Resume path: load previously processed dataframe ---
    if not os.path.exists(RESUME_CSV):
        raise FileNotFoundError(f"RESUME_CSV not found: {RESUME_CSV}")
    df = pd.read_csv(RESUME_CSV)
    print(f"Loaded previously processed dataframe: {RESUME_CSV}")
    print("Columns detected:", df.columns.tolist())

    # infer column names from loaded dataframe
    cycle_col    = df.columns[0]
    capacity_col = df.columns[1]
    voltage_col  = df.columns[2]

    # confirm expected smoothed columns are present
    for col in ["Voltage_smooth", "dVdQ_raw", "dVdQ_smooth"]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' not found in {RESUME_CSV}. "
                f"Available: {df.columns.tolist()}"
            )
    print("  Smoothed columns confirmed — skipping to feature detection.")
    csv_out = RESUME_CSV   # reuse same base name for output files

else:
    # --- Full pipeline: run from raw data ---
    df = pd.read_csv(csv_path)

    cycle_col    = df.columns[0]
    capacity_col = df.columns[1]
    voltage_col  = df.columns[2]

    print("Columns detected:", df.columns.tolist())
    print("Cycles present:  ", [int(c) for c in sorted(df[cycle_col].unique())])

    # Step 1 — interactive voltage smoothing
    df, edge_trim_pct = apply_smoothing(
        df, voltage_col, cycle_col, out_col="Voltage_smooth",
        prompt_edge_trim=True
    )

    # Step 2 — compute raw dV/dQ
    print(f"\n{'='*60}")
    print(f"dV/dQ computation — threshold settings")
    print(f"Current dQ threshold: {DQ_THRESHOLD}")
    print(f"  None = no threshold (appropriate for uniform capacity grids).")
    print(f"  Set a value if your data has variable dQ steps (GITT, raw cycler).")
    if input("Edit dQ threshold? (yes/no): ").strip().lower() == "yes":
        raw = input("  New dQ threshold in mAh (or 'none' to disable): ").strip().lower()
        if raw == "none":
            DQ_THRESHOLD = None
        else:
            try:
                DQ_THRESHOLD = float(raw)
            except ValueError:
                print("  Invalid — keeping current threshold.")

    # Troughs in dV/dQ vs. capacity mark redox reactions:
    #   LLI  → lateral shift of trough positions along capacity axis
    #   LAM  → reduction in trough depth or area
    df = compute_dvdq(
        df, capacity_col, "Voltage_smooth", cycle_col,
        out_col="dVdQ_raw",
        dq_threshold=DQ_THRESHOLD
    )

    # Step 3 — interactive dV/dQ smoothing
    df, _ = apply_smoothing(
        df, "dVdQ_raw", cycle_col, out_col="dVdQ_smooth",
        prompt_edge_trim=False
    )

    print("\nFinal dataframe columns:", df.columns.tolist())

    # =============================================================================
    # OUTPUT — save processed dataframe and overlay plot
    # =============================================================================
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

# =============================================================================
# TROUGH DETECTION AND TRACKING
# =============================================================================

# Uncomment to debug critical point placement on a specific cycle:
# from differential_functions import debug_critical_points
# debug_critical_points(df, cycle_col, capacity_col, "dVdQ_smooth",
#     mode="trough", cycle=0, prominence_threshold=2e-4)

feature_df = find_features(
    df, cycle_col,
    x_col=capacity_col,
    y_col="dVdQ_smooth",
    mode="trough",
    reference_cycle=None
)

print("\nFeature summary (first 10 rows):")
print(feature_df.head(10).to_string(index=False))

feature_out = csv_out.replace(".csv", "_features.csv")
feature_df.to_csv(feature_out, index=False)
print(f"  Saved feature summary: {feature_out}")

features, tracked_df = track_features(
    feature_df, cycle_col,
    mode="trough",
    max_shift=MAX_FEATURE_SHIFT
)

tracked_out = csv_out.replace(".csv", "_tracked.csv")
tracked_df.to_csv(tracked_out, index=False)
print(f"  Saved tracked features: {tracked_out}")