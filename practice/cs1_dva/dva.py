# dva_analysis.py
from differential_functions import (apply_smoothing, compute_dvdq,
                                     plot_cycles_overlay, find_features,
                                     track_features)
import pandas as pd
import os

# =============================================================================
# CONFIGURATION — edit here only
# =============================================================================
DVA_CSV_PATH      = "your_dva_data.csv"
RESUME_CSV        = None        # path to previously saved processed CSV, or None
DQ_THRESHOLD      = None        # None for uniform grids; set for GITT/raw data
MAX_FEATURE_SHIFT = 300         # mAh
DVA_YLIM          = None        # e.g. (-0.001, 0); None = autoscale

# =============================================================================

def run_dva(csv_path=DVA_CSV_PATH,
            resume_csv=RESUME_CSV,
            dq_threshold=DQ_THRESHOLD,
            max_feature_shift=MAX_FEATURE_SHIFT,
            dva_ylim=DVA_YLIM,
            reference_cycle=None,
            interactive=True):
    """
    Full DVA pipeline. Returns (df, feature_df, tracked_df, features,
    cycle_col, capacity_col, voltage_col, csv_out).

    interactive=True  : runs all prompts as normal (standalone mode)
    interactive=False : skips prompts, uses parameter defaults (combined mode)
    """
    if resume_csv is not None:
        if not os.path.exists(resume_csv):
            raise FileNotFoundError(f"RESUME_CSV not found: {resume_csv}")
        df = pd.read_csv(resume_csv)
        print(f"Loaded previously processed DVA dataframe: {resume_csv}")
        cycle_col    = df.columns[0]
        capacity_col = df.columns[1]
        voltage_col  = df.columns[2]
        for col in ["Voltage_smooth", "dVdQ_raw", "dVdQ_smooth"]:
            if col not in df.columns:
                raise ValueError(
                    f"Expected column '{col}' not found in {resume_csv}.")
        print("  Smoothed columns confirmed — skipping to feature detection.")
        csv_out = resume_csv
    else:
        df = pd.read_csv(csv_path)
        cycle_col    = df.columns[0]
        capacity_col = df.columns[1]
        voltage_col  = df.columns[2]
        print("DVA — Columns detected:", df.columns.tolist())
        print("DVA — Cycles present:  ",
              [int(c) for c in sorted(df[cycle_col].unique())])

        # Step 1 — voltage smoothing
        df, _ = apply_smoothing(
            df, voltage_col, cycle_col, out_col="Voltage_smooth",
            x_col=capacity_col, prompt_edge_trim=True
        )

        # Step 2 — compute raw dV/dQ
        if interactive:
            print(f"\n{'='*60}")
            print(f"dV/dQ computation — dQ threshold: {dq_threshold}")
            print(f"  None = no threshold (uniform grids).")
            if input("Edit dQ threshold? (yes/no): ").strip().lower() == "yes":
                raw = input("  New threshold in mAh (or 'none'): ").strip().lower()
                dq_threshold = None if raw == "none" else float(raw)

        df = compute_dvdq(
            df, capacity_col, "Voltage_smooth", cycle_col,
            out_col="dVdQ_raw", dq_threshold=dq_threshold
        )

        # Step 3 — dV/dQ smoothing
        df, _ = apply_smoothing(
            df, "dVdQ_raw", cycle_col, out_col="dVdQ_smooth",
            x_col=capacity_col, prompt_edge_trim=False
        )

        # Output naming
        if interactive:
            print(f"\n{'='*60}")
            csv_out  = input("  DVA processed CSV filename: ").strip()
            plot_out = input("  DVA overlay plot filename:  ").strip()
            if not csv_out.endswith(".csv"):  csv_out  += ".csv"
            if not plot_out.endswith(".png"): plot_out += ".png"
        else:
            base     = os.path.splitext(os.path.basename(csv_path))[0]
            csv_out  = f"{base}_dva_processed.csv"
            plot_out = f"{base}_dva_overlay.png"

        df.to_csv(csv_out, index=False)
        print(f"  Saved DVA dataframe: {csv_out}")

        plot_cycles_overlay(
            df, cycle_col,
            x_col=capacity_col,
            y_col="dVdQ_smooth",
            xlabel=capacity_col,
            ylabel="dV/dQ (V/mAh)",
            title="DVA — dV/dQ vs. Capacity by Cycle",
            save_path=plot_out,
            ylim=dva_ylim
        )
        print(f"  Saved DVA plot: {plot_out}")

    # Feature detection
    feature_df = find_features(
        df, cycle_col,
        x_col=capacity_col,
        y_col="dVdQ_smooth",
        mode="trough",
        reference_cycle=reference_cycle
    )

    feature_out = csv_out.replace(".csv", "_features.csv")
    feature_df.to_csv(feature_out, index=False)
    print(f"  Saved DVA feature summary: {feature_out}")

    features, tracked_df = track_features(
        feature_df, cycle_col,
        mode="trough",
        max_shift=max_feature_shift
    )

    tracked_out = csv_out.replace(".csv", "_tracked.csv")
    tracked_df.to_csv(tracked_out, index=False)
    print(f"  Saved DVA tracked features: {tracked_out}")

    return df, feature_df, tracked_df, features, \
           cycle_col, capacity_col, voltage_col, csv_out


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_dva()