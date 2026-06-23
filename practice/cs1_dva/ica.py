# ica_analysis.py
from differential_functions import (apply_smoothing, compute_dqdv,
                                     plot_cycles_overlay, find_features,
                                     track_features)
import pandas as pd
import os

# =============================================================================
# CONFIGURATION — edit here only
# =============================================================================
ICA_CSV_PATH      = "your_ica_data.csv"
RESUME_CSV        = None
DV_THRESHOLD      = 0.001       # V — guards against dV→0 blow-up at plateaus
MAX_FEATURE_SHIFT = 0.05        # V
ICA_YLIM          = None

# =============================================================================

def run_ica(csv_path=ICA_CSV_PATH,
            resume_csv=RESUME_CSV,
            dv_threshold=DV_THRESHOLD,
            max_feature_shift=MAX_FEATURE_SHIFT,
            ica_ylim=ICA_YLIM,
            reference_cycle=None,
            interactive=True):
    """
    Full ICA pipeline. Returns (df, feature_df, tracked_df, features,
    cycle_col, capacity_col, voltage_col, csv_out).
    """
    if resume_csv is not None:
        if not os.path.exists(resume_csv):
            raise FileNotFoundError(f"RESUME_CSV not found: {resume_csv}")
        df = pd.read_csv(resume_csv)
        print(f"Loaded previously processed ICA dataframe: {resume_csv}")
        cycle_col    = df.columns[0]
        capacity_col = df.columns[1]
        voltage_col  = df.columns[2]
        for col in ["Voltage_smooth", "dQdV_raw", "dQdV_smooth"]:
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
        print("ICA — Columns detected:", df.columns.tolist())
        print("ICA — Cycles present:  ",
              [int(c) for c in sorted(df[cycle_col].unique())])

        # Step 1 — voltage smoothing
        df, _ = apply_smoothing(
            df, voltage_col, cycle_col, out_col="Voltage_smooth",
            x_col=capacity_col, prompt_edge_trim=True
        )

        # Step 2 — compute raw dQ/dV
        if interactive:
            print(f"\n{'='*60}")
            print(f"dQ/dV computation — dV threshold: {dv_threshold} V")
            print(f"  Guards against dV→0 blow-up at voltage plateaus.")
            if input("Edit dV threshold? (yes/no): ").strip().lower() == "yes":
                raw = input("  New threshold in V (or 'none'): ").strip().lower()
                dv_threshold = None if raw == "none" else float(raw)

        df = compute_dqdv(
            df, voltage_col, capacity_col, cycle_col,
            out_col="dQdV_raw", dv_threshold=dv_threshold
        )

        # Step 3 — dQ/dV smoothing
        df, _ = apply_smoothing(
            df, "dQdV_raw", cycle_col, out_col="dQdV_smooth",
            x_col=voltage_col, prompt_edge_trim=False
        )

        # Output naming
        if interactive:
            print(f"\n{'='*60}")
            csv_out  = input("  ICA processed CSV filename: ").strip()
            plot_out = input("  ICA overlay plot filename:  ").strip()
            if not csv_out.endswith(".csv"):  csv_out  += ".csv"
            if not plot_out.endswith(".png"): plot_out += ".png"
        else:
            base     = os.path.splitext(os.path.basename(csv_path))[0]
            csv_out  = f"{base}_ica_processed.csv"
            plot_out = f"{base}_ica_overlay.png"

        df.to_csv(csv_out, index=False)
        print(f"  Saved ICA dataframe: {csv_out}")

        plot_cycles_overlay(
            df, cycle_col,
            x_col=voltage_col,
            y_col="dQdV_smooth",
            xlabel=voltage_col,
            ylabel="dQ/dV (mAh/V)",
            title="ICA — dQ/dV vs. Voltage by Cycle",
            save_path=plot_out,
            ylim=ica_ylim
        )
        print(f"  Saved ICA plot: {plot_out}")

    # Feature detection
    feature_df = find_features(
        df, cycle_col,
        x_col=voltage_col,
        y_col="dQdV_smooth",
        mode="peak",
        reference_cycle=reference_cycle
    )

    feature_out = csv_out.replace(".csv", "_features.csv")
    feature_df.to_csv(feature_out, index=False)
    print(f"  Saved ICA feature summary: {feature_out}")

    features, tracked_df = track_features(
        feature_df, cycle_col,
        mode="peak",
        max_shift=max_feature_shift
    )

    tracked_out = csv_out.replace(".csv", "_tracked.csv")
    tracked_df.to_csv(tracked_out, index=False)
    print(f"  Saved ICA tracked features: {tracked_out}")

    return df, feature_df, tracked_df, features, \
           cycle_col, capacity_col, voltage_col, csv_out


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_ica()