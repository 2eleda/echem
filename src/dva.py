# dva.py
from differential_functions import (apply_smoothing, compute_dvdq,
                                     plot_cycles_overlay, ensure_uniform_grid)
from feature_functions import find_features, track_features
from config import (CSV_PATH, CYCLE_COL, CAPACITY_COL, VOLTAGE_COL,
                    DVA_GRID_STEP, UNIFORMITY_THRESHOLD,
                    DQ_THRESHOLD, DVA_MAX_FEATURE_SHIFT, DVA_YLIM,
                    IS_DISCHARGE)
import pandas as pd
import os

# =============================================================================
# DVA PIPELINE
# Order of operations per cycle:
#   1. ensure_uniform_grid  — resample capacity axis if non-uniform
#   2. smooth voltage        — SG or spline on voltage vs. capacity
#   3. compute dV/dQ         — derivative on uniform grid
#   4. smooth dV/dQ          — SG or spline on derivative
#   5. find_features         — interactive trough detection
#   6. track_features        — cross-cycle identity matching
# =============================================================================

def run_dva(csv_path=CSV_PATH,
            resume_csv=None,
            cycle_col=CYCLE_COL,
            capacity_col=CAPACITY_COL,
            voltage_col=VOLTAGE_COL,
            dva_grid_step=DVA_GRID_STEP,
            uniformity_threshold=UNIFORMITY_THRESHOLD,
            dq_threshold=DQ_THRESHOLD,
            max_feature_shift=DVA_MAX_FEATURE_SHIFT,
            dva_ylim=DVA_YLIM,
            is_discharge=IS_DISCHARGE,
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
            raise FileNotFoundError(f"resume_csv not found: {resume_csv}")
        df = pd.read_csv(resume_csv)
        print(f"Loaded previously processed DVA dataframe: {resume_csv}")
        for col in ["Voltage_smooth", "dVdQ_raw", "dVdQ_smooth"]:
            if col not in df.columns:
                raise ValueError(
                    f"Expected column '{col}' not found in {resume_csv}.")
        print("  Smoothed columns confirmed — skipping to feature detection.")
        csv_out  = resume_csv
        dva_mode = "trough" if is_discharge else "peak"

    else:
        df = pd.read_csv(csv_path)
        print("DVA — Columns detected:", df.columns.tolist())
        print("DVA — Cycles present:  ",
              [int(c) for c in sorted(df[cycle_col].unique())])

        # DVA: discharge -> dV/dQ negative (troughs), negate=False
        #      charge    -> dV/dQ positive (peaks),   negate=False
        #      find_features mode flips to "peak" for charge
        dva_mode   = "trough" if is_discharge else "peak"
        # negate only needed if capacity convention is inverted
        # (e.g. counts down on discharge) — rare, left as False
        dva_negate = False

        # Step 1 — ensure uniform capacity grid before smoothing
        print(f"\n{'='*60}")
        print("Step 1 — Uniform grid check (capacity axis)")
        df, resampled, step_used = ensure_uniform_grid(
            df,
            x_col=capacity_col,
            y_col=voltage_col,
            cycle_col=cycle_col,
            step=dva_grid_step,
            uniformity_threshold=uniformity_threshold
        )

        # Step 2 — smooth voltage
        print(f"\n{'='*60}")
        print("Step 2 — Voltage smoothing")
        df, _ = apply_smoothing(
            df, voltage_col, cycle_col, out_col="Voltage_smooth",
            x_col=capacity_col, prompt_edge_trim=True
        )

        # Step 3 — compute raw dV/dQ
        if interactive:
            print(f"\n{'='*60}")
            print(f"Step 3 — dV/dQ computation")
            print(f"  dQ threshold: {dq_threshold}  "
                  f"(None = disabled; only needed for non-uniform grids)")
            if input("Edit dQ threshold? (yes/no): ").strip().lower() == "yes":
                raw = input("  New threshold in mAh (or 'none'): ").strip().lower()
                dq_threshold = None if raw == "none" else float(raw)

        df = compute_dvdq(
            df, capacity_col, "Voltage_smooth", cycle_col,
            out_col="dVdQ_raw", dq_threshold=dq_threshold,
            negate=dva_negate
        )

        # Step 4 — smooth dV/dQ
        print(f"\n{'='*60}")
        print("Step 4 — dV/dQ smoothing")
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

    # Step 5 — feature detection
    print(f"\n{'='*60}")
    print("Step 5 — DVA feature detection")
    feature_df = find_features(
        df, cycle_col,
        x_col=capacity_col,
        y_col="dVdQ_smooth",
        mode=dva_mode,
        reference_cycle=reference_cycle
    )

    feature_out = csv_out.replace(".csv", "_features.csv")
    feature_df.to_csv(feature_out, index=False)
    print(f"  Saved DVA feature summary: {feature_out}")

    # Step 6 — feature tracking
    features, tracked_df = track_features(
        feature_df, cycle_col,
        mode=dva_mode,
        max_shift=max_feature_shift
    )

    tracked_out = csv_out.replace(".csv", "_tracked.csv")
    tracked_df.to_csv(tracked_out, index=False)
    print(f"  Saved DVA tracked features: {tracked_out}")

    return (df, feature_df, tracked_df, features,
            cycle_col, capacity_col, voltage_col, csv_out)


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_dva()