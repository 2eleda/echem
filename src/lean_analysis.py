# lean.py
import numpy as np
import pandas as pd
from differential_functions import plot_cycles_overlay
from feature_functions import find_features, track_features
from config import (CSV_PATH, CYCLE_COL, CAPACITY_COL, VOLTAGE_COL,
                    CURRENT_COL, TIME_COL, LEAN_DQ_STEP, LEAN_BIN_WIDTH,
                    IS_DISCHARGE)

# =============================================================================
# LEAN COMPUTATION
# =============================================================================

def compute_lean(df, voltage_col, cycle_col, out_col,
                 bin_width,
                 current_col=None, time_col=None, dq_step=None):
    """
    Compute LEAN (Level Evaluation ANalysis) dQ/dV per cycle group.

    LEAN avoids differentiation instability by binning voltage samples into
    histogram bins of width ΔV and counting samples per bin. This sidesteps
    near-zero denominator issues at voltage plateaus entirely.

    Formula: dQ/dVᵢ = (I₀ · dt / ΔV) · Nᵢ
    where Nᵢ is the number of samples in bin i.

    For uniform capacity grids (e.g. 0.5 mAh/pt resampled data), I₀ · dt per
    point equals the fixed capacity step, simplifying to:
        dQ/dVᵢ = (dq_step / ΔV) · Nᵢ

    Parameters
    ----------
    df          : dataframe with cycle_col and voltage_col
    voltage_col : raw (unsmoothed) voltage column — LEAN operates directly on
                  voltage samples, no pre-smoothing needed
    cycle_col   : column identifying cycle number
    out_col     : name for the output dQ/dV column
    bin_width   : ΔV bin width in volts. Choose so ΔV ≥ voltage noise scale.
                  For ~0.1 mV noise, ΔV = 0.001 V (1 mV) is a reasonable start.
                  Larger ΔV = smoother but lower resolution; see paper Fig. 4.
    current_col : optional — column name for current (A). If provided along
                  with time_col, I₀ · dt is computed per point from the data.
    time_col    : optional — column name for timestamp (s).
    dq_step     : fixed capacity increment per sample (mAh). Used when
                  current_col/time_col are absent. For 0.5 mAh/pt grid: 0.5.
                  Must provide either (current_col + time_col) or dq_step.

    Returns dataframe with two new columns:
        '{out_col}_v_centres' : voltage bin centres (V)
        '{out_col}'           : dQ/dV values (mAh/V) at each original index,
                                mapped from bin centres for easy overlay with
                                other differential methods
    """
    if current_col is None and dq_step is None:
        raise ValueError(
            "Must provide either (current_col + time_col) for I₀·dt computation, "
            "or dq_step as a fixed capacity increment per sample."
        )

    def _lean_group(group):
        v = group[voltage_col].values.astype(float)

        # compute charge increment per sample
        if current_col is not None and time_col is not None:
            I  = np.abs(group[current_col].values.astype(float))
            dt = np.diff(group[time_col].values.astype(float))
            # I₀·dt gives mAh per interval; prepend 0 to match length
            dq_per_sample = np.concatenate([[0], I[:-1] * dt * 1000 / 3600])
        else:
            dq_per_sample = np.full(len(v), dq_step)

        # build histogram bin edges aligned to digital resolution
        v_min = np.nanmin(v)
        v_max = np.nanmax(v)
        edges = np.arange(
            np.floor(v_min / bin_width) * bin_width,
            np.ceil(v_max  / bin_width) * bin_width + bin_width,
            bin_width
        )
        centres = (edges[:-1] + edges[1:]) / 2

        # count samples and total charge per bin
        bin_idx   = np.digitize(v, edges) - 1
        bin_idx   = np.clip(bin_idx, 0, len(centres) - 1)
        n_bins    = len(centres)

        # vectorised: sum charge per bin and mark empty bins as NaN
        bin_sums   = np.bincount(bin_idx, weights=dq_per_sample, minlength=n_bins).astype(float)
        bin_counts = np.bincount(bin_idx, minlength=n_bins)
        dqdv_bins  = np.where(bin_counts > 0, bin_sums / bin_width, np.nan)

        # map bin dQ/dV back to each original sample point for dataframe alignment
        result = dqdv_bins[bin_idx]
        return pd.Series(result, index=group.index)

    df[out_col] = df.groupby(cycle_col, group_keys=False).apply(_lean_group).values
    return df

# =============================================================================
# LEAN PIPELINE
# LEAN (Level Evaluation ANalysis) computes dQ/dV by binning voltage samples
# into histogram bins of width ΔV. This avoids differentiation instability
# entirely — no pre-smoothing or grid resampling needed.
# =============================================================================

def run_lean(csv_path=CSV_PATH,
             cycle_col=CYCLE_COL,
             capacity_col=CAPACITY_COL,
             voltage_col=VOLTAGE_COL,
             current_col=CURRENT_COL,
             time_col=TIME_COL,
             dq_step=LEAN_DQ_STEP,
             bin_width=LEAN_BIN_WIDTH,
             is_discharge=IS_DISCHARGE,
             interactive=True):
    """
    Full LEAN pipeline. Returns (df, csv_out).
    """
    df = pd.read_csv(csv_path)
    print("LEAN — Columns detected:", df.columns.tolist())
    print("LEAN — Cycles present:  ",
          [int(c) for c in sorted(df[cycle_col].unique())])

    # Parameter prompt
    if interactive:
        print(f"\n{'='*60}")
        print(f"LEAN computation — bin width settings")
        print(f"  Current bin width: {bin_width} V ({bin_width*1000:.1f} mV)")
        print(f"  Must be >= voltage noise scale.")
        print(f"  Larger bin = smoother output, lower voltage resolution.")
        print(f"  Rule of thumb: >= 5 samples per peak half-width.")
        if input("Edit bin width? (yes/no): ").strip().lower() == "yes":
            try:
                bin_width = float(
                    input(f"  New bin width in V (current {bin_width}): "))
            except ValueError:
                print("  Invalid — keeping current bin width.")

        if current_col is None:
            print(f"\n  No current/time columns set — using fixed dQ step: "
                  f"{dq_step} mAh/pt")
            if input("Edit dQ step? (yes/no): ").strip().lower() == "yes":
                try:
                    dq_step = float(
                        input(f"  New dQ step in mAh (current {dq_step}): "))
                except ValueError:
                    print("  Invalid — keeping current dQ step.")

    print(f"\nComputing LEAN dQ/dV: bin_width={bin_width} V, "
          f"dq_step={dq_step} mAh...")
    df = compute_lean(
        df, voltage_col, cycle_col,
        out_col="dQdV_lean",
        bin_width=bin_width,
        current_col=current_col,
        time_col=time_col,
        dq_step=dq_step
    )
    print("LEAN computation complete.")

    # Output naming
    if interactive:
        print(f"\n{'='*60}")
        csv_out  = input("  LEAN processed CSV filename: ").strip()
        plot_out = input("  LEAN overlay plot filename:  ").strip()
        if not csv_out.endswith(".csv"):  csv_out  += ".csv"
        if not plot_out.endswith(".png"): plot_out += ".png"
    else:
        import os
        base     = os.path.splitext(os.path.basename(csv_path))[0]
        csv_out  = f"{base}_lean_processed.csv"
        plot_out = f"{base}_lean_overlay.png"

    df.to_csv(csv_out, index=False)
    print(f"  Saved dataframe: {csv_out}")

    plot_cycles_overlay(
        df, cycle_col,
        x_col=voltage_col,
        y_col="dQdV_lean",
        xlabel=voltage_col,
        ylabel="dQ/dV — LEAN (mAh/V)",
        title="LEAN — dQ/dV vs. Voltage by Cycle",
        save_path=plot_out,
        ylim=None
    )
    print(f"  Saved plot: {plot_out}")

    return df, csv_out


# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_lean()