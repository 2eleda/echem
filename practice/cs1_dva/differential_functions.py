import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
import math

# =============================================================================
# DEFAULT SMOOTHING PARAMETERS
# =============================================================================
SAVGOL_WINDOW    = 21
SAVGOL_POLYORDER = 3
SPLINE_SMOOTHING = None   # None = scipy auto; increase to smooth more
EDGE_TRIM_PCT    = 0.0    # fraction of points to trim each end before smoothing fit


# =============================================================================
# SMOOTHING BACKENDS (generic — no derivative knowledge)
# =============================================================================

def _contiguous_valid_interior(vals):
    """
    Find the indices of the first and last non-NaN values in an array.
    Returns (first_valid, last_valid+1) as a slice, or None if no valid data.
    Used to strip leading/trailing NaNs before smoothing so the filter never
    sees a NaN boundary, eliminating edge distortion artifacts.
    """
    valid_mask = ~np.isnan(vals)
    if not valid_mask.any():
        return None
    first = np.argmax(valid_mask)
    last  = len(vals) - np.argmax(valid_mask[::-1])
    return first, last


def _sg_smooth(vals, window, polyorder):
    """
    Savitzky-Golay on a 1D array.
    Strips leading/trailing NaNs before filtering so the SG window never hits
    a NaN boundary — eliminates Gibbs-like edge distortion on trimmed regions.
    Interior NaNs (rare) are handled via valid_mask within the interior slice.
    """
    bounds = _contiguous_valid_interior(vals)
    if bounds is None:
        return vals.copy()
    first, last = bounds

    interior   = vals[first:last]
    valid_mask = ~np.isnan(interior)
    n_valid    = valid_mask.sum()
    if n_valid <= polyorder + 1:
        return vals.copy()

    window_actual      = min(window, n_valid if n_valid % 2 == 1 else n_valid - 1)
    smoothed_interior  = interior.copy()
    smoothed_interior[valid_mask] = savgol_filter(
        interior[valid_mask], window_length=window_actual, polyorder=polyorder
    )

    result = vals.copy()
    result[first:last] = smoothed_interior
    return result


def _spline_smooth(x_vals, y_vals, smoothing_factor):
    """
    Smoothing spline on a 1D array.
    Strips leading/trailing NaNs before fitting so the spline boundary
    is the actual data boundary, not a NaN wall.
    x_vals: independent axis (e.g. capacity), must be strictly increasing.
    smoothing_factor: passed to UnivariateSpline s=. None = scipy auto.
    """
    bounds = _contiguous_valid_interior(y_vals)
    if bounds is None:
        return y_vals.copy()
    first, last = bounds

    x_interior = x_vals[first:last]
    y_interior = y_vals[first:last]
    valid_mask  = ~np.isnan(y_interior)

    if valid_mask.sum() < 4:
        return y_vals.copy()

    x_clean = x_interior[valid_mask]
    y_clean = y_interior[valid_mask]
    _, unique_idx = np.unique(x_clean, return_index=True)
    x_u = x_clean[unique_idx]
    y_u = y_clean[unique_idx]
    spline = UnivariateSpline(x_u, y_u, s=smoothing_factor, k=3)

    result = y_vals.copy()
    result[first:last][valid_mask] = spline(x_clean)
    return result


def _apply_smoothing_to_df(df, target_col, x_col, cycle_col, out_col,
                            method, window, polyorder, spline_s, edge_trim_pct):
    """
    Apply chosen smoothing method per cycle group, trimming edges before fitting.
    Trimmed points are set to NaN so the dataframe shape is preserved.
    Edge trimming prevents end-dropout artifacts from distorting the fit.
    Generic — no knowledge of which derivative will follow.
    """
    def _smooth_group(group):
        n    = len(group)
        trim = max(0, int(np.floor(n * edge_trim_pct)))
        inner = group.iloc[trim: n - trim]
        vals  = inner[target_col].values.astype(float)
        if method == "spline":
            x_vals         = inner[x_col].values.astype(float)
            smoothed_inner = _spline_smooth(x_vals, vals, spline_s)
        else:
            smoothed_inner = _sg_smooth(vals, window, polyorder)
        result = np.full(n, np.nan)
        result[trim: n - trim] = smoothed_inner
        return pd.Series(result, index=group.index)

    df[out_col] = df.groupby(cycle_col, group_keys=False).apply(_smooth_group).values
    return df


# =============================================================================
# DERIVATIVE COMPUTATION
# Both functions are generic — thresholds and column naming decided by caller.
# =============================================================================

def compute_dvdq(df, capacity_col, smoothed_voltage_col, cycle_col,
                 out_col, dq_threshold):
    """
    Compute raw dV/dQ per cycle group. Used for DVA.
    Troughs in dV/dQ vs. capacity mark phase transitions:
      LLI → lateral shift of trough positions along capacity axis
      LAM → reduction in trough depth or collapse
    dq_threshold: caller decides appropriate value based on capacity grid spacing.
    Edge-trimmed NaNs in smoothed_voltage_col propagate naturally.
    """
    def _dvdq_group(group):
        v  = group[smoothed_voltage_col].values.astype(float)
        q  = group[capacity_col].values.astype(float)
        dv = np.diff(v)
        dq = np.diff(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            dvdq = np.where(np.abs(dq) > dq_threshold, dv / dq, np.nan)
        return pd.Series(np.concatenate([[np.nan], dvdq]), index=group.index)

    df[out_col] = df.groupby(cycle_col, group_keys=False).apply(_dvdq_group).values
    return df


def compute_dqdv(df, capacity_col, smoothed_voltage_col, cycle_col,
                 out_col, dv_threshold):
    """
    Compute raw dQ/dV per cycle group. Used for ICA.
    Peaks in dQ/dV vs. voltage mark phase transitions.
    dv_threshold: caller decides appropriate value based on voltage resolution.
    Edge-trimmed NaNs in smoothed_voltage_col propagate naturally.
    """
    def _dqdv_group(group):
        v  = group[smoothed_voltage_col].values.astype(float)
        q  = group[capacity_col].values.astype(float)
        dv = np.diff(v)
        dq = np.diff(q)
        with np.errstate(divide="ignore", invalid="ignore"):
            dqdv = np.where(np.abs(dv) > dv_threshold, dq / dv, np.nan)
        return pd.Series(np.concatenate([[np.nan], dqdv]), index=group.index)

    df[out_col] = df.groupby(cycle_col, group_keys=False).apply(_dqdv_group).values
    return df


# =============================================================================
# INTERACTIVE INSPECT LOOP (generic — works for any column pair)
# =============================================================================

def _inspect_cycle(df, cycle_col, x_col, raw_col, smooth_col,
                   ylabel, title_prefix, cyc, method, window, polyorder, spline_s):
    """Plot raw vs smoothed for a single cycle with zoom/save loop."""
    xlim = None
    ylim = None
    last_fig = None
    param_str = (f"window={window}, polyorder={polyorder}"
                 if method == "sg" else f"spline s={spline_s}")

    while True:
        sub = df[df[cycle_col] == cyc].dropna(subset=[raw_col])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(sub[x_col], sub[raw_col],    label=f"Raw {raw_col}",         alpha=0.4, color="gray")
        ax.plot(sub[x_col], sub[smooth_col], label=f"Smoothed {smooth_col}", color="tab:blue")
        ax.set_xlabel(x_col)
        ax.set_ylabel(ylabel)
        ax.set_title(f"Cycle {int(cyc)} — {title_prefix} | {param_str}")
        ax.legend()
        if xlim: ax.set_xlim(xlim)
        if ylim: ax.set_ylim(ylim)
        fig.tight_layout()
        last_fig = fig
        plt.show()

        zoom = input("  Zoom into a region? (yes/no): ").strip().lower()
        if zoom == "yes":
            try:
                x_min = float(input("    x min: "))
                x_max = float(input("    x max: "))
                y_min = float(input("    y min: "))
                y_max = float(input("    y max: "))
                xlim = (x_min, x_max)
                ylim = (y_min, y_max)
            except ValueError:
                print("    Invalid input — keeping current view.")
        else:
            if input("  Save this plot? (yes/no): ").strip().lower() == "yes":
                save_path = input("  Enter filename (e.g. cycle0_voltage.png): ").strip()
                last_fig.savefig(save_path, dpi=150, bbox_inches="tight")
                print(f"  Saved: {save_path}")
            break

    plt.close("all")


# =============================================================================
# INTERACTIVE SMOOTHING ROUTINE (generic — no derivative knowledge)
# =============================================================================

def apply_smoothing(df, target_col, cycle_col, out_col,
                    window=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER,
                    spline_s=SPLINE_SMOOTHING,
                    edge_trim_pct=EDGE_TRIM_PCT,
                    prompt_edge_trim=True):
    """
    Interactive smoothing routine supporting Savitzky-Golay or smoothing spline.
    Generic — no knowledge of which derivative follows or what thresholds mean.
    Threshold decisions (dv_threshold, dq_threshold) are handled by the calling
    analysis script (dva_analysis.py, ica_analysis.py), not here.

    prompt_edge_trim=False suppresses edge trim prompt for derivative smoothing
    passes where edges are already NaN from the voltage smoothing step.

    Returns (df, edge_trim_pct).
    """
    cycles      = sorted(df[cycle_col].unique())
    x_col       = df.columns[1]
    first_cycle = cycles[0]

    print(f"\n{'='*60}")
    print(f"Smoothing column: '{target_col}' → '{out_col}'")
    print("Choose smoothing method:")
    print("  sg     — Savitzky-Golay (local polynomial; good for uniform noise)")
    print("  spline — Smoothing spline (global fit; better for resampled grid data)")
    method = ""
    while method not in ("sg", "spline"):
        method = input("Enter method (sg / spline): ").strip().lower()

    while True:
        print(f"\n{'='*60}")
        print(f"Method: {method.upper()}")

        if method == "sg":
            print(f"Current settings:  window_length={window}, polyorder={polyorder}")
            if input("Edit SG settings? (yes/no): ").strip().lower() == "yes":
                try:
                    new_window = int(input(f"  New window_length (current {window}): "))
                    if new_window % 2 == 0:
                        new_window += 1
                        print(f"  Adjusted to odd: {new_window}")
                    new_poly = int(input(f"  New polyorder (current {polyorder}): "))
                    if new_poly >= new_window:
                        print(f"  polyorder must be < window_length — keeping {polyorder}")
                        new_poly = polyorder
                    window    = new_window
                    polyorder = new_poly
                except ValueError:
                    print("  Invalid — keeping current settings.")
        else:
            print(f"Current settings:  spline s={spline_s}  (None = scipy auto; larger = smoother)")
            if input("Edit spline settings? (yes/no): ").strip().lower() == "yes":
                raw = input(f"  New s (current {spline_s}, enter 'none' for auto): ").strip().lower()
                if raw == "none":
                    spline_s = None
                else:
                    try:
                        spline_s = float(raw)
                    except ValueError:
                        print("  Invalid — keeping current settings.")

        if prompt_edge_trim:
            print(f"\nCurrent edge trim: {edge_trim_pct*100:.1f}% of points from each end")
            print("  Excludes noisy voltage-cutoff turnaround from spline/SG fit entirely.")
            if input("Edit edge trim? (yes/no): ").strip().lower() == "yes":
                try:
                    pct = float(input(f"  New trim % (e.g. 1.5, current {edge_trim_pct*100:.1f}%): "))
                    edge_trim_pct = pct / 100.0
                except ValueError:
                    print("  Invalid — keeping current trim.")

        print(f"\nApplying {method.upper()} smoothing...")
        df = _apply_smoothing_to_df(df, target_col, x_col, cycle_col, out_col,
                                    method, window, polyorder, spline_s, edge_trim_pct)

        print(f"\nDisplaying cycle {int(first_cycle)} first...")
        _inspect_cycle(df, cycle_col, x_col,
                       raw_col=target_col, smooth_col=out_col,
                       ylabel=target_col, title_prefix=f"{out_col} Smoothing",
                       cyc=first_cycle, method=method,
                       window=window, polyorder=polyorder, spline_s=spline_s)

        while True:
            if input("\nInspect another cycle? (yes/no): ").strip().lower() != "yes":
                break
            print(f"  Available cycles: {[int(c) for c in cycles]}")
            try:
                cyc_input = int(input("  Enter cycle number: "))
                if cyc_input not in [int(c) for c in cycles]:
                    print(f"  Cycle {cyc_input} not found — skipping.")
                    continue
                matched = [c for c in cycles if int(c) == cyc_input][0]
                _inspect_cycle(df, cycle_col, x_col,
                               raw_col=target_col, smooth_col=out_col,
                               ylabel=target_col, title_prefix=f"{out_col} Smoothing",
                               cyc=matched, method=method,
                               window=window, polyorder=polyorder, spline_s=spline_s)
            except ValueError:
                print("  Invalid — skipping.")

        if input("\nSwitch smoothing method (sg ↔ spline)? (yes/no): ").strip().lower() == "yes":
            method = "spline" if method == "sg" else "sg"
            print(f"  Switched to: {method.upper()}")
            continue

        if input("Adjust parameters and redo smoothing? (yes/no): ").strip().lower() == "yes":
            continue

        param_str = (f"window={window}, polyorder={polyorder}"
                     if method == "sg" else f"spline s={spline_s}")
        confirm = input(
            f"Confirm {method.upper()} ({param_str}), "
            f"edge trim={edge_trim_pct*100:.1f}% — keep '{out_col}' column? (yes/no): "
        ).strip().lower()

        if confirm == "yes":
            print(f"  ✓ '{out_col}' confirmed.")
            break
        else:
            print("  Restarting...")

    return df, edge_trim_pct


# =============================================================================
# MULTI-CYCLE OVERLAY PLOT
# =============================================================================

def plot_cycles_overlay(df, cycle_col, x_col, y_col, xlabel, ylabel, title,
                        save_path=None, ylim=None, cmap="plasma"):
    """
    Plot y_col vs x_col for all cycles as separate series on one graph.
    Each cycle gets its own color drawn from a colormap so evolution across
    cycling is visually trackable (early cycles one end, late cycles the other).
    Generic — works for dV/dQ vs. capacity (DVA), dQ/dV vs. voltage (ICA),
    voltage curves, or any per-cycle x/y data.

    Parameters
    ----------
    df         : dataframe with cycle_col, x_col, y_col
    cycle_col  : column identifying cycle number
    x_col      : x-axis column (e.g. capacity_mAh)
    y_col      : y-axis column (e.g. dVdQ_smooth)
    xlabel     : x-axis label string
    ylabel     : y-axis label string
    title      : plot title string
    save_path  : optional filename — if None, displays only without saving
    ylim       : optional tuple (ymin, ymax) to clip y-axis scale
    cmap       : matplotlib colormap name (default 'plasma')
    """
    cycles  = sorted(df[cycle_col].unique())
    n       = len(cycles)
    colors  = plt.colormaps[cmap](np.linspace(0, 1, n))

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, cyc in enumerate(cycles):
        sub = df[df[cycle_col] == cyc].dropna(subset=[y_col])
        ax.plot(sub[x_col], sub[y_col],
                color=colors[i], linewidth=0.9,
                label=f"Cycle {int(cyc)}")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if ylim is not None:
        ax.set_ylim(ylim)

    sm = plt.cm.ScalarMappable(
        cmap=plt.colormaps[cmap],
        norm=plt.Normalize(vmin=int(cycles[0]), vmax=int(cycles[-1]))
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(cycle_col)

    fig.tight_layout()

    plt.show()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")


# =============================================================================
# BATCH PLOTTING (generic — non-interactive final review)
# =============================================================================

def plot_smoothing_comparison(df, cycle_col, x_col, raw_col, smooth_col,
                              ylabel, title_prefix, save_prefix=None,
                              cycles_per_page=6):
    """
    Batch plot raw vs. smoothed for all cycles, 6 per page in a 3x2 grid.
    Non-interactive — use for final review after smoothing is confirmed.
    Generic — works for voltage, dV/dQ, dQ/dV, or any column pair.
    """
    cycles = sorted(df[cycle_col].unique())
    pages  = [cycles[i:i + cycles_per_page] for i in range(0, len(cycles), cycles_per_page)]

    for page_num, page_cycles in enumerate(pages, start=1):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10), squeeze=False)
        axes = axes.flatten()

        for i, cyc in enumerate(page_cycles):
            ax  = axes[i]
            sub = df[df[cycle_col] == cyc].dropna(subset=[raw_col])
            ax.plot(sub[x_col], sub[raw_col],    label=f"Raw {raw_col}",         alpha=0.4, color="gray")
            ax.plot(sub[x_col], sub[smooth_col], label=f"Smoothed {smooth_col}", color="tab:blue")
            ax.set_xlabel(x_col)
            ax.set_ylabel(ylabel)
            ax.set_title(f"Cycle {int(cyc)} — {title_prefix}")
            ax.legend()

        for j in range(len(page_cycles), len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(f"{title_prefix} — Page {page_num} of {len(pages)}", fontsize=14, y=1.01)
        fig.tight_layout()

        if save_prefix:
            sp = f"{save_prefix}_page{page_num}.png"
            plt.savefig(sp, dpi=150, bbox_inches="tight")
            print(f"Saved: {sp}")

        plt.show()