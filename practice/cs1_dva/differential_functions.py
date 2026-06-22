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

    interior = vals[first:last]
    valid_mask = ~np.isnan(interior)
    n_valid = valid_mask.sum()
    if n_valid <= polyorder + 1:
        return vals.copy()

    window_actual = min(window, n_valid if n_valid % 2 == 1 else n_valid - 1)
    smoothed_interior = interior.copy()
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
                 out_col, dq_threshold=None):
    """
    Compute raw dV/dQ per cycle group. Used for DVA.
    Troughs in dV/dQ vs. capacity mark redox reactions:
      LLI → lateral shift of trough positions along capacity axis
      LAM → reduction in trough depth or collapse
    Edge-trimmed NaNs in smoothed_voltage_col propagate naturally.

    dq_threshold: minimum |dQ| to compute derivative. Default None = no
                  threshold applied. Only needed for non-uniform capacity
                  grids (e.g. raw cycler data, GITT) where dQ can approach
                  zero between points and cause dV/dQ to blow up.
                  For uniform resampled grids (e.g. 0.5 mAh/pt), dQ is
                  fixed so no threshold is needed — small dV at plateaus
                  produces small dV/dQ (the trough signal), not a blow-up.
    """
    def _dvdq_group(group):
        v  = group[smoothed_voltage_col].values.astype(float)
        q  = group[capacity_col].values.astype(float)
        dv = np.diff(v)
        dq = np.diff(q)
        if dq_threshold is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                dvdq = np.where(np.abs(dq) > dq_threshold, dv / dq, np.nan)
        else:
            dvdq = dv / dq
        return pd.Series(np.concatenate([[np.nan], dvdq]), index=group.index)

    df[out_col] = df.groupby(cycle_col, group_keys=False).apply(_dvdq_group).values
    return df


def compute_dqdv(df, capacity_col, smoothed_voltage_col, cycle_col,
                 out_col, dv_threshold):
    """
    Compute raw dQ/dV per cycle group. Used for ICA.
    Peaks in dQ/dV vs. voltage mark phase transitions.
    Edge-trimmed NaNs in smoothed_voltage_col propagate naturally.

    dv_threshold: minimum |dV| to compute derivative — caller decides appropriate
                  value based on voltage resolution and noise level.
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

    - Prompts user to choose method and confirm/edit smoothing parameters.
    - Edge trimming applied before fitting to exclude end-dropout artifacts.
    - prompt_edge_trim=False suppresses edge trim prompt (use for derivative
      smoothing passes where edges are already NaN from the voltage smooth step).
    - Shows first cycle with zoom/inspect loop, then any additional cycles on request.
    - Outer loop: adjust parameters and repeat, or confirm and retain column.

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
            print("  Recommended: 1.5% based on dataset spec.")
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
    save_path  : optional filename to save (e.g. 'dva_overlay.png')
    ylim       : optional tuple (ymin, ymax) to clip y-axis scale
                 useful when a steep end-of-discharge feature compresses the
                 bulk of the plot — set to e.g. (-0.0007, 0) for DVA
    cmap       : matplotlib colormap name (default 'plasma'; try 'viridis',
                 'coolwarm', 'turbo' for different visual styles)
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

    # colorbar as cycle legend (cleaner than per-line legend when many cycles)
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
        dqdv_bins = np.full(n_bins, np.nan)

        for b in range(n_bins):
            mask = bin_idx == b
            if mask.sum() > 0:
                # total charge in bin / bin width = dQ/dV for this bin
                dqdv_bins[b] = dq_per_sample[mask].sum() / bin_width

        # map bin dQ/dV back to each original sample point for dataframe alignment
        result = dqdv_bins[bin_idx]
        return pd.Series(result, index=group.index)

    df[out_col] = df.groupby(cycle_col, group_keys=False).apply(_lean_group).values
    return df

# =============================================================================
# FEATURE PROPERTIES (peaks or troughs)
# Generic — works for DVA (troughs in dV/dQ), ICA (peaks in dQ/dV), LEAN (peaks)
# =============================================================================

def find_features(df, cycle_col, x_col, y_col,
                  mode,
                  reference_cycle=None,
                  prominence_threshold=None):
    """
    Interactively identify peaks or troughs in a differential curve and compute
    per-feature properties (position, extremum, depth/height, area) relative to
    a local linear baseline fitted between the shoulders on either side of each
    feature.

    Generic — covers all three analysis types:
      DVA  : mode="trough", y_col="dVdQ_smooth", x_col=capacity_col
      ICA  : mode="peak",   y_col="dQdV_smooth", x_col=voltage_col
      LEAN : mode="peak",   y_col="dQdV_lean",   x_col=voltage_col

    Workflow
    --------
    1. Display prominence histogram for a reference cycle.
    2. Prompt user to set prominence threshold.
    3. Display detected features overlaid on the curve with local baselines
       and shaded areas — loop until threshold is confirmed.
    4. Apply confirmed threshold to all cycles and return a summary dataframe.

    Parameters
    ----------
    df                   : dataframe containing cycle_col, x_col, y_col
    cycle_col            : column identifying cycle number
    x_col                : x-axis column (capacity or voltage)
    y_col                : smoothed differential column
    mode                 : "trough" for DVA, "peak" for ICA/LEAN
    reference_cycle      : cycle for threshold tuning. None = first cycle.
    prominence_threshold : starting prominence. None = prompted after histogram.

    Returns
    -------
    feature_df : dataframe, one row per (cycle, feature), columns:
                 cycle_col, feature_idx, x_position, y_extremum,
                 baseline_at_extremum, depth_or_height, area
    """
    from scipy.signal import find_peaks, peak_prominences

    plt.close("all")   # clear any figures left open from previous pipeline steps

    if mode not in ("peak", "trough"):
        raise ValueError("mode must be 'peak' or 'trough'")

    sign     = 1 if mode == "peak" else -1
    label    = "Peak"   if mode == "peak" else "Trough"
    label_pl = "Peaks"  if mode == "peak" else "Troughs"

    cycles = sorted(df[cycle_col].unique())
    if reference_cycle is None:
        reference_cycle = cycles[0]

    def _find_shoulders(y, peak_idx, search_radius=None, exclusion_zone=5):
        """
        Find the nearest inflection points (zero crossings of dy) on either
        side of a trough/peak as shoulder points for local baseline fitting.

        Clips extreme dy values (>99th percentile absolute) before finding sign
        changes — this suppresses end-of-discharge spike artifacts that would
        otherwise create spurious critical points far from the feature, without
        needing a search_radius that might accidentally exclude legitimate nearby
        shoulders.

        exclusion_zone: points around peak_idx to ignore — prevents the
        extremum's own sign change from being picked up as a shoulder.
        Falls back to array edges if no inflection point found on a given side.
        """
        n  = len(y)
        dy = np.gradient(y)

        # clip extreme dy to suppress end-spike artifacts before finding zero crossings
        clip_limit = np.percentile(np.abs(dy), 99)
        dy_clipped = np.clip(dy, -clip_limit, clip_limit)

        sign_changes = np.where(np.diff(np.sign(dy_clipped)))[0]
        sign_changes = sign_changes[np.abs(sign_changes - peak_idx) > exclusion_zone]

        left_candidates  = sign_changes[sign_changes < peak_idx]
        right_candidates = sign_changes[sign_changes > peak_idx]

        if search_radius is not None:
            left_candidates  = left_candidates[left_candidates  >= peak_idx - search_radius]
            right_candidates = right_candidates[right_candidates <= peak_idx + search_radius]

        lb = int(left_candidates[-1])  if len(left_candidates)  > 0 else 0
        rb = int(right_candidates[0])  if len(right_candidates) > 0 else n - 1
        return lb, rb



    def _fit_local_baseline(x, y, lb, rb, mode, fixed_baseline=None):
        """
        Horizontal baseline for area integration.

        If fixed_baseline is provided (computed from cycle 10), uses that
        fixed value for all cycles — recommended for LAM quantification
        since it gives area a consistent absolute meaning across cycles.

        Otherwise computes per-cycle from global max (trough) or min (peak),
        clipped to 0 if the curve crosses zero — used for the interactive
        visualisation loop where per-cycle display is appropriate.
        """
        if fixed_baseline is not None:
            return np.full(len(x), fixed_baseline)

        if mode == "trough":
            baseline_level = float(np.max(y))
            if baseline_level > 0:
                baseline_level = 0.0
        else:
            baseline_level = float(np.min(y))
            if baseline_level < 0:
                baseline_level = 0.0

        return np.full(len(x), baseline_level)

    def _compute_fixed_baseline(cyc):
        sub = df[df[cycle_col] == cyc].dropna(subset=[y_col])
        y_base = sub[y_col].values.astype(float)

        if mode == "trough":
            baseline_level = float(np.max(y_base))
            if baseline_level > 0:
                baseline_level = 0.0
        else:
            baseline_level = float(np.min(y_base))
            if baseline_level < 0:
                baseline_level = 0.0
        print(f"Fixed baseline for cycle {cyc}: {baseline_level:.3e}")

        return baseline_level

    def _get_ref_data(cyc):
        sub    = df[df[cycle_col] == cyc].dropna(subset=[y_col])
        x_r    = sub[x_col].values
        y_r    = sub[y_col].values
        y_w    = sign * y_r
        pks, _ = find_peaks(y_w, prominence=0)
        proms  = peak_prominences(y_w, pks)[0] if len(pks) > 0 else np.array([])
        return x_r, y_r, y_w, proms

    def _plot_cycle_with_threshold(x_c, y_c, y_work_c, cyc, thresh, fixed_baseline):
        """Plot a single cycle's curve with detected features and baselines."""
        pks, _ = find_peaks(y_work_c, prominence=thresh)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        ax.plot(x_c, y_c, color="tab:blue", linewidth=1, label=y_col)
        for i, pk in enumerate(pks):
            lb, rb   = _find_shoulders(y_c, pk, search_radius=None)
            baseline = _fit_local_baseline(x_c, y_c, lb, rb, mode,
                                           fixed_baseline=fixed_baseline)
            ax.fill_between(x_c[lb:rb+1], y_c[lb:rb+1], baseline[lb:rb+1],
                            alpha=0.25, color="tab:orange")
            ax.plot(x_c[lb:rb+1], baseline[lb:rb+1],
                    color="tab:orange", linewidth=1, linestyle="--")
            ax.scatter(x_c[pk], y_c[pk], color="red", s=50, zorder=5,
                       label=f"{label} extremum" if i == 0 else "")
            ax.annotate(f"F{i+1}\n{x_c[pk]:.3g}", (x_c[pk], y_c[pk]),
                        textcoords="offset points",
                        xytext=(0, 12 if mode == "peak" else -20),
                        fontsize=7, ha="center", color="red")
        ax.set_xlabel(x_col); ax.set_ylabel(y_col)
        ax.set_title(f"Detected {label_pl} — Cycle {int(cyc)}\n"
                     f"prominence >= {thresh:.2e}")
        ax.legend(fontsize=8)

        ax2 = axes[1]
        pks_all, _ = find_peaks(y_work_c, prominence=0)
        proms_all  = peak_prominences(y_work_c, pks_all)[0] if len(pks_all) > 0 else np.array([])
        if len(proms_all) > 0:
            ax2.hist(proms_all, bins=30, color="tab:blue",
                     edgecolor="white", linewidth=0.5)
        ax2.axvline(thresh, color="red", linewidth=1.5, linestyle="--",
                    label=f"Threshold: {thresh:.2e}")
        ax2.set_xlabel("Prominence"); ax2.set_ylabel("Count")
        ax2.set_title(f"Prominence Histogram — Cycle {int(cyc)}")
        ax2.legend(fontsize=8)

        fig.tight_layout()
        plt.show()
        plt.close("all")
        return len(pks)

    # --- Step 1: choose reference cycle, show curve + histogram without threshold ---
    print(f"\n{'='*60}")
    print(f"{label_pl} detection")
    print(f"  Available cycles: {[int(c) for c in cycles]}")
    print(f"  Default reference cycle: {int(reference_cycle)}")
    if input("  Use a different reference cycle? (yes/no): ").strip().lower() == "yes":
        try:
            cyc_input = int(input("  Enter cycle number: "))
            if cyc_input in [int(c) for c in cycles]:
                reference_cycle = [c for c in cycles if int(c) == cyc_input][0]
                print(f"  Reference cycle set to {int(reference_cycle)}")
            else:
                print(f"  Cycle {cyc_input} not found — keeping {int(reference_cycle)}")
        except ValueError:
            print(f"  Invalid — keeping {int(reference_cycle)}")

    fixed_baseline = _compute_fixed_baseline(reference_cycle)
    x_ref, y_ref, y_work, all_proms = _get_ref_data(reference_cycle)

    print(f"\n  {len(all_proms)} candidate {label_pl.lower()} in cycle {int(reference_cycle)}.")
    if len(all_proms) > 0:
        print(f"  Prominence range: {all_proms.min():.2e} — {all_proms.max():.2e}")
    print(f"  Look for a natural gap between noise and real features.")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(x_ref, y_ref, color="tab:blue", linewidth=1, label=y_col)
    axes[0].set_xlabel(x_col); axes[0].set_ylabel(y_col)
    axes[0].set_title(f"Cycle {int(reference_cycle)} — {y_col}\n(no threshold set)")
    axes[0].legend(fontsize=8)
    if len(all_proms) > 0:
        axes[1].hist(all_proms, bins=30, color="tab:blue",
                     edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("Prominence"); axes[1].set_ylabel("Count")
    axes[1].set_title(f"{label} Prominence Histogram — Cycle {int(reference_cycle)}")
    fig.suptitle("Identify gap between noise and real features to set threshold",
                 fontsize=11)
    fig.tight_layout()
    plt.show()
    plt.close("all")

    # --- Step 2: input threshold ---
    if prominence_threshold is None:
        try:
            prominence_threshold = float(
                input(f"  Enter prominence threshold: ")
            )
        except ValueError:
            prominence_threshold = float(all_proms.max()) * 0.3 if len(all_proms) > 0 else 1e-5
            print(f"  Invalid — defaulting to {prominence_threshold:.2e}")

    # --- Steps 3-5: show threshold applied, optionally check other cycles, confirm ---
    while True:
        # Step 3: show reference cycle with threshold applied
        n_detected = _plot_cycle_with_threshold(
            x_ref, y_ref, y_work, reference_cycle, prominence_threshold, fixed_baseline
        )
        print(f"  Threshold = {prominence_threshold:.2e} → "
              f"{n_detected} {label_pl.lower()} detected on cycle {int(reference_cycle)}")

        # Step 4: optionally view a different cycle
        if input("  View a different cycle with this threshold? (yes/no): ").strip().lower() == "yes":
            print(f"  Available cycles: {[int(c) for c in cycles]}")
            try:
                cyc_input = int(input("  Enter cycle number: "))
                if cyc_input in [int(c) for c in cycles]:
                    view_cyc = [c for c in cycles if int(c) == cyc_input][0]
                    view_sub = df[df[cycle_col] == view_cyc].dropna(subset=[y_col])
                    x_v      = view_sub[x_col].values
                    y_v      = view_sub[y_col].values
                    y_work_v = sign * y_v
                    _plot_cycle_with_threshold(x_v, y_v, y_work_v, view_cyc, prominence_threshold, fixed_baseline
                                               )
                else:
                    print(f"  Cycle {cyc_input} not found — skipping.")
            except ValueError:
                print("  Invalid — skipping.")

        # Step 5: confirm or adjust threshold
        if input(f"  Confirm threshold {prominence_threshold:.2e}? (yes/no): ").strip().lower() == "yes":
            print(f"  ✓ Confirmed: {prominence_threshold:.2e}")
            break
        else:
            try:
                prominence_threshold = float(
                    input(f"  New threshold (current {prominence_threshold:.2e}): ")
                )
            except ValueError:
                print("  Invalid — keeping current threshold.")
            # loop back to Step 3 with new threshold


    # --- Step 4: apply to all cycles ---
    print(f"\nComputing {label_pl.lower()} properties for all cycles...")
    print(f"  Fixed baseline (reference cycle {int(reference_cycle)}): {fixed_baseline:.4e}")
    records = []

    for cyc in cycles:
        sub      = df[df[cycle_col] == cyc].dropna(subset=[y_col])
        x_c      = sub[x_col].values
        y_c      = sub[y_col].values
        y_work_c = sign * y_c

        pks, _ = find_peaks(y_work_c, prominence=prominence_threshold)
        if len(pks) == 0:
            continue

        for i, pk in enumerate(pks):
            lb, rb          = _find_shoulders(y_c, pk, search_radius=None)
            baseline        = _fit_local_baseline(x_c, y_c, lb, rb, mode,
                                                  fixed_baseline=fixed_baseline)
            baseline_at_ext = baseline[pk]
            depth_or_height = abs(y_c[pk] - baseline_at_ext)
            area            = np.abs(np.trapezoid(
                                y_c[lb:rb+1] - baseline[lb:rb+1],
                                x_c[lb:rb+1]
                              ))

            records.append({
                cycle_col:               int(cyc),
                "feature_idx":           i + 1,
                "x_position":            x_c[pk],
                "y_extremum":            y_c[pk],
                "baseline_at_extremum":  baseline_at_ext,
                "depth_or_height":       depth_or_height,
                "area":                  area,
                "baseline_level":        fixed_baseline
            })

    feature_df = pd.DataFrame(records)
    print(f"  Done — {len(feature_df)} {label_pl.lower()} across {len(cycles)} cycles.")
    return feature_df

# =============================================================================
# DEBUG — CRITICAL POINT MAPPING
# =============================================================================

def debug_critical_points(df, cycle_col, x_col, y_col, mode,
                           cycle=None, prominence_threshold=0,
                           search_radius=400, exclusion_zone=5):
    """
    Debug helper to visualise critical points (dy = 0 sign changes) on a
    single cycle's differential curve alongside detected features.

    Plots three panels:
      Left   — the raw y curve with all critical points marked (grey dots),
                the detected feature extrema (red dots), and the shoulder
                pairs selected for each feature (orange vertical lines).
      Middle — the first derivative dy with zero line, showing where sign
               changes occur.
      Right  — zoomed table printout in terminal of every critical point
               index, x position, and which feature (if any) it was
               assigned to as left/right shoulder.

    Parameters
    ----------
    df                 : dataframe with cycle_col, x_col, y_col
    cycle_col          : cycle number column
    x_col              : x-axis column
    y_col              : smoothed differential column
    mode               : "trough" or "peak"
    cycle              : which cycle to debug. None = first cycle.
    prominence_threshold: feature detection threshold (use same value as
                          find_features to see the same features)
    search_radius      : same value used in find_features (default 400)
    exclusion_zone     : same value used in find_features (default 5)
    """

    sign     = 1 if mode == "peak" else -1
    label    = "Peak"   if mode == "peak" else "Trough"
    label_pl = "Peaks"  if mode == "peak" else "Troughs"

    cycles = sorted(df[cycle_col].unique())
    if cycle is None:
        cycle = cycles[0]

    sub    = df[df[cycle_col] == cycle].dropna(subset=[y_col])
    x      = sub[x_col].values
    y      = sub[y_col].values
    y_work = sign * y
    dy     = np.gradient(y)

    # clip extreme dy to match _find_shoulders behavior
    clip_limit  = np.percentile(np.abs(dy), 99)
    dy_clipped  = np.clip(dy, -clip_limit, clip_limit)

    # all critical points — use clipped dy so debug reflects what _find_shoulders sees
    sign_changes = np.where(np.diff(np.sign(dy_clipped)))[0]

    # detected features
    pks, _ = find_peaks(y_work, prominence=prominence_threshold)

    # for each feature find its shoulders using same clipped dy
    shoulder_pairs = []
    for pk in pks:
        sc = sign_changes.copy()
        sc = sc[np.abs(sc - pk) > exclusion_zone]
        lc = sc[sc < pk]
        rc = sc[sc > pk]
        if search_radius is not None:
            lc = lc[lc >= pk - search_radius]
            rc = rc[rc <= pk + search_radius]
        lb = int(lc[-1]) if len(lc) > 0 else 0
        rb = int(rc[0])  if len(rc) > 0 else len(x) - 1
        shoulder_pairs.append((pk, lb, rb))

    # --- terminal printout ---
    print(f"\n{'='*60}")
    print(f"Debug critical points — Cycle {int(cycle)}, mode={mode}")
    print(f"  {len(sign_changes)} critical points after dy clipping (99th pct = {clip_limit:.2e})")
    print(f"  {len(pks)} features detected at prominence >= {prominence_threshold:.2e}")
    print(f"\n  Critical points:")
    print(f"  {'idx':>6}  {'x_pos':>10}  {'y_val':>12}  {'assigned_to'}")
    for sc in sign_changes:
        assigned = ""
        for pk, lb, rb in shoulder_pairs:
            if sc == lb:
                assigned = f"LEFT  shoulder of F{pks.tolist().index(pk)+1} (pk@{x[pk]:.1f})"
            elif sc == rb:
                assigned = f"RIGHT shoulder of F{pks.tolist().index(pk)+1} (pk@{x[pk]:.1f})"
        print(f"  {sc:>6}  {x[sc]:>10.2f}  {y[sc]:>12.4e}  {assigned}")

    print(f"\n  Feature shoulders:")
    for pk, lb, rb in shoulder_pairs:
        fidx = pks.tolist().index(pk) + 1
        print(f"  F{fidx}: peak@{x[pk]:.1f}  left_shoulder@{x[lb]:.1f} (idx={lb})"
              f"  right_shoulder@{x[rb]:.1f} (idx={rb})")

    # --- figure: 2 panels ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # left: y curve with critical points and shoulder pairs
    ax = axes[0]
    ax.plot(x, y, color="tab:blue", linewidth=1, label=y_col, zorder=1)

    # all critical points as grey dots
    ax.scatter(x[sign_changes], y[sign_changes],
               color="grey", s=20, zorder=2, label="Critical points (dy=0)")

    # shoulder pairs as vertical orange lines + shaded region
    for pk, lb, rb in shoulder_pairs:
        fidx = pks.tolist().index(pk) + 1
        ax.axvline(x[lb], color="tab:orange", linewidth=1, linestyle="--", alpha=0.7)
        ax.axvline(x[rb], color="tab:orange", linewidth=1, linestyle="--", alpha=0.7)
        ax.axvspan(x[lb], x[rb], alpha=0.08, color="tab:orange")
        ax.scatter(x[pk], y[pk], color="red", s=60, zorder=5,
                   label=f"{label} extremum" if fidx == 1 else "")
        ax.annotate(f"F{fidx}\n{x[pk]:.0f}",
                    (x[pk], y[pk]),
                    textcoords="offset points",
                    xytext=(0, 12 if mode == "peak" else -20),
                    fontsize=7, ha="center", color="red")

    ax.set_xlabel(x_col); ax.set_ylabel(y_col)
    ax.set_title(f"Critical Points & Shoulders — Cycle {int(cycle)}\n"
                 f"orange dashed = selected shoulders, grey = all dy=0")
    ax.legend(fontsize=7)

    # right: first derivative dy with zero line
    ax2 = axes[1]
    ax2.plot(x, dy_clipped, color="tab:purple", linewidth=1, label="dy (clipped 99th pct)")
    ax2.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax2.scatter(x[sign_changes], dy_clipped[sign_changes],
                color="grey", s=20, zorder=3, label="Sign changes")
    for pk, lb, rb in shoulder_pairs:
        ax2.axvline(x[lb], color="tab:orange", linewidth=1, linestyle="--", alpha=0.7)
        ax2.axvline(x[rb], color="tab:orange", linewidth=1, linestyle="--", alpha=0.7)
        ax2.scatter(x[pk], dy_clipped[pk], color="red", s=60, zorder=5)
    ax2.set_xlabel(x_col)
    ax2.set_ylabel(f"d({y_col})/d(index)")
    ax2.set_title("First Derivative dy\n(shoulders where dy crosses zero)")
    ax2.legend(fontsize=7)

    fig.suptitle(f"Debug: Critical Points — Cycle {int(cycle)}, mode={mode}, "
                 f"search_radius={search_radius}, exclusion_zone={exclusion_zone}",
                 fontsize=10)
    fig.tight_layout()
    plt.show()
    plt.close("all")


# =============================================================================
# FEATURE TRACKING — class-based cross-cycle feature identity
# =============================================================================

class DifferentialFeature:
    """
    Represents a single identified peak or trough across multiple cycles.

    Each instance corresponds to one physical electrochemical feature
    (e.g. a specific redox reaction / phase transition). Properties are
    stored per-cycle in dictionaries keyed by cycle number.

    Attributes
    ----------
    feature_id   : int — unique identifier assigned at creation
    mode         : "peak" or "trough"
    cycles       : list of cycle numbers this feature was matched to
    x_position   : dict {cycle: x_position} — feature position on x-axis
    y_extremum   : dict {cycle: y_extremum} — peak/trough value
    baseline_at_extremum : dict {cycle: value}
    depth_or_height      : dict {cycle: value}
    area                 : dict {cycle: value}
    """

    _id_counter = 0

    def __init__(self, mode, seed_cycle, seed_record):
        DifferentialFeature._id_counter += 1
        self.feature_id = DifferentialFeature._id_counter
        self.mode       = mode
        self.cycles     = [seed_cycle]
        self.x_position          = {seed_cycle: seed_record["x_position"]}
        self.y_extremum          = {seed_cycle: seed_record["y_extremum"]}
        self.baseline_at_extremum = {seed_cycle: seed_record["baseline_at_extremum"]}
        self.depth_or_height     = {seed_cycle: seed_record["depth_or_height"]}
        self.area                = {seed_cycle: seed_record["area"]}

    def add_cycle(self, cycle, record):
        """Add measurements from a new cycle to this feature instance."""
        self.cycles.append(cycle)
        self.x_position[cycle]           = record["x_position"]
        self.y_extremum[cycle]           = record["y_extremum"]
        self.baseline_at_extremum[cycle] = record["baseline_at_extremum"]
        self.depth_or_height[cycle]      = record["depth_or_height"]
        self.area[cycle]                 = record["area"]

    def to_dataframe(self):
        """Return a long-format dataframe of this feature's properties across cycles."""
        rows = []
        for cyc in sorted(self.cycles):
            rows.append({
                "feature_id":            self.feature_id,
                "mode":                  self.mode,
                "cycle":                 cyc,
                "x_position":            self.x_position[cyc],
                "y_extremum":            self.y_extremum[cyc],
                "baseline_at_extremum":  self.baseline_at_extremum[cyc],
                "depth_or_height":       self.depth_or_height[cyc],
                "area":                  self.area[cyc],
            })
        return pd.DataFrame(rows)

    def __repr__(self):
        return (f"DifferentialFeature(id={self.feature_id}, mode={self.mode}, "
                f"cycles={len(self.cycles)}, "
                f"x_range=[{min(self.x_position.values()):.1f}, "
                f"{max(self.x_position.values()):.1f}])")


def track_features(feature_df, cycle_col, mode, max_shift=None):
    """
    Match features across cycles using nearest-neighbour tracking on x_position.

    Each feature detected in the first cycle seeds a DifferentialFeature instance.
    For each subsequent cycle, detected features are matched to existing instances
    by finding the closest x_position. Unmatched features (further than max_shift
    from any existing instance) seed new DifferentialFeature instances.

    Parameters
    ----------
    feature_df  : dataframe output of find_features — one row per (cycle, feature)
    cycle_col   : name of cycle column in feature_df
    mode        : "peak" or "trough" — stored on each DifferentialFeature instance
    max_shift   : maximum x_position shift (in x_col units) allowed for a match.
                  None = always match to nearest regardless of distance.
                  Recommended: set to ~10-20% of the x range to prevent
                  mismatches when a feature appears/disappears mid-experiment.

    Returns
    -------
    features    : list of DifferentialFeature instances
    tracked_df  : long-format dataframe with feature_id column added, suitable
                  for plotting and LLI/LAM calculations
    """
    # reset class counter so IDs start fresh each call
    DifferentialFeature._id_counter = 0

    cycles   = sorted(feature_df[cycle_col].unique())
    features = []   # list of DifferentialFeature instances

    for cyc in cycles:
        cyc_rows = feature_df[feature_df[cycle_col] == cyc].to_dict("records")

        if not features:
            # first cycle — seed one instance per detected feature
            for row in cyc_rows:
                features.append(DifferentialFeature(mode, cyc, row))
            continue

        # subsequent cycles — match each detected feature to nearest existing instance
        # using x_position of the most recently seen cycle for each instance
        unmatched_rows     = list(cyc_rows)
        matched_instance   = set()

        for row in cyc_rows:
            x_new = row["x_position"]

            # find nearest existing feature by last known x_position
            best_instance = None
            best_dist     = float("inf")
            for feat in features:
                if id(feat) in matched_instance:
                    continue   # already matched this cycle
                last_x = feat.x_position[feat.cycles[-1]]
                dist   = abs(x_new - last_x)
                if dist < best_dist:
                    best_dist     = dist
                    best_instance = feat

            if best_instance is not None:
                if max_shift is None or best_dist <= max_shift:
                    best_instance.add_cycle(cyc, row)
                    matched_instance.add(id(best_instance))
                else:
                    # too far from any existing feature — new instance
                    features.append(DifferentialFeature(mode, cyc, row))
            else:
                features.append(DifferentialFeature(mode, cyc, row))

    # compile tracked_df
    all_dfs   = [f.to_dataframe() for f in features]
    tracked_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    print(f"\nFeature tracking complete:")
    for feat in features:
        print(f"  {feat}")

    return features, tracked_df