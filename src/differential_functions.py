import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

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

    # UnivariateSpline requires at least k+1 unique x points (k=3 → min 4).
    # In practice scipy needs more headroom when s > 0; guard with 10 as a
    # safe minimum — if a cycle group is this small the data isn't worth
    # fitting anyway and we return it unsmoothed rather than crashing.
    if len(x_u) < 10:
        return y_vals.copy()

    try:
        spline = UnivariateSpline(x_u, y_u, s=smoothing_factor, k=3)
    except Exception as e:
        print(f"  Warning: spline fit failed ({e}) — returning unsmoothed.")
        return y_vals.copy()

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
                 out_col, dq_threshold=None, negate=False):
    """
    Compute raw dV/dQ per cycle group. Used for DVA.
    Troughs in dV/dQ vs. capacity mark redox reactions:
      LLI  -> lateral shift of trough positions along capacity axis
      LAM  -> reduction in trough depth or collapse
    Edge-trimmed NaNs in smoothed_voltage_col propagate naturally.

    dq_threshold: minimum |dQ| to compute derivative. Default None = no
                  threshold applied. Only needed for non-uniform capacity
                  grids (e.g. raw cycler data, GITT) where dQ can approach
                  zero between points and cause dV/dQ to blow up.
                  For uniform resampled grids (e.g. 0.5 mAh/pt), dQ is
                  fixed so no threshold is needed.

    negate      : flip the sign of the output. Default False.
                  Discharge DVA: capacity ascending, voltage descending ->
                  dV/dQ naturally negative (troughs are negative). negate=False.
                  Charge DVA: capacity ascending, voltage ascending ->
                  dV/dQ naturally positive (troughs become peaks). negate=False
                  but find_features mode should be set to "peak".
                  Only set negate=True if your dataset capacity or voltage
                  convention produces the wrong sign — e.g. capacity counted
                  downward on discharge instead of upward, which would flip
                  dV/dQ positive when it should be negative.
                  Controlled via IS_DISCHARGE in config.py.
    """
    sign = -1 if negate else 1

    def _dvdq_group(group):
        v  = group[smoothed_voltage_col].values.astype(float)
        q  = group[capacity_col].values.astype(float)

        # NaN out the first and last valid point adjacent to the trim boundary
        # to prevent a large dV spike at the NaN/valid transition edge.
        valid = ~np.isnan(v)
        if valid.any():
            first_valid = np.argmax(valid)
            last_valid  = len(valid) - np.argmax(valid[::-1]) - 1
            if first_valid > 0:
                v[first_valid] = np.nan
            if last_valid < len(v) - 1:
                v[last_valid] = np.nan

        dv = np.diff(v)
        dq = np.diff(q)
        if dq_threshold is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                dvdq = np.where(np.abs(dq) > dq_threshold, dv / dq, np.nan)
        else:
            dvdq = dv / dq
        return pd.Series(np.concatenate([[np.nan], sign * dvdq]), index=group.index)

    df[out_col] = df.groupby(cycle_col, group_keys=False).apply(_dvdq_group).values
    return df


def compute_dqdv(df, capacity_col, smoothed_voltage_col, cycle_col,
                 out_col, dv_threshold=None, negate=False):
    """
    Compute raw dQ/dV per cycle group. Used for ICA.
    Peaks in dQ/dV vs. voltage mark phase transitions.
    Edge-trimmed NaNs in smoothed_voltage_col propagate naturally.

    dv_threshold: minimum |dV| to compute derivative. Default None = no
                  threshold applied. Only needed for non-uniform voltage
                  grids (e.g. raw cycler data) where dV can approach zero
                  and cause dQ/dV to blow up. For uniform resampled grids
                  (e.g. 0.001 V/pt), dV is fixed so no threshold is needed.

    negate      : flip the sign of the output. Default False.
                  Charge ICA: voltage ascending (natural grid direction) ->
                  dQ/dV naturally positive (peaks are positive). negate=False.
                  Discharge ICA: resampling sorts voltage ascending, which
                  reverses the discharge direction. dQ is negative (capacity
                  decreasing as voltage increases in sorted order), making
                  dQ/dV negative. Set negate=True to recover positive peaks.
                  Only override if your dataset produces unexpected sign —
                  e.g. discharge capacity stored as decreasing values would
                  make dQ/dV positive on discharge without needing negate=True.
                  Controlled via IS_DISCHARGE in config.py.
    """
    sign = -1 if negate else 1

    def _dqdv_group(group):
        v  = group[smoothed_voltage_col].values.astype(float)
        q  = group[capacity_col].values.astype(float)

        # NaN out the first and last valid point adjacent to the trim boundary.
        # np.diff at the NaN/valid transition computes valid - NaN = NaN correctly
        # for the NaN side, but the first interior diff (valid[1] - valid[0]) sees
        # a large capacity jump at the boundary edge, producing a spike. Masking
        # one extra point on each side of the NaN boundary prevents this.
        valid = ~np.isnan(q)
        if valid.any():
            first_valid = np.argmax(valid)
            last_valid  = len(valid) - np.argmax(valid[::-1]) - 1
            if first_valid > 0:
                q[first_valid] = np.nan
            if last_valid < len(q) - 1:
                q[last_valid] = np.nan

        dv = np.diff(v)
        dq = np.diff(q)
        if dv_threshold is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                dqdv = np.where(np.abs(dv) > dv_threshold, dq / dv, np.nan)
        else:
            dqdv = dq / dv
        return pd.Series(np.concatenate([[np.nan], sign * dqdv]), index=group.index)

    df[out_col] = df.groupby(cycle_col, group_keys=False).apply(_dqdv_group).values
    return df


# =============================================================================
# UNIFORM GRID RESAMPLING
# Used by both DVA (x=capacity) and ICA (x=voltage) before differentiation.
# Eliminates dV/dQ or dQ/dV blow-up caused by near-zero steps in the x-axis.
# =============================================================================

def resample_to_uniform_grid(df, x_col, y_col, cycle_col, step,
                              x_min=None, x_max=None,
                              per_cycle_bounds=False):
    """
    Resample each cycle onto a uniform x-axis grid by interpolating y onto
    evenly spaced x points. Returns a new dataframe — original is unchanged.

    This is the correct fix for dQ/dV or dV/dQ noise caused by a non-uniform
    x-axis: once x steps are fixed, the denominator in the derivative is
    constant and cannot blow up at plateaus or turnaround regions.

    DVA : x_col=capacity_col, y_col=voltage_col, step in mAh (e.g. 0.5)
          Use per_cycle_bounds=False (default) — global bounds preserve the
          capacity fade signal across cycles. Cycles that no longer reach
          the full capacity range naturally drop out at the high end.

    ICA : x_col=voltage_col, y_col=capacity_col, step in V (e.g. 0.001)
          Use per_cycle_bounds=True — each cycle's voltage window is
          physically meaningful and reflects the actual cell state. Using
          global bounds extends the grid below cycles that don't reach the
          lowest voltage, producing NaN-padded regions where the derivative
          blows up at the boundary. Per-cycle bounds avoids this entirely.
          Feature tracking across cycles is by position (track_features),
          not grid index, so varying windows cause no alignment issues.

    Parameters
    ----------
    df               : dataframe with cycle_col, x_col, y_col
    x_col            : column to use as the uniform grid axis
    y_col            : column to interpolate onto the grid
    cycle_col        : column identifying cycle number
    step             : grid spacing in x_col units
    x_min            : global lower bound. Only used when per_cycle_bounds=False.
    x_max            : global upper bound. Only used when per_cycle_bounds=False.
    per_cycle_bounds : if True, each cycle is resampled on its own [min, max]
                       range, ignoring x_min/x_max. Standard for ICA.
                       if False (default), global bounds are used when provided,
                       falling back to per-cycle if not provided. Standard for DVA.

    Returns
    -------
    resampled_df : new dataframe with columns [cycle_col, x_col, y_col],
                   one row per grid point per cycle. Any additional columns
                   from the original df are dropped — resample only the
                   columns needed for the derivative step.

    Notes
    -----
    - Interpolation is linear (np.interp). Sufficient for electrochemical
      curves sampled at reasonable resolution; no extrapolation beyond the
      cycle's own x range.
    - When per_cycle_bounds=False and global bounds extend beyond a cycle's
      actual data range, those points are set to NaN (left=nan, right=nan).
    - step should be >= the median x spacing of the raw data to avoid
      artificial interpolation artefacts. Check with:
          df.groupby(cycle_col)[x_col].apply(lambda g: np.median(np.diff(g.values)))
    """
    records = []
    cycles  = sorted(df[cycle_col].unique())

    # global bounds — only computed and used when per_cycle_bounds=False
    if not per_cycle_bounds:
        g_min = x_min if x_min is not None else None
        g_max = x_max if x_max is not None else None
    else:
        g_min = None
        g_max = None

    for cyc in cycles:
        sub   = df[df[cycle_col] == cyc].dropna(subset=[x_col, y_col])
        x_raw = sub[x_col].values.astype(float)
        y_raw = sub[y_col].values.astype(float)

        # sort by x to guarantee np.interp monotonicity requirement
        order = np.argsort(x_raw)
        x_raw = x_raw[order]
        y_raw = y_raw[order]

        # per_cycle_bounds=True always uses the cycle's own range
        # per_cycle_bounds=False uses global bounds if provided, else cycle range
        lo = x_raw.min() if (per_cycle_bounds or g_min is None) else g_min
        hi = x_raw.max() if (per_cycle_bounds or g_max is None) else g_max

        x_grid = np.arange(lo, hi + step * 0.5, step)  # +0.5*step avoids float edge exclusion

        # interpolate; points outside cycle's own range become NaN
        y_grid = np.interp(x_grid, x_raw, y_raw,
                           left=np.nan, right=np.nan)

        for xi, yi in zip(x_grid, y_grid):
            records.append({cycle_col: cyc, x_col: xi, y_col: yi})

    resampled_df = pd.DataFrame(records)
    return resampled_df


def check_grid_uniformity(df, x_col, cycle_col, sample_cycles=5):
    """
    Print a quick diagnostic of x-axis spacing to help decide whether
    resampling is needed and what step size to use.

    Reports median, min, max, and std of diff(x_col) for a sample of cycles.
    A large std relative to the median indicates a non-uniform grid that will
    cause derivative blow-up and should be resampled before differentiation.

    Parameters
    ----------
    df            : dataframe with cycle_col and x_col
    x_col         : column to check spacing on
    cycle_col     : column identifying cycle number
    sample_cycles : how many cycles to report (evenly spaced across all cycles)
    """
    cycles  = sorted(df[cycle_col].unique())
    indices = np.linspace(0, len(cycles) - 1, min(sample_cycles, len(cycles)), dtype=int)
    sample  = [cycles[i] for i in indices]

    print(f"\nGrid uniformity check — '{x_col}'")
    print(f"  {'cycle':>8}  {'n_pts':>6}  {'median_step':>12}  {'min_step':>10}  "
          f"{'max_step':>10}  {'std_step':>10}  {'uniform?':>9}")

    for cyc in sample:
        sub  = df[df[cycle_col] == cyc].dropna(subset=[x_col])
        vals = sub[x_col].values.astype(float)
        d    = np.diff(np.sort(vals))
        med  = np.median(d)
        uniform = "yes" if (np.std(d) / med < 0.05) else "NO"
        print(f"  {int(cyc):>8}  {len(vals):>6}  {med:>12.5f}  {d.min():>10.5f}  "
              f"{d.max():>10.5f}  {np.std(d):>10.5f}  {uniform:>9}")

    print(f"\n  Suggested step: median of medians = "
          f"{np.median([np.median(np.diff(np.sort(df[df[cycle_col]==c][x_col].dropna().values.astype(float)))) for c in sample]):.5f}")


# =============================================================================
# ENSURE UNIFORM GRID — wrapper called by DVA and ICA before smoothing
# =============================================================================

def ensure_uniform_grid(df, x_col, y_col, cycle_col,
                        step=None, uniformity_threshold=0.05,
                        x_min=None, x_max=None,
                        per_cycle_bounds=False):
    """
    Check whether the x-axis grid is uniform and resample if not.
    Called by both DVA (x=capacity) and ICA (x=voltage) before the smoothing
    and derivative steps — no assumption is made about the input data.

    Uniformity is defined as: std(diff(x)) / median(diff(x)) < threshold.
    If ANY sampled cycle fails this test, the entire dataframe is resampled.

    If step is None, the median x-spacing across all cycles is used as the
    grid step — a safe default that avoids interpolation artefacts. Pass an
    explicit step to override (e.g. 0.5 mAh for DVA, 0.001 V for ICA).

    Parameters
    ----------
    df                   : dataframe with cycle_col, x_col, y_col
    x_col                : column to check and use as grid axis
    y_col                : column to interpolate onto the grid
    cycle_col            : column identifying cycle number
    step                 : grid spacing in x_col units. None = auto from data.
    uniformity_threshold : std/median ratio above which resampling is triggered.
                           Default 0.05 (5%). Lower = stricter.
    x_min                : optional global grid lower bound. Only used when
                           per_cycle_bounds=False. None = per-cycle min.
    x_max                : optional global grid upper bound. Only used when
                           per_cycle_bounds=False. None = per-cycle max.
    per_cycle_bounds     : passed through to resample_to_uniform_grid.
                           True = each cycle resampled on its own range (ICA).
                           False = global bounds used when provided (DVA).

    Returns
    -------
    df       : original df if already uniform; resampled df otherwise.
               Resampled df contains only [cycle_col, x_col, y_col].
    resampled: bool — True if resampling was applied.
    step_used: float — grid step used (auto or provided).
    """
    cycles  = sorted(df[cycle_col].unique())
    indices = np.linspace(0, len(cycles) - 1, min(5, len(cycles)), dtype=int)
    sample  = [cycles[i] for i in indices]

    ratios = []
    medians = []
    for cyc in sample:
        sub  = df[df[cycle_col] == cyc].dropna(subset=[x_col])
        vals = np.sort(sub[x_col].values.astype(float))
        d    = np.diff(vals)
        if len(d) == 0:
            continue
        med = np.median(d)
        medians.append(med)
        ratios.append(np.std(d) / med if med > 0 else 0.0)

    needs_resample = any(r > uniformity_threshold for r in ratios)
    step_used      = step if step is not None else float(np.median(medians))

    if needs_resample:
        max_ratio = max(ratios)
        print(f"\n  Grid uniformity check — '{x_col}':")
        print(f"  Non-uniform grid detected (max std/median = {max_ratio:.3f} "
              f"> threshold {uniformity_threshold}).")
        print(f"  Resampling to uniform grid: step = {step_used:.5f} {x_col} units.")
        df = resample_to_uniform_grid(df, x_col=x_col, y_col=y_col,
                                      cycle_col=cycle_col, step=step_used,
                                      x_min=x_min, x_max=x_max,
                                      per_cycle_bounds=per_cycle_bounds)
        print(f"  Resampling complete — {len(df)} total grid points across "
              f"{len(cycles)} cycles.")
    else:
        max_ratio = max(ratios) if ratios else 0.0
        print(f"\n  Grid uniformity check — '{x_col}':")
        print(f"  Grid is uniform (max std/median = {max_ratio:.3f} "
              f"<= threshold {uniformity_threshold}). No resampling needed.")

    return df, needs_resample, step_used


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
                    x_col=None,
                    window=SAVGOL_WINDOW, polyorder=SAVGOL_POLYORDER,
                    spline_s=SPLINE_SMOOTHING,
                    edge_trim_pct=EDGE_TRIM_PCT,
                    prompt_edge_trim=True):
    """
    Interactive smoothing routine supporting Savitzky-Golay or smoothing spline.
    Generic — no knowledge of which derivative follows or what thresholds mean.
    Threshold decisions (dv_threshold, dq_threshold) are handled by the calling
    analysis script (dva_analysis.py, ica_analysis.py), not here.

    - x_col: column to use as the x-axis in inspection plots. Pass the capacity
      column for DVA voltage smoothing, or the voltage column for ICA dQ/dV
      smoothing. Defaults to df.columns[1] for backwards compatibility.
    - Prompts user to choose method and confirm/edit smoothing parameters.
    - Edge trimming applied before fitting to exclude end-dropout artifacts.
    - prompt_edge_trim=False suppresses edge trim prompt (use for derivative
      smoothing passes where edges are already NaN from the voltage smooth step).
    - Shows first cycle with zoom/inspect loop, then any additional cycles on request.
    - Outer loop: adjust parameters and repeat, or confirm and retain column.

    Returns (df, edge_trim_pct).
    """
    cycles      = sorted(df[cycle_col].unique())
    x_col       = x_col if x_col is not None else df.columns[1]
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
                    pct = float(input(f"  New trim % (e.g. 1.0, current {edge_trim_pct*100:.1f}%): "))
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