import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences

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
    a fixed horizontal baseline fitted from the reference cycle.

    Generic — covers all three analysis types:
      DVA  : mode="trough", y_col="dVdQ_smooth", x_col=capacity_col
      ICA  : mode="peak",   y_col="dQdV_smooth", x_col=voltage_col
      LEAN : mode="peak",   y_col="dQdV_lean",   x_col=voltage_col

    Baseline
    --------
    A single fixed baseline level is computed once from the reference cycle
    (global max for troughs, global min for peaks, clipped to 0 if the curve
    crosses zero). This fixed value is reused for all cycles, giving area an
    absolute meaning across cycles — essential for LAM quantification.

    Workflow
    --------
    1. Display prominence histogram for a reference cycle.
    2. Prompt user to set prominence threshold.
    3. Display detected features overlaid on the curve with baselines
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
    plt.close("all")   # clear any figures left open from previous pipeline steps

    if mode not in ("peak", "trough"):
        raise ValueError("mode must be 'peak' or 'trough'")

    sign     = 1 if mode == "peak" else -1
    label    = "Peak"   if mode == "peak" else "Trough"
    label_pl = "Peaks"  if mode == "peak" else "Troughs"

    cycles = sorted(df[cycle_col].unique())
    if reference_cycle is None:
        reference_cycle = cycles[0]

    def _find_shoulders(y, peak_idx, exclusion_zone=5):
        """
        Find the nearest inflection points (zero crossings of dy) on either
        side of a trough/peak as shoulder points for local baseline fitting.

        Clips extreme dy values (>99th percentile absolute) before finding sign
        changes — suppresses end-of-discharge spike artifacts that would
        otherwise create spurious critical points far from the feature.

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

        lb = int(left_candidates[-1])  if len(left_candidates)  > 0 else 0
        rb = int(right_candidates[0])  if len(right_candidates) > 0 else n - 1
        return lb, rb



    def _compute_fixed_baseline(cyc):
        """
        Compute a single fixed baseline level from the reference cycle.
        Trough mode: global max of the curve, clipped to 0 if positive.
        Peak  mode: global min of the curve, clipped to 0 if negative.
        This value is reused for all cycles so that area has a consistent
        absolute meaning across cycles — required for LAM quantification.
        """
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
            baseline_arr = np.full(len(x_c), fixed_baseline)
            ax.fill_between(x_c, y_c, baseline_arr,
                            where=((np.arange(len(x_c)) >= _find_shoulders(y_c, pk)[0]) &
                                   (np.arange(len(x_c)) <= _find_shoulders(y_c, pk)[1])),
                            alpha=0.25, color="tab:orange")
            lb, rb = _find_shoulders(y_c, pk)
            ax.plot(x_c[lb:rb+1], baseline_arr[lb:rb+1],
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

        baseline_arr = np.full(len(x_c), fixed_baseline)

        for i, pk in enumerate(pks):
            lb, rb          = _find_shoulders(y_c, pk)
            baseline_at_ext = fixed_baseline
            depth_or_height = abs(y_c[pk] - baseline_at_ext)
            area            = np.abs(np.trapezoid(
                                y_c[lb:rb+1] - baseline_arr[lb:rb+1],
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
                           exclusion_zone=5):
    """
    Debug helper to visualise critical points (dy = 0 sign changes) on a
    single cycle's differential curve alongside detected features.

    Plots two panels:
      Left   — the raw y curve with all critical points marked (grey dots),
                the detected feature extrema (red dots), and the shoulder
                pairs selected for each feature (orange vertical lines).
      Right  — the first derivative dy with zero line, showing where sign
               changes occur.

    A table of every critical point index, x position, and shoulder assignment
    is printed to the terminal.

    Parameters
    ----------
    df                  : dataframe with cycle_col, x_col, y_col
    cycle_col           : cycle number column
    x_col               : x-axis column
    y_col               : smoothed differential column
    mode                : "trough" or "peak"
    cycle               : which cycle to debug. None = first cycle.
    prominence_threshold: feature detection threshold (use same value as
                          find_features to see the same features)
    exclusion_zone      : same value used in find_features (default 5)
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
                 f"exclusion_zone={exclusion_zone}",
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

    def __init__(self, mode, seed_cycle, seed_record, feature_id):
        self.feature_id = feature_id
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
    cycles   = sorted(feature_df[cycle_col].unique())
    features = []   # list of DifferentialFeature instances
    next_id  = 1    # local counter — avoids class-level state

    for cyc in cycles:
        cyc_rows = feature_df[feature_df[cycle_col] == cyc].to_dict("records")

        if not features:
            # first cycle — seed one instance per detected feature
            for row in cyc_rows:
                features.append(DifferentialFeature(mode, cyc, row, next_id))
                next_id += 1
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
                    features.append(DifferentialFeature(mode, cyc, row, next_id))
                    next_id += 1
            else:
                features.append(DifferentialFeature(mode, cyc, row, next_id))
                next_id += 1

    # compile tracked_df
    all_dfs   = [f.to_dataframe() for f in features]
    tracked_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()

    print(f"\nFeature tracking complete:")
    for feat in features:
        print(f"  {feat}")

    return features, tracked_df