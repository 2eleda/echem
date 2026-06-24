# ica.py
from differential_functions import (apply_smoothing, compute_dqdv,
                                     plot_cycles_overlay, ensure_uniform_grid)
from feature_functions import find_features, track_features
from config import (CSV_PATH, CYCLE_COL, CAPACITY_COL, VOLTAGE_COL,
                    ICA_GRID_STEP, UNIFORMITY_THRESHOLD,
                    DV_THRESHOLD, ICA_MAX_FEATURE_SHIFT, ICA_YLIM,
                    IS_DISCHARGE)
import pandas as pd
import os
import json

# =============================================================================
# ICA PIPELINE
# Order of operations per cycle:
#   1. ensure_uniform_grid  — resample voltage axis if non-uniform
#   2. smooth capacity       — SG or spline on capacity vs. voltage
#   3. compute dQ/dV         — derivative on uniform grid
#   4. smooth dQ/dV          — SG or spline on derivative
#   5. find_features         — interactive peak detection
#   6. track_features        — cross-cycle identity matching
#
# Note: capacity (numerator of dQ/dV) is smoothed, not voltage.
# Voltage is the x-axis (denominator) — fixed by resampling, not smoothing.
# =============================================================================

def run_ica(csv_path=CSV_PATH,
            resume_csv=None,
            cycle_col=CYCLE_COL,
            capacity_col=CAPACITY_COL,
            voltage_col=VOLTAGE_COL,
            ica_grid_step=ICA_GRID_STEP,
            uniformity_threshold=UNIFORMITY_THRESHOLD,
            dv_threshold=DV_THRESHOLD,
            max_feature_shift=ICA_MAX_FEATURE_SHIFT,
            ica_ylim=ICA_YLIM,
            is_discharge=IS_DISCHARGE,
            reference_cycle=None,
            interactive=True):
    """
    Full ICA pipeline. Returns (df, feature_df, tracked_df, features,
    cycle_col, capacity_col, voltage_col, csv_out).

    interactive=True  : runs all prompts as normal (standalone mode)
    interactive=False : skips prompts, uses parameter defaults (combined mode)
    """
    if resume_csv is not None:
        if not os.path.exists(resume_csv):
            raise FileNotFoundError(f"resume_csv not found: {resume_csv}")
        df = pd.read_csv(resume_csv)
        print(f"Loaded previously processed ICA dataframe: {resume_csv}")
        for col in ["Capacity_smooth", "dQdV_raw", "dQdV_smooth"]:
            if col not in df.columns:
                raise ValueError(
                    f"Expected column '{col}' not found in {resume_csv}.")
        print("  Smoothed columns confirmed — skipping to feature detection.")
        csv_out    = resume_csv
        ica_negate = is_discharge

    else:
        df = pd.read_csv(csv_path)
        print("ICA — Columns detected:", df.columns.tolist())
        print("ICA — Cycles present:  ",
              [int(c) for c in sorted(df[cycle_col].unique())])

        # ICA: discharge -> resampling sorts voltage ascending, reversing
        #      discharge direction -> dQ/dV negative -> negate=True
        #      charge   -> voltage ascending matches charge direction
        #      -> dQ/dV naturally positive -> negate=False
        ica_negate = is_discharge

        # Step 1 — ensure uniform voltage grid before smoothing
        print(f"\n{'='*60}")
        print("Step 1 — Uniform grid check (voltage axis)")
        df, resampled, step_used = ensure_uniform_grid(
            df,
            x_col=voltage_col,
            y_col=capacity_col,
            cycle_col=cycle_col,
            step=ica_grid_step,
            uniformity_threshold=uniformity_threshold,
            per_cycle_bounds=True   # each cycle resampled on its own voltage
                                    # range — standard for ICA since voltage
                                    # windows differ across cycles as cell ages
        )

        # Step 2 — smooth capacity (numerator of dQ/dV)
        print(f"\n{'='*60}")
        print("Step 2 — Capacity smoothing")
        df, _ = apply_smoothing(
            df, capacity_col, cycle_col, out_col="Capacity_smooth",
            x_col=voltage_col, prompt_edge_trim=True
        )

        # Step 3 — compute raw dQ/dV
        if interactive:
            print(f"\n{'='*60}")
            print(f"Step 3 — dQ/dV computation")
            print(f"  dV threshold: {dv_threshold}  "
                  f"(None = disabled; only needed for non-uniform grids)")
            if input("Edit dV threshold? (yes/no): ").strip().lower() == "yes":
                raw = input("  New threshold in V (or 'none'): ").strip().lower()
                dv_threshold = None if raw == "none" else float(raw)

        df = compute_dqdv(
            df, voltage_col, "Capacity_smooth", cycle_col,
            out_col="dQdV_raw", dv_threshold=dv_threshold,
            negate=ica_negate
        )

        # Step 4 — smooth dQ/dV
        print(f"\n{'='*60}")
        print("Step 4 — dQ/dV smoothing")
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

    # Step 5 — feature detection
    print(f"\n{'='*60}")
    print("Step 5 — ICA peak detection")
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

    # Step 6 — feature tracking
    features, tracked_df = track_features(
        feature_df, cycle_col,
        mode="peak",
        max_shift=max_feature_shift
    )

    tracked_out = csv_out.replace(".csv", "_tracked.csv")
    tracked_df.to_csv(tracked_out, index=False)
    print(f"  Saved ICA tracked features: {tracked_out}")

    # ==========================================================
    # ICA FEATURE EXTRACTION — choose mode
    # ==========================================================
    print(f"\n{'='*60}")
    print("ICA feature extraction")
    print("  1 — Voltage window analysis (robust, works for all cycles)")
    print("  2 — Curve fitting — Gaussian or pseudo-Voigt (best for early cycles)")
    print("  3 — Both")
    print("  0 — Skip")
    mode = ""
    while mode not in ("0", "1", "2", "3"):
        mode = input("Choose mode (0/1/2/3): ").strip()

    window_df = None
    fit_df    = None

    if mode in ("1", "3"):
        # check for existing windows file
        win_path = input(
            "  Load existing windows JSON? (enter path or press Enter to define new): "
        ).strip()
        existing_windows = load_ica_windows(win_path) if win_path else None
        window_df, _ = analyze_ica_windows(
            df, cycle_col, voltage_col, "dQdV_smooth",
            reference_cycle=reference_cycle,
            windows=existing_windows,
            windows_path=win_path if win_path else None
        )

    if mode in ("2", "3"):
        fit_df = analyze_ica_fitting(
            df, cycle_col, voltage_col, "dQdV_smooth",
            reference_cycle=reference_cycle
        )

    return (df, feature_df, tracked_df, features,
            cycle_col, capacity_col, voltage_col, csv_out,
            window_df, fit_df)


# =============================================================================

def analyze_ica_fitting(df, cycle_col, voltage_col, y_col,
                         reference_cycle=None,
                         fit_model=None,
                         r2_threshold=0.95,
                         csv_out=None):
    """
    Mode 2 ICA feature extraction — curve fitting per cycle.

    User chooses Gaussian or pseudo-Voigt peak model. Peaks are identified
    interactively on the reference cycle, then auto-fitted to all cycles.
    Poor fits (R² < r2_threshold) are flagged in the output and offered
    for manual review.

    Pseudo-Voigt is a linear mix of Gaussian and Lorentzian:
        pV(v) = eta * L(v) + (1-eta) * G(v)
    where eta (0–1) controls the mix. eta=0 is pure Gaussian, eta=1 is
    pure Lorentzian. More physically accurate for electrochemical peaks
    which typically have Lorentzian tails from kinetic broadening.

    Gaussian is simpler and sufficient when peaks are well-resolved and
    symmetric — appropriate for early-life cycles at low C-rate.

    Parameters
    ----------
    df             : dataframe with cycle_col, voltage_col, y_col
    cycle_col      : cycle number column
    voltage_col    : voltage column
    y_col          : smoothed dQ/dV column
    reference_cycle: cycle to fit interactively. None = first cycle.
    fit_model      : "gaussian", "pseudo_voigt", or None (prompts user).
    r2_threshold   : R² below which a cycle is flagged as poor fit.
    csv_out        : output CSV path. None = prompt user.

    Returns
    -------
    fit_df : dataframe, one row per (cycle, peak), columns:
             cycle_col, peak_idx, center_V, amplitude, width_V,
             area, eta (pseudo-voigt only), r2, fit_quality
    """
    plt.close("all")
    cycles = sorted(df[cycle_col].unique())
    if reference_cycle is None:
        reference_cycle = cycles[0]

    # --- choose model ---
    if fit_model is None:
        print(f"\n{'='*60}")
        print("ICA curve fitting — choose peak model:")
        print("  gaussian     — symmetric, simpler, good for well-resolved peaks")
        print("  pseudo_voigt — Gaussian + Lorentzian mix, more physically accurate")
        while fit_model not in ("gaussian", "pseudo_voigt"):
            fit_model = input("Model (gaussian / pseudo_voigt): ").strip().lower()

    # --- model functions ---
    def _gaussian(v, center, amplitude, width):
        return amplitude * np.exp(-0.5 * ((v - center) / width) ** 2)

    def _lorentzian(v, center, amplitude, width):
        return amplitude / (1 + ((v - center) / width) ** 2)

    def _pseudo_voigt(v, center, amplitude, width, eta):
        eta = np.clip(eta, 0, 1)
        return (eta * _lorentzian(v, center, amplitude, width) +
                (1 - eta) * _gaussian(v, center, amplitude, width))

    def _multi_gaussian(v, *params):
        # params = [center, amplitude, width] * n_peaks
        y = np.zeros_like(v)
        for i in range(0, len(params), 3):
            y += _gaussian(v, params[i], params[i+1], params[i+2])
        return y

    def _multi_pseudo_voigt(v, *params):
        # params = [center, amplitude, width, eta] * n_peaks
        y = np.zeros_like(v)
        for i in range(0, len(params), 4):
            y += _pseudo_voigt(v, params[i], params[i+1], params[i+2], params[i+3])
        return y

    n_params = 3 if fit_model == "gaussian" else 4

    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # --- interactive fit on reference cycle ---
    ref_sub = df[df[cycle_col] == reference_cycle].dropna(
        subset=[voltage_col, y_col])
    v_ref   = ref_sub[voltage_col].values.astype(float)
    y_ref   = ref_sub[y_col].values.astype(float)

    print(f"\n{'='*60}")
    print(f"Interactive fit — reference cycle {int(reference_cycle)}")
    print(f"  Model: {fit_model}")

    # initial peak detection on reference cycle
    pks, _ = find_peaks(y_ref, prominence=0)
    proms  = np.array([y_ref[p] for p in pks])

    print(f"\n  Prominence histogram shown — set threshold to select real peaks.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(v_ref, y_ref, color="tab:blue", linewidth=1)
    axes[0].set_xlabel(voltage_col); axes[0].set_ylabel(y_col)
    axes[0].set_title(f"Reference cycle {int(reference_cycle)}")
    axes[1].hist(proms, bins=30, color="tab:blue", edgecolor="white")
    axes[1].set_xlabel("Peak amplitude"); axes[1].set_ylabel("Count")
    axes[1].set_title("Peak amplitude histogram")
    fig.tight_layout(); plt.show(); plt.close("all")

    try:
        amp_threshold = float(input("  Minimum peak amplitude threshold: "))
    except ValueError:
        amp_threshold = np.percentile(proms, 50)
        print(f"  Invalid — using median: {amp_threshold:.4e}")

    selected_pks = pks[proms >= amp_threshold]

    # show selected peaks
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(v_ref, y_ref, color="tab:blue", linewidth=1, label=y_col)
    ax.scatter(v_ref[selected_pks], y_ref[selected_pks],
               color="red", s=60, zorder=5, label="Selected peaks")
    for i, pk in enumerate(selected_pks):
        ax.annotate(f"P{i+1}\n{v_ref[pk]:.3f}",
                    (v_ref[pk], y_ref[pk]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=7, ha="center", color="red")
    ax.set_xlabel(voltage_col); ax.set_ylabel(y_col)
    ax.set_title(f"Selected peaks — threshold {amp_threshold:.4e}")
    ax.legend(fontsize=8); fig.tight_layout()
    plt.show(); plt.close("all")

    if input("  Adjust threshold? (yes/no): ").strip().lower() == "yes":
        try:
            amp_threshold = float(input("  New threshold: "))
            selected_pks  = pks[proms >= amp_threshold]
        except ValueError:
            print("  Invalid — keeping current threshold.")

    n_peaks = len(selected_pks)
    print(f"  {n_peaks} peaks selected for fitting.")

    # build initial parameters from reference cycle
    p0 = []
    bounds_lo = []
    bounds_hi = []
    v_range   = v_ref.max() - v_ref.min()

    for pk in selected_pks:
        center    = v_ref[pk]
        amplitude = y_ref[pk]
        width     = 0.01   # 10 mV starting width
        p0.extend([center, amplitude, width])
        bounds_lo.extend([center - 0.05, 0,     0.001])
        bounds_hi.extend([center + 0.05, amplitude * 3, 0.1])
        if fit_model == "pseudo_voigt":
            p0.append(0.5)
            bounds_lo.append(0.0)
            bounds_hi.append(1.0)

    # fit reference cycle
    model_fn = _multi_gaussian if fit_model == "gaussian" else _multi_pseudo_voigt
    while True:
        try:
            popt, _ = curve_fit(model_fn, v_ref, y_ref,
                                 p0=p0,
                                 bounds=(bounds_lo, bounds_hi),
                                 maxfev=10000)
            y_fit = model_fn(v_ref, *popt)
            r2    = _r2(y_ref, y_fit)
        except Exception as e:
            print(f"  Fit failed on reference cycle: {e}")
            popt  = p0
            y_fit = np.zeros_like(y_ref)
            r2    = 0.0

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(v_ref, y_ref,  color="tab:blue",   linewidth=1,  label="Data")
        ax.plot(v_ref, y_fit,  color="tab:orange",  linewidth=1.5,
                linestyle="--", label=f"Fit (R²={r2:.4f})")
        for i in range(n_peaks):
            idx = i * n_params
            pk_params = list(popt[idx:idx+n_params])
            if fit_model == "gaussian":
                y_pk = _gaussian(v_ref, *pk_params)
            else:
                y_pk = _pseudo_voigt(v_ref, *pk_params)
            ax.fill_between(v_ref, y_pk, alpha=0.2,
                            label=f"P{i+1} @ {popt[idx]:.3f} V")
        ax.set_xlabel(voltage_col); ax.set_ylabel(y_col)
        ax.set_title(f"Reference fit — {fit_model}, R²={r2:.4f}")
        ax.legend(fontsize=7); fig.tight_layout()
        plt.show(); plt.close("all")

        if input("  Accept fit? (yes/no): ").strip().lower() == "yes":
            break
        print("  Adjust peak selection or try different threshold.")
        try:
            amp_threshold = float(input("  New amplitude threshold: "))
            selected_pks  = pks[proms >= amp_threshold]
            n_peaks       = len(selected_pks)
            p0 = []; bounds_lo = []; bounds_hi = []
            for pk in selected_pks:
                center = v_ref[pk]; amplitude = y_ref[pk]
                p0.extend([center, amplitude, 0.01])
                bounds_lo.extend([center-0.05, 0, 0.001])
                bounds_hi.extend([center+0.05, amplitude*3, 0.1])
                if fit_model == "pseudo_voigt":
                    p0.append(0.5); bounds_lo.append(0.0); bounds_hi.append(1.0)
        except ValueError:
            print("  Invalid — keeping current parameters.")

    ref_popt = popt  # starting parameters for all-cycle fitting

    # --- auto-fit all cycles ---
    print(f"\nAuto-fitting {len(cycles)} cycles...")
    poor_cycles = []
    records     = []

    for cyc in cycles:
        sub = df[df[cycle_col] == cyc].dropna(subset=[voltage_col, y_col])
        v_c = sub[voltage_col].values.astype(float)
        y_c = sub[y_col].values.astype(float)

        try:
            popt_c, _ = curve_fit(model_fn, v_c, y_c,
                                   p0=ref_popt,
                                   bounds=(bounds_lo, bounds_hi),
                                   maxfev=10000)
            y_fit_c = model_fn(v_c, *popt_c)
            r2_c    = _r2(y_c, y_fit_c)
            quality = "good" if r2_c >= r2_threshold else "poor"
        except Exception:
            popt_c  = [np.nan] * len(ref_popt)
            r2_c    = np.nan
            quality = "poor"

        if quality == "poor":
            poor_cycles.append(cyc)

        for i in range(n_peaks):
            idx    = i * n_params
            center = popt_c[idx]     if not np.isnan(popt_c[idx]) else np.nan
            amp    = popt_c[idx+1]   if len(popt_c) > idx+1 else np.nan
            width  = popt_c[idx+2]   if len(popt_c) > idx+2 else np.nan

            # area = integral of fitted peak
            if not np.isnan(center):
                pk_params = list(popt_c[idx:idx+n_params])
                if fit_model == "gaussian":
                    area = np.trapezoid(_gaussian(v_c, *pk_params), v_c)
                    eta  = np.nan
                else:
                    area = np.trapezoid(_pseudo_voigt(v_c, *pk_params), v_c)
                    eta  = popt_c[idx+3] if len(popt_c) > idx+3 else np.nan
            else:
                area = np.nan
                eta  = np.nan

            row = {
                cycle_col:    int(cyc),
                "peak_idx":   i + 1,
                "center_V":   center,
                "amplitude":  amp,
                "width_V":    width,
                "area":       area,
                "r2":         r2_c,
                "fit_quality": quality,
            }
            if fit_model == "pseudo_voigt":
                row["eta"] = eta
            records.append(row)

    fit_df = pd.DataFrame(records)

    # --- review poor fits ---
    if poor_cycles:
        print(f"\n  {len(poor_cycles)} cycles with poor fit "
              f"(R² < {r2_threshold}): "
              f"{[int(c) for c in poor_cycles]}")
        if input("  Review poor fits? (yes/no): ").strip().lower() == "yes":
            for cyc in poor_cycles:
                sub   = df[df[cycle_col] == cyc].dropna(
                    subset=[voltage_col, y_col])
                v_c   = sub[voltage_col].values.astype(float)
                y_c   = sub[y_col].values.astype(float)
                r2_c  = fit_df[fit_df[cycle_col] == int(cyc)]["r2"].iloc[0]

                cyc_rows = fit_df[fit_df[cycle_col] == int(cyc)]
                y_fit_c  = np.zeros_like(v_c)
                for _, row in cyc_rows.iterrows():
                    if np.isnan(row["center_V"]):
                        continue
                    if fit_model == "gaussian":
                        y_fit_c += _gaussian(
                            v_c, row["center_V"], row["amplitude"], row["width_V"])
                    else:
                        y_fit_c += _pseudo_voigt(
                            v_c, row["center_V"], row["amplitude"],
                            row["width_V"], row["eta"])

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(v_c, y_c,     color="tab:blue",   linewidth=1, label="Data")
                ax.plot(v_c, y_fit_c, color="tab:orange", linewidth=1.5,
                        linestyle="--", label=f"Fit (R²={r2_c:.4f})")
                ax.set_xlabel(voltage_col); ax.set_ylabel(y_col)
                ax.set_title(f"Poor fit — Cycle {int(cyc)}, R²={r2_c:.4f}")
                ax.legend(fontsize=8); fig.tight_layout()
                plt.show(); plt.close("all")
                print(f"  Cycle {int(cyc)} flagged as poor fit — "
                      f"retained in output with fit_quality='poor'.")

    # --- save ---
    if csv_out is None:
        csv_out = input("  ICA fitting results CSV filename: ").strip()
        if not csv_out.endswith(".csv"):
            csv_out += ".csv"
    fit_df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # --- summary plot ---
    good_df = fit_df[fit_df["fit_quality"] == "good"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for pk_idx in fit_df["peak_idx"].unique():
        pk_sub = good_df[good_df["peak_idx"] == pk_idx].dropna(
            subset=["center_V", "amplitude"])
        axes[0].plot(pk_sub[cycle_col], pk_sub["center_V"],
                     marker="o", markersize=4, label=f"Peak {int(pk_idx)}")
        axes[1].plot(pk_sub[cycle_col], pk_sub["amplitude"],
                     marker="o", markersize=4, label=f"Peak {int(pk_idx)}")
    axes[0].set_xlabel(cycle_col); axes[0].set_ylabel("Peak voltage (V)")
    axes[0].set_title("Peak position vs cycle (good fits only)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(cycle_col); axes[1].set_ylabel("Peak amplitude (mAh/V)")
    axes[1].set_title("Peak amplitude vs cycle (good fits only)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    plt.show(); plt.close("all")

    return fit_df
# =============================================================================
# ICA FEATURE EXTRACTION — MODE 1: VOLTAGE WINDOW ANALYSIS
# =============================================================================

def save_ica_windows(windows, path):
    """
    Save named voltage windows to a JSON file for reuse across sessions
    or experiments with the same chemistry.

    windows : dict of {name: {"lo": float, "hi": float}}
    path    : file path to save to (e.g. "ica_windows_nmc532.json")
    """
    with open(path, "w") as f:
        json.dump(windows, f, indent=2)
    print(f"  Saved ICA windows: {path}")


def load_ica_windows(path):
    """
    Load named voltage windows previously saved by save_ica_windows.

    Returns dict of {name: {"lo": float, "hi": float}}, or None if
    file not found.
    """
    if not os.path.exists(path):
        print(f"  Window file not found: {path}")
        return None
    with open(path) as f:
        windows = json.load(f)
    print(f"  Loaded ICA windows: {path}")
    for name, w in windows.items():
        print(f"    {name}: {w['lo']:.4f} — {w['hi']:.4f} V")
    return windows


def analyze_ica_windows(df, cycle_col, voltage_col, y_col,
                         reference_cycle=None,
                         windows=None,
                         windows_path=None,
                         csv_out=None):
    """
    Mode 1 ICA feature extraction — voltage window analysis.

    Shows the multi-cycle overlay so the user can see the full aging
    evolution, then prompts the user to define named voltage windows on
    the reference cycle. For each cycle and each window reports:
      - peak_voltage  : voltage of max dQ/dV within the window
      - peak_amplitude: max dQ/dV within the window
      - integral      : integral of dQ/dV above zero baseline within window

    Windows are saved to JSON for reuse. Existing windows can be passed
    in directly (windows=) or loaded from file (windows_path=) to skip
    the interactive definition step.

    Parameters
    ----------
    df             : dataframe with cycle_col, voltage_col, y_col
    cycle_col      : cycle number column
    voltage_col    : voltage column
    y_col          : smoothed dQ/dV column
    reference_cycle: cycle to use for window definition. None = first cycle.
    windows        : pre-defined windows dict {name: {"lo": float, "hi": float}}.
                     If provided, skips interactive definition.
    windows_path   : path to load/save windows JSON.
                     None = prompt user for path.
    csv_out        : output CSV path. None = prompt user.

    Returns
    -------
    window_df : dataframe, one row per (cycle, window), columns:
                cycle_col, window_name, lo_V, hi_V,
                peak_voltage, peak_amplitude, integral
    windows   : dict of defined windows (for reuse)
    """
    plt.close("all")
    cycles = sorted(df[cycle_col].unique())
    if reference_cycle is None:
        reference_cycle = cycles[0]

    # --- show overlay for context ---
    n      = len(cycles)
    colors = plt.colormaps["plasma"](np.linspace(0, 1, n))
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, cyc in enumerate(cycles):
        sub = df[df[cycle_col] == cyc].dropna(subset=[y_col])
        ax.plot(sub[voltage_col], sub[y_col],
                color=colors[i], linewidth=0.8, alpha=0.8)
    sm = plt.cm.ScalarMappable(
        cmap=plt.colormaps["plasma"],
        norm=plt.Normalize(vmin=int(cycles[0]), vmax=int(cycles[-1])))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label=cycle_col)
    ax.set_xlabel(voltage_col)
    ax.set_ylabel(y_col)
    ax.set_title("ICA overlay — all cycles\n"
                 "Use this to identify voltage window boundaries across aging")
    ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
    fig.tight_layout()
    plt.show()
    plt.close("all")

    # --- load or define windows ---
    if windows is None and windows_path is not None:
        windows = load_ica_windows(windows_path)

    if windows is None:
        # interactive window definition on reference cycle
        ref_sub = df[df[cycle_col] == reference_cycle].dropna(subset=[y_col])
        windows = {}
        print(f"\n{'='*60}")
        print(f"Define voltage windows on reference cycle {int(reference_cycle)}")
        print(f"  Enter window name and voltage bounds.")
        print(f"  Type 'done' as window name when finished.\n")

        while True:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(ref_sub[voltage_col], ref_sub[y_col],
                    color="tab:blue", linewidth=1)
            ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
            for name, w in windows.items():
                ax.axvspan(w["lo"], w["hi"], alpha=0.2, label=name)
                ax.text((w["lo"] + w["hi"]) / 2,
                        ref_sub[y_col].max() * 0.95,
                        name, ha="center", fontsize=8)
            ax.set_xlabel(voltage_col)
            ax.set_ylabel(y_col)
            ax.set_title(f"Reference cycle {int(reference_cycle)} — define windows")
            if windows:
                ax.legend(fontsize=8)
            fig.tight_layout()
            plt.show()
            plt.close("all")

            name = input("  Window name (or 'done'): ").strip()
            if name.lower() == "done":
                if not windows:
                    print("  No windows defined — enter at least one.")
                    continue
                break
            try:
                lo = float(input(f"  '{name}' lower bound (V): "))
                hi = float(input(f"  '{name}' upper bound (V): "))
                if hi <= lo:
                    print("  Upper bound must be > lower bound — try again.")
                    continue
                windows[name] = {"lo": lo, "hi": hi}
                print(f"  Added: {name} = [{lo:.4f}, {hi:.4f}] V")
            except ValueError:
                print("  Invalid — try again.")

        # save windows
        if windows_path is None:
            windows_path = input(
                "  Save windows to JSON (e.g. ica_windows.json): ").strip()
            if not windows_path.endswith(".json"):
                windows_path += ".json"
        save_ica_windows(windows, windows_path)

    # --- compute per-cycle per-window metrics ---
    print(f"\nComputing window metrics across {len(cycles)} cycles...")
    records = []
    for cyc in cycles:
        sub = df[df[cycle_col] == cyc].dropna(subset=[voltage_col, y_col])
        v   = sub[voltage_col].values
        y   = sub[y_col].values

        for name, w in windows.items():
            mask = (v >= w["lo"]) & (v <= w["hi"])
            if mask.sum() == 0:
                records.append({
                    cycle_col:        int(cyc),
                    "window_name":    name,
                    "lo_V":           w["lo"],
                    "hi_V":           w["hi"],
                    "peak_voltage":   np.nan,
                    "peak_amplitude": np.nan,
                    "integral":       np.nan,
                })
                continue

            v_win = v[mask]
            y_win = y[mask]

            peak_idx      = np.argmax(y_win)
            peak_voltage  = v_win[peak_idx]
            peak_amplitude = y_win[peak_idx]

            # integrate above zero baseline
            y_above_zero = np.maximum(y_win, 0)
            integral     = np.trapezoid(y_above_zero, v_win)

            records.append({
                cycle_col:        int(cyc),
                "window_name":    name,
                "lo_V":           w["lo"],
                "hi_V":           w["hi"],
                "peak_voltage":   peak_voltage,
                "peak_amplitude": peak_amplitude,
                "integral":       integral,
            })

    window_df = pd.DataFrame(records)

    # --- save ---
    if csv_out is None:
        csv_out = input("  ICA window analysis CSV filename: ").strip()
        if not csv_out.endswith(".csv"):
            csv_out += ".csv"
    window_df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # --- summary plot: peak amplitude and integral per window across cycles ---
    window_names = list(windows.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name in window_names:
        sub = window_df[window_df["window_name"] == name].dropna(
            subset=["peak_amplitude", "integral"])
        axes[0].plot(sub[cycle_col], sub["peak_amplitude"],
                     marker="o", markersize=4, label=name)
        axes[1].plot(sub[cycle_col], sub["integral"],
                     marker="o", markersize=4, label=name)
    axes[0].set_xlabel(cycle_col)
    axes[0].set_ylabel("Peak dQ/dV (mAh/V)")
    axes[0].set_title("Peak amplitude per window")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(cycle_col)
    axes[1].set_ylabel("Integral (mAh)")
    axes[1].set_title("Window integral (above zero baseline)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    plt.close("all")

    return window_df, windows


# =============================================================================
# ICA FEATURE EXTRACTION — MODE 2: CURVE FITTING
# =============================================================================

def analyze_ica_fitting(df, cycle_col, voltage_col, y_col,
                         reference_cycle=None,
                         fit_model=None,
                         r2_threshold=0.95,
                         csv_out=None):
    """
    Mode 2 ICA feature extraction — curve fitting per cycle.

    User chooses Gaussian or pseudo-Voigt peak model. Peaks are identified
    interactively on the reference cycle, then auto-fitted to all cycles.
    Poor fits (R² < r2_threshold) are flagged in the output and offered
    for manual review.

    Pseudo-Voigt is a linear mix of Gaussian and Lorentzian:
        pV(v) = eta * L(v) + (1-eta) * G(v)
    where eta (0–1) controls the mix. eta=0 is pure Gaussian, eta=1 is
    pure Lorentzian. More physically accurate for electrochemical peaks
    which typically have Lorentzian tails from kinetic broadening.

    Gaussian is simpler and sufficient when peaks are well-resolved and
    symmetric — appropriate for early-life cycles at low C-rate.

    Parameters
    ----------
    df             : dataframe with cycle_col, voltage_col, y_col
    cycle_col      : cycle number column
    voltage_col    : voltage column
    y_col          : smoothed dQ/dV column
    reference_cycle: cycle to fit interactively. None = first cycle.
    fit_model      : "gaussian", "pseudo_voigt", or None (prompts user).
    r2_threshold   : R² below which a cycle is flagged as poor fit.
    csv_out        : output CSV path. None = prompt user.

    Returns
    -------
    fit_df : dataframe, one row per (cycle, peak), columns:
             cycle_col, peak_idx, center_V, amplitude, width_V,
             area, eta (pseudo-voigt only), r2, fit_quality
    """
    plt.close("all")
    cycles = sorted(df[cycle_col].unique())
    if reference_cycle is None:
        reference_cycle = cycles[0]

    # --- choose model ---
    if fit_model is None:
        print(f"\n{'='*60}")
        print("ICA curve fitting — choose peak model:")
        print("  gaussian     — symmetric, simpler, good for well-resolved peaks")
        print("  pseudo_voigt — Gaussian + Lorentzian mix, more physically accurate")
        while fit_model not in ("gaussian", "pseudo_voigt"):
            fit_model = input("Model (gaussian / pseudo_voigt): ").strip().lower()

    # --- model functions ---
    def _gaussian(v, center, amplitude, width):
        return amplitude * np.exp(-0.5 * ((v - center) / width) ** 2)

    def _lorentzian(v, center, amplitude, width):
        return amplitude / (1 + ((v - center) / width) ** 2)

    def _pseudo_voigt(v, center, amplitude, width, eta):
        eta = np.clip(eta, 0, 1)
        return (eta * _lorentzian(v, center, amplitude, width) +
                (1 - eta) * _gaussian(v, center, amplitude, width))

    def _multi_gaussian(v, *params):
        # params = [center, amplitude, width] * n_peaks
        y = np.zeros_like(v)
        for i in range(0, len(params), 3):
            y += _gaussian(v, params[i], params[i+1], params[i+2])
        return y

    def _multi_pseudo_voigt(v, *params):
        # params = [center, amplitude, width, eta] * n_peaks
        y = np.zeros_like(v)
        for i in range(0, len(params), 4):
            y += _pseudo_voigt(v, params[i], params[i+1], params[i+2], params[i+3])
        return y

    n_params = 3 if fit_model == "gaussian" else 4

    def _r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # --- interactive fit on reference cycle ---
    ref_sub = df[df[cycle_col] == reference_cycle].dropna(
        subset=[voltage_col, y_col])
    v_ref   = ref_sub[voltage_col].values.astype(float)
    y_ref   = ref_sub[y_col].values.astype(float)

    print(f"\n{'='*60}")
    print(f"Interactive fit — reference cycle {int(reference_cycle)}")
    print(f"  Model: {fit_model}")

    # initial peak detection on reference cycle
    pks, _ = find_peaks(y_ref, prominence=0)
    proms  = np.array([y_ref[p] for p in pks])

    print(f"\n  Prominence histogram shown — set threshold to select real peaks.")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(v_ref, y_ref, color="tab:blue", linewidth=1)
    axes[0].set_xlabel(voltage_col); axes[0].set_ylabel(y_col)
    axes[0].set_title(f"Reference cycle {int(reference_cycle)}")
    axes[1].hist(proms, bins=30, color="tab:blue", edgecolor="white")
    axes[1].set_xlabel("Peak amplitude"); axes[1].set_ylabel("Count")
    axes[1].set_title("Peak amplitude histogram")
    fig.tight_layout(); plt.show(); plt.close("all")

    try:
        amp_threshold = float(input("  Minimum peak amplitude threshold: "))
    except ValueError:
        amp_threshold = np.percentile(proms, 50)
        print(f"  Invalid — using median: {amp_threshold:.4e}")

    selected_pks = pks[proms >= amp_threshold]

    # show selected peaks
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(v_ref, y_ref, color="tab:blue", linewidth=1, label=y_col)
    ax.scatter(v_ref[selected_pks], y_ref[selected_pks],
               color="red", s=60, zorder=5, label="Selected peaks")
    for i, pk in enumerate(selected_pks):
        ax.annotate(f"P{i+1}\n{v_ref[pk]:.3f}",
                    (v_ref[pk], y_ref[pk]),
                    textcoords="offset points", xytext=(0, 10),
                    fontsize=7, ha="center", color="red")
    ax.set_xlabel(voltage_col); ax.set_ylabel(y_col)
    ax.set_title(f"Selected peaks — threshold {amp_threshold:.4e}")
    ax.legend(fontsize=8); fig.tight_layout()
    plt.show(); plt.close("all")

    if input("  Adjust threshold? (yes/no): ").strip().lower() == "yes":
        try:
            amp_threshold = float(input("  New threshold: "))
            selected_pks  = pks[proms >= amp_threshold]
        except ValueError:
            print("  Invalid — keeping current threshold.")

    n_peaks = len(selected_pks)
    print(f"  {n_peaks} peaks selected for fitting.")

    # build initial parameters from reference cycle
    p0 = []
    bounds_lo = []
    bounds_hi = []
    v_range   = v_ref.max() - v_ref.min()

    for pk in selected_pks:
        center    = v_ref[pk]
        amplitude = y_ref[pk]
        width     = 0.01   # 10 mV starting width
        p0.extend([center, amplitude, width])
        bounds_lo.extend([center - 0.05, 0,     0.001])
        bounds_hi.extend([center + 0.05, amplitude * 3, 0.1])
        if fit_model == "pseudo_voigt":
            p0.append(0.5)
            bounds_lo.append(0.0)
            bounds_hi.append(1.0)

    # fit reference cycle
    model_fn = _multi_gaussian if fit_model == "gaussian" else _multi_pseudo_voigt
    while True:
        try:
            popt, _ = curve_fit(model_fn, v_ref, y_ref,
                                 p0=p0,
                                 bounds=(bounds_lo, bounds_hi),
                                 maxfev=10000)
            y_fit = model_fn(v_ref, *popt)
            r2    = _r2(y_ref, y_fit)
        except Exception as e:
            print(f"  Fit failed on reference cycle: {e}")
            popt  = p0
            y_fit = np.zeros_like(y_ref)
            r2    = 0.0

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(v_ref, y_ref,  color="tab:blue",   linewidth=1,  label="Data")
        ax.plot(v_ref, y_fit,  color="tab:orange",  linewidth=1.5,
                linestyle="--", label=f"Fit (R²={r2:.4f})")
        for i in range(n_peaks):
            idx = i * n_params
            pk_params = list(popt[idx:idx+n_params])
            if fit_model == "gaussian":
                y_pk = _gaussian(v_ref, *pk_params)
            else:
                y_pk = _pseudo_voigt(v_ref, *pk_params)
            ax.fill_between(v_ref, y_pk, alpha=0.2,
                            label=f"P{i+1} @ {popt[idx]:.3f} V")
        ax.set_xlabel(voltage_col); ax.set_ylabel(y_col)
        ax.set_title(f"Reference fit — {fit_model}, R²={r2:.4f}")
        ax.legend(fontsize=7); fig.tight_layout()
        plt.show(); plt.close("all")

        if input("  Accept fit? (yes/no): ").strip().lower() == "yes":
            break
        print("  Adjust peak selection or try different threshold.")
        try:
            amp_threshold = float(input("  New amplitude threshold: "))
            selected_pks  = pks[proms >= amp_threshold]
            n_peaks       = len(selected_pks)
            p0 = []; bounds_lo = []; bounds_hi = []
            for pk in selected_pks:
                center = v_ref[pk]; amplitude = y_ref[pk]
                p0.extend([center, amplitude, 0.01])
                bounds_lo.extend([center-0.05, 0, 0.001])
                bounds_hi.extend([center+0.05, amplitude*3, 0.1])
                if fit_model == "pseudo_voigt":
                    p0.append(0.5); bounds_lo.append(0.0); bounds_hi.append(1.0)
        except ValueError:
            print("  Invalid — keeping current parameters.")

    ref_popt = popt  # starting parameters for all-cycle fitting

    # --- auto-fit all cycles ---
    print(f"\nAuto-fitting {len(cycles)} cycles...")
    poor_cycles = []
    records     = []

    for cyc in cycles:
        sub = df[df[cycle_col] == cyc].dropna(subset=[voltage_col, y_col])
        v_c = sub[voltage_col].values.astype(float)
        y_c = sub[y_col].values.astype(float)

        try:
            popt_c, _ = curve_fit(model_fn, v_c, y_c,
                                   p0=ref_popt,
                                   bounds=(bounds_lo, bounds_hi),
                                   maxfev=10000)
            y_fit_c = model_fn(v_c, *popt_c)
            r2_c    = _r2(y_c, y_fit_c)
            quality = "good" if r2_c >= r2_threshold else "poor"
        except Exception:
            popt_c  = [np.nan] * len(ref_popt)
            r2_c    = np.nan
            quality = "poor"

        if quality == "poor":
            poor_cycles.append(cyc)

        for i in range(n_peaks):
            idx    = i * n_params
            center = popt_c[idx]     if not np.isnan(popt_c[idx]) else np.nan
            amp    = popt_c[idx+1]   if len(popt_c) > idx+1 else np.nan
            width  = popt_c[idx+2]   if len(popt_c) > idx+2 else np.nan

            # area = integral of fitted peak
            if not np.isnan(center):
                pk_params = list(popt_c[idx:idx+n_params])
                if fit_model == "gaussian":
                    area = np.trapezoid(_gaussian(v_c, *pk_params), v_c)
                    eta  = np.nan
                else:
                    area = np.trapezoid(_pseudo_voigt(v_c, *pk_params), v_c)
                    eta  = popt_c[idx+3] if len(popt_c) > idx+3 else np.nan
            else:
                area = np.nan
                eta  = np.nan

            row = {
                cycle_col:    int(cyc),
                "peak_idx":   i + 1,
                "center_V":   center,
                "amplitude":  amp,
                "width_V":    width,
                "area":       area,
                "r2":         r2_c,
                "fit_quality": quality,
            }
            if fit_model == "pseudo_voigt":
                row["eta"] = eta
            records.append(row)

    fit_df = pd.DataFrame(records)

    # --- review poor fits ---
    if poor_cycles:
        print(f"\n  {len(poor_cycles)} cycles with poor fit "
              f"(R² < {r2_threshold}): "
              f"{[int(c) for c in poor_cycles]}")
        if input("  Review poor fits? (yes/no): ").strip().lower() == "yes":
            for cyc in poor_cycles:
                sub   = df[df[cycle_col] == cyc].dropna(
                    subset=[voltage_col, y_col])
                v_c   = sub[voltage_col].values.astype(float)
                y_c   = sub[y_col].values.astype(float)
                r2_c  = fit_df[fit_df[cycle_col] == int(cyc)]["r2"].iloc[0]

                cyc_rows = fit_df[fit_df[cycle_col] == int(cyc)]
                y_fit_c  = np.zeros_like(v_c)
                for _, row in cyc_rows.iterrows():
                    if np.isnan(row["center_V"]):
                        continue
                    if fit_model == "gaussian":
                        y_fit_c += _gaussian(
                            v_c, row["center_V"], row["amplitude"], row["width_V"])
                    else:
                        y_fit_c += _pseudo_voigt(
                            v_c, row["center_V"], row["amplitude"],
                            row["width_V"], row["eta"])

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(v_c, y_c,     color="tab:blue",   linewidth=1, label="Data")
                ax.plot(v_c, y_fit_c, color="tab:orange", linewidth=1.5,
                        linestyle="--", label=f"Fit (R²={r2_c:.4f})")
                ax.set_xlabel(voltage_col); ax.set_ylabel(y_col)
                ax.set_title(f"Poor fit — Cycle {int(cyc)}, R²={r2_c:.4f}")
                ax.legend(fontsize=8); fig.tight_layout()
                plt.show(); plt.close("all")
                print(f"  Cycle {int(cyc)} flagged as poor fit — "
                      f"retained in output with fit_quality='poor'.")

    # --- save ---
    if csv_out is None:
        csv_out = input("  ICA fitting results CSV filename: ").strip()
        if not csv_out.endswith(".csv"):
            csv_out += ".csv"
    fit_df.to_csv(csv_out, index=False)
    print(f"  Saved: {csv_out}")

    # --- summary plot ---
    good_df = fit_df[fit_df["fit_quality"] == "good"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for pk_idx in fit_df["peak_idx"].unique():
        pk_sub = good_df[good_df["peak_idx"] == pk_idx].dropna(
            subset=["center_V", "amplitude"])
        axes[0].plot(pk_sub[cycle_col], pk_sub["center_V"],
                     marker="o", markersize=4, label=f"Peak {int(pk_idx)}")
        axes[1].plot(pk_sub[cycle_col], pk_sub["amplitude"],
                     marker="o", markersize=4, label=f"Peak {int(pk_idx)}")
    axes[0].set_xlabel(cycle_col); axes[0].set_ylabel("Peak voltage (V)")
    axes[0].set_title("Peak position vs cycle (good fits only)")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel(cycle_col); axes[1].set_ylabel("Peak amplitude (mAh/V)")
    axes[1].set_title("Peak amplitude vs cycle (good fits only)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    plt.show(); plt.close("all")

    return fit_df

# =============================================================================
# STANDALONE ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_ica()