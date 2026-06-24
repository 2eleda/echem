# config_ex.py
# =============================================================================
# EXAMPLE EXPERIMENT CONFIGURATION
# Copy this file into your experiment folder and rename it config.py.
# Edit the values below to match your dataset. All pipeline scripts read
# from this file — nothing else needs to be changed for a new experiment.
#
# Folder setup:
#   your_experiment/
#       config.py           ← this file, renamed
#       run_dva.py          ← copy from practice/ and adjust src path if needed
#       run_ica.py
#       run_lean.py
#       run_combined.py
#       your_data.csv       ← input data
#       processed/          ← created automatically, CSVs saved here
#       plots/              ← created automatically, figures saved here
# =============================================================================

import os

# =============================================================================
# OUTPUT DIRECTORIES
# All outputs (CSVs, plots) are saved relative to this config file's location.
# Folders are created automatically on first run.
# =============================================================================
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(BASE_DIR, "processed")
PLOTS_DIR     = os.path.join(BASE_DIR, "plots")

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,     exist_ok=True)

# =============================================================================
# INPUT DATA
# Path to your raw discharge (or charge) CSV.
# The file should contain one row per data point with columns for cycle number,
# capacity, and voltage. Column names are set below — order in the CSV does
# not matter as long as the names match exactly.
# =============================================================================

CSV_PATH = os.path.join(BASE_DIR, "your_data.csv")   # ← rename to your file

# Column names exactly as they appear in the CSV header
CYCLE_COL    = "cycle_number"    # ← e.g. "Cycle", "cycle_num", "cyc"
CAPACITY_COL = "capacity_mAh"    # ← e.g. "Q_mAh", "Capacity", "cap"
VOLTAGE_COL  = "voltage_V"       # ← e.g. "V", "Voltage", "ewe_V"

# =============================================================================
# DATA DIRECTION
# Set IS_DISCHARGE = True  if your CSV contains discharge data
#                          (voltage decreasing, capacity increasing).
# Set IS_DISCHARGE = False if your CSV contains charge data
#                          (voltage increasing, capacity increasing).
#
# Never pass full charge+discharge cycles — split them upstream first.
# Always run DVA and ICA on the same direction to keep cross-reference valid.
# =============================================================================
IS_DISCHARGE = True

# =============================================================================
# UNIFORM GRID
# The pipeline checks whether the x-axis (capacity for DVA, voltage for ICA)
# is uniformly spaced before computing derivatives. If not, it resamples.
#
# Leave step sizes as None to auto-detect from the data median spacing.
# Set explicitly if you know your target resolution:
#   DVA_GRID_STEP = 0.5    → 0.5 mAh per point (typical for mAh-scale cells)
#   ICA_GRID_STEP = 0.001  → 1 mV per point (standard for NMC/graphite ICA)
#
# UNIFORMITY_THRESHOLD: std/median ratio above which resampling is triggered.
# Default 0.05 (5%) handles most cycler data. Increase only if your instrument
# produces intentionally variable step sizes (e.g. GITT).
# =============================================================================
DVA_GRID_STEP        = None    # mAh — e.g. 0.5; None = auto
ICA_GRID_STEP        = None    # V   — e.g. 0.001; None = auto
UNIFORMITY_THRESHOLD = 0.05

# =============================================================================
# DERIVATIVE THRESHOLDS
# Guards against blow-up if grid resampling is skipped or data is non-standard
# (e.g. GITT, raw non-uniform data). On a uniform resampled grid these will
# never trigger — leave as None unless you have a specific reason to set them.
# =============================================================================
DQ_THRESHOLD = None    # mAh — minimum |dQ| for dV/dQ (DVA). None = disabled.
DV_THRESHOLD = None    # V   — minimum |dV| for dQ/dV (ICA). None = disabled.

# =============================================================================
# FEATURE TRACKING
# Maximum x-position shift allowed when matching a feature between consecutive
# cycles. Features that shift further than this seed a new tracked instance
# rather than extending an existing one.
#
# Set to ~10-20% of the x-axis range for your dataset:
#   DVA: capacity axis in mAh — e.g. 300 for a ~3000 mAh cell
#   ICA: voltage axis in V    — e.g. 0.05 for a 2.5–4.2 V window
# =============================================================================
DVA_MAX_FEATURE_SHIFT = 300     # mAh
ICA_MAX_FEATURE_SHIFT = 0.05    # V

# =============================================================================
# PLOT Y-AXIS LIMITS
# None = autoscale. Set to (ymin, ymax) to clip the y-axis — useful when a
# steep end-of-discharge artefact compresses the bulk of the plot.
# Examples:
#   DVA_YLIM = (-0.001, 0)    clips negative dV/dQ below -0.001
#   ICA_YLIM = (0, 0.005)     clips dQ/dV to the range of real peaks
# =============================================================================
DVA_YLIM = None
ICA_YLIM = None

# =============================================================================
# COMBINED OVERLAY (run_combined.py)
# =============================================================================

# Cycles to highlight in the cross-reference overlay. None = all cycles.
CYCLES_TO_PLOT = None    # e.g. [0, 10, 30, 50, 70]

# Point to previously saved processed CSVs to skip smoothing and go straight
# to feature detection on a repeat run. None = run full pipeline from CSV_PATH.
DVA_RESUME_CSV = None    # e.g. os.path.join(PROCESSED_DIR, "dva_processed.csv")
ICA_RESUME_CSV = None    # e.g. os.path.join(PROCESSED_DIR, "ica_processed.csv")

OVERLAY_PLOT_OUT = os.path.join(PLOTS_DIR, "dva_ica_overlay.png")

# =============================================================================
# LEAN
# =============================================================================

# dq_step: fixed capacity increment per sample (mAh).
# Only used when current and time columns are absent from the CSV.
# Should match the capacity resolution of your resampled grid.
LEAN_DQ_STEP  = 0.5      # mAh

# bin_width: voltage bin width ΔV in volts.
# Must be >= your voltage noise floor. Larger = smoother, lower resolution.
# Rule of thumb: >= 5 raw samples per peak half-width.
LEAN_BIN_WIDTH = 0.001   # V (1 mV — standard starting point)

# Optional current and time column names for I·dt charge computation.
# Set to None if not present in the CSV.
CURRENT_COL = None    # e.g. "current_A"
TIME_COL    = None    # e.g. "time_s"