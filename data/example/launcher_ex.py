# launcher_ex.py
# =============================================================================
# EXAMPLE LAUNCHER — HOW TO SET UP A NEW EXPERIMENT
# =============================================================================
#
# QUICK START
# -----------
# 1. Create a folder for your experiment anywhere in the project, e.g.:
#       practice/my_experiment/
#
# 2. Copy config_ex.py into that folder and rename it config.py:
#       practice/my_experiment/config.py
#
# 3. Edit config.py:
#       - Set CSV_PATH to your data file
#       - Set CYCLE_COL, CAPACITY_COL, VOLTAGE_COL to match your CSV headers
#       - Set IS_DISCHARGE = True or False
#       - Adjust any other parameters as needed (rest can stay as defaults)
#
# 4. Copy this file into your experiment folder, rename it (e.g. run_dva.py),
#    and change the SRC_DIR path and the pipeline function to match.
#
# 5. Run from anywhere:
#       python practice/my_experiment/run_dva.py
#    or from inside the experiment folder:
#       cd practice/my_experiment
#       python run_dva.py
#
# FOLDER STRUCTURE
# ----------------
# After setup your experiment folder should look like:
#
#   practice/my_experiment/
#       config.py           ← copied and edited from config_ex.py
#       run_dva.py          ← copied and edited from this file
#       run_ica.py          ← same pattern, call run_ica()
#       run_lean.py         ← same pattern, call run_lean()
#       run_combined.py     ← same pattern, call the combined script
#       your_data.csv       ← input data
#       processed/          ← created automatically on first run
#       plots/              ← created automatically on first run
#
# The processed/ and plots/ folders are created automatically by config.py
# the first time any pipeline script is run. All CSVs and figures are saved
# there, keeping your experiment folder organised.
#
# SOURCE PATH
# -----------
# SRC_DIR points from this launcher file up to the src/ folder.
# The default below assumes this structure:
#
#   echem/
#   ├── src/                ← pipeline scripts live here
#   └── practice/
#       └── my_experiment/  ← this launcher lives here
#           └── run_dva.py
#
# If your experiment folder is at a different depth, adjust the number of
# os.path.join('..') steps accordingly:
#   one level deep from src/:   os.path.join(HERE, '..', 'src')
#   two levels deep (default):  os.path.join(HERE, '..', '..', 'src')
#   three levels deep:          os.path.join(HERE, '..', '..', '..', 'src')
#
# =============================================================================

import sys
import os

# path to this file's directory — do not change this line
HERE = os.path.dirname(os.path.abspath(__file__))

# ↓ adjust this if your experiment folder is at a different depth from src/
SRC_DIR = os.path.join(HERE, '..', '..', 'src')

# add src/ to path so pipeline scripts can be imported
sys.path.insert(0, SRC_DIR)

# add this folder to path so config.py is importable
sys.path.insert(0, HERE)

# =============================================================================
# RUN A PIPELINE
# Replace run_dva with run_ica, run_lean, or the combined script as needed.
# =============================================================================

from dva import run_dva    # ← change to run_ica, run_lean, etc.

if __name__ == "__main__":
    run_dva()              # ← change to match the import above