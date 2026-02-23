"""
Configuration module for the Medical Survival Analysis Pipeline.
Full Dataset Version (No Split).
"""

from pathlib import Path

# =============================================================================
# 1. PATH MANAGEMENT
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data"
LOGS_PATH = BASE_DIR / "logs"

LOGS_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# 2. DATA FILES
# =============================================================================
# Le fichier unique sur lequel on travaille
SOURCE_FILE = "lung_cancer.csv"

# =============================================================================
# 3. SCHEMA OVERRIDES (Optional)
# =============================================================================
OVERRIDE_SCHEMA = {
    "duration_col": None,
    "event_col": None,
    "numerical_cols": None,
    "categorical_cols": None,
    "dropped_cols": None,
}

# =============================================================================
# 4. GLOBAL PARAMETERS
# =============================================================================
RANDOM_SEED = 42
CATEGORICAL_THRESHOLD_RATIO = 0.05
CATEGORICAL_THRESHOLD_COUNT = 10
