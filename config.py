# config.py
import os
from pathlib import Path

# --- Core Paths ---
PROJECT_ROOT = Path(__file__).parent
DB_DIR = PROJECT_ROOT / 'database'
REPORTS_DIR = PROJECT_ROOT / 'reports'
HEATMAPS_DIR = REPORTS_DIR / 'heatmaps'
LINE_CHARTS_DIR = REPORTS_DIR / 'line_charts'
COMBINED_CHARTS_DIR = REPORTS_DIR / 'combined_charts'
LOG_DIR = PROJECT_ROOT / 'logs'
INDICATOR_PARAMS_PATH = PROJECT_ROOT / 'indicator_params.json'

# --- Database ---
DB_NAME_TEMPLATE = "{symbol}_{timeframe}.db"

# --- Analysis Parameters ---
DEFAULT_MAX_LAG = 50 # Default max lag if not calculable
MIN_DATA_POINTS_FOR_LAG = 51 # Minimum data points needed beyond max_lag
# >>> MODIFIED: Target for correlation calculations <<<
TARGET_MAX_CORRELATIONS = 30000
# >>> END MODIFIED <<<

# --- Visualization ---
# >>> MODIFIED: Max indicators/configs to show on heatmap/generate via optimizer <<<
HEATMAP_MAX_CONFIGS = 50
# >>> END MODIFIED <<<
PLOT_DPI = 300

# --- Parameter Optimization ---
OPTIMIZER_ITERATIONS = 100 # Number of iterations for the optimizer in tweak mode
OPTIMIZER_CANDIDATES_PER_ITERATION = 2 # How many new candidates to generate each iteration
OPTIMIZER_RANDOM_EXPLORE_PROB = 0.1 # Probability of generating a completely random candidate vs. perturbing the best

# --- Ensure Directories Exist ---
DB_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
HEATMAPS_DIR.mkdir(exist_ok=True)
LINE_CHARTS_DIR.mkdir(exist_ok=True)
COMBINED_CHARTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)