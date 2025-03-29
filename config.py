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
# >>> Leaderboard DB Path <<<
LEADERBOARD_DB_PATH = PROJECT_ROOT / 'correlation_leaderboard.db'
# >>> END <<<

# --- Database ---
DB_NAME_TEMPLATE = "{symbol}_{timeframe}.db"

# --- Analysis Parameters ---
DEFAULT_MAX_LAG = 300 # Default max lag if not specified by user
MIN_DATA_POINTS_FOR_LAG = 51 # Minimum data points needed beyond max_lag for reliable calculation
TARGET_MAX_CORRELATIONS = 30000

# --- Visualization ---
HEATMAP_MAX_CONFIGS = 50
PLOT_DPI = 300

# --- Parameter Optimization ---
OPTIMIZER_ITERATIONS = 100 # Number of *new* evaluations to attempt
OPTIMIZER_CANDIDATES_PER_ITERATION = 2 # How many new candidates to generate each iteration
OPTIMIZER_RANDOM_EXPLORE_PROB = 0.15 # Probability of generating a completely random candidate vs. perturbing the best
# >>> UPDATED OPTIONS <<<
OPTIMIZER_SCORING_METHOD = 'max_abs' # Options: 'max_abs', 'mean_abs', 'max_positive', 'max_negative', 'mean_positive', 'mean_negative'
# >>> END UPDATED <<<

# --- Ensure Directories Exist ---
DB_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)
HEATMAPS_DIR.mkdir(exist_ok=True)
LINE_CHARTS_DIR.mkdir(exist_ok=True)
COMBINED_CHARTS_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
# Leaderboard directory (project root is default, no need to create separate dir unless specified)
# LEADERBOARD_DB_PATH.parent.mkdir(exist_ok=True)