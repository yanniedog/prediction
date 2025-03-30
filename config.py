# config.py
import os
from pathlib import Path

# --- Core Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent # Use resolve() for absolute path
DB_DIR = PROJECT_ROOT / 'database'
REPORTS_DIR = PROJECT_ROOT / 'reports'
HEATMAPS_DIR = REPORTS_DIR / 'heatmaps'
LINE_CHARTS_DIR = REPORTS_DIR / 'line_charts'
COMBINED_CHARTS_DIR = REPORTS_DIR / 'combined_charts'
LOG_DIR = PROJECT_ROOT / 'logs'
INDICATOR_PARAMS_PATH = PROJECT_ROOT / 'indicator_params.json'
LEADERBOARD_DB_PATH = PROJECT_ROOT / 'correlation_leaderboard.db' # Now in root

# --- Database ---
DB_NAME_TEMPLATE = "{symbol}_{timeframe}.db"

# === Modifiable Defaults ===
# User can change these values directly
DEFAULTS = {
    # --- Data Source ---
    "symbol": "BTCUSDT",           # Default symbol for download prompt
    "timeframe": "1d",             # Default timeframe for download prompt

    # --- Analysis Parameters ---
    "max_lag": 7,                 # Default max correlation lag if user doesn't specify
    "min_data_points_for_lag": 51, # Minimum data points needed beyond max_lag for reliable calculation
    "target_max_correlations": 50000, # Target limit for estimated correlations (triggers warning)
    "default_param_range_steps": 5,   # +/- steps around default value for Default Path param generation
                                      # Set higher for more exploration, lower for speed. (Used if not in indicator_params.json)

    # --- Regression/Prediction ---
    "min_regression_points": 30,  # Minimum historical (Indicator[t], Close[t+lag]) pairs needed for regression

    # --- Visualization ---
    "heatmap_max_configs": 50,     # Max indicators/configs to show on heatmap/combined chart
    "plot_dpi": 300,               # DPI for saved plots

    # --- Parameter Optimization (Bayesian) ---
    "optimizer_n_calls": 50,       # Total evaluations per lag per indicator (Increased for guided steps)
    "optimizer_n_initial_points": 10, # Random points before fitting model per lag (Standard value)
    "optimizer_acq_func": 'gp_hedge',# Acquisition function ('LCB', 'EI', 'PI', 'gp_hedge')
}
# ==========================


# --- Ensure Core Directories Exist ---
DB_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
LINE_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
# Ensure Leaderboard directory exists (its parent, which is PROJECT_ROOT)
LEADERBOARD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)