# config.py
import os
from pathlib import Path

# --- Core Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DB_DIR = PROJECT_ROOT / 'database'
REPORTS_DIR = PROJECT_ROOT / 'reports'
HEATMAPS_DIR = REPORTS_DIR / 'heatmaps'
LINE_CHARTS_DIR = REPORTS_DIR / 'line_charts'
COMBINED_CHARTS_DIR = REPORTS_DIR / 'combined_charts'
LOG_DIR = PROJECT_ROOT / 'logs'
INDICATOR_PARAMS_PATH = PROJECT_ROOT / 'indicator_params.json'
LEADERBOARD_DB_PATH = PROJECT_ROOT / 'correlation_leaderboard.db'

# --- Database ---
DB_NAME_TEMPLATE = "{symbol}_{timeframe}.db"
SQLITE_MAX_VARIABLE_NUMBER = 900

# === Modifiable Defaults ===
# User can change these values directly
DEFAULTS = {
    # --- Data Source ---
    "symbol": "BTCUSDT",
    "timeframe": "1d",

    # --- Analysis Parameters ---
    "max_lag": 30,                   # Default max correlation lag if user doesn't specify
    "min_data_points_for_lag": 220,  # Min points needed beyond max_lag (e.g., for indicators)
    "target_max_correlations": 75000,# Target limit for estimated correlations (triggers warning)
    "default_analysis_path": 'tweak', # 'tweak' (Bayesian) or 'classical' (Default+Range)
    "total_analysis_steps_estimate": 10, # Rough estimate for overall progress bar

    # --- Classical Path (Previously Default) ---
    "classical_path_range_steps": 5,# +/- steps around default for Classical Path param generation

    # --- Bayesian Optimization Path (Tweak) ---
    "optimizer_n_calls": 50,        # Total evaluations per lag per indicator (Incl. initial)
    "optimizer_n_initial_points": 10,# Random points before fitting model per lag
    "optimizer_acq_func": 'gp_hedge',# Acquisition function ('LCB', 'EI', 'PI', 'gp_hedge')
    "weak_corr_threshold_skip_opt": 0.25, # Abs corr threshold below which initial points skip opt for a lag
    "max_total_opt_failures_indicator": 50, # Abort optimizing an indicator if objective fails this many times across all lags

    # --- Regression/Prediction ---
    "min_regression_points": 30,    # Min historical pairs needed for regression

    # --- Backtester / Historical Check ---
    "backtester_default_points": 50,# Default number of points for historical check per lag

    # --- Visualization ---
    "heatmap_max_configs": 50,      # Max indicators/configs on heatmap/combined chart
    "linechart_max_configs": 50,    # Max indicators/configs on combined line chart
    "plot_dpi": 150,                # DPI for saved plots

    # --- Reporting & Progress ---
    "eta_update_interval_seconds": 15,# How often to update ETA displays (seconds)
    "interim_report_frequency": 10, # Generate interim reports every N indicators during Bayesian opt
}
# ==========================


# --- Ensure Core Directories Exist ---
DB_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
LINE_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
COMBINED_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)
LEADERBOARD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)