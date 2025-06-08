# config.py
import os
from pathlib import Path
from types import MappingProxyType

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

class Config:
    """Configuration class that encapsulates all settings."""
    def __init__(self):
        # Core Paths
        self.PROJECT_ROOT = PROJECT_ROOT
        self.DB_DIR = DB_DIR
        self.REPORTS_DIR = REPORTS_DIR
        self.HEATMAPS_DIR = HEATMAPS_DIR
        self.LINE_CHARTS_DIR = LINE_CHARTS_DIR
        self.COMBINED_CHARTS_DIR = COMBINED_CHARTS_DIR
        self.LOG_DIR = LOG_DIR
        self.INDICATOR_PARAMS_PATH = INDICATOR_PARAMS_PATH
        self.LEADERBOARD_DB_PATH = LEADERBOARD_DB_PATH
        
        # Database
        self.DB_NAME_TEMPLATE = DB_NAME_TEMPLATE
        self.SQLITE_MAX_VARIABLE_NUMBER = SQLITE_MAX_VARIABLE_NUMBER
        
        # Defaults - Make a deep copy to prevent modification
        self.DEFAULTS = DEFAULTS.copy()
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        self.DB_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        self.HEATMAPS_DIR.mkdir(parents=True, exist_ok=True)
        self.LINE_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        self.COMBINED_CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.LEADERBOARD_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    def get_config_dict(self):
        """Return configuration as an immutable dictionary."""
        # Create a regular dict first
        config_dict = {
            'project_root': self.PROJECT_ROOT,
            'db_dir': self.DB_DIR,
            'reports_dir': self.REPORTS_DIR,
            'heatmaps_dir': self.HEATMAPS_DIR,
            'line_charts_dir': self.LINE_CHARTS_DIR,
            'combined_charts_dir': self.COMBINED_CHARTS_DIR,
            'log_dir': self.LOG_DIR,
            'indicator_params_path': self.INDICATOR_PARAMS_PATH,
            'leaderboard_db_path': self.LEADERBOARD_DB_PATH,
            'defaults': MappingProxyType(self.DEFAULTS)  # Make defaults immutable
        }
        # Return an immutable view of the entire dict
        return MappingProxyType(config_dict)

# Create a default instance for backward compatibility
config = Config()