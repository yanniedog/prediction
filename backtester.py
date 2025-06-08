# backtester.py
# TODO: Implement a point-in-time backtesting approach for realistic performance simulation.
#       This would involve:
#       1. Splitting data into Training and Backtest sets.
#       2. Generating a leaderboard *only* using the Training set.
#       3. Running the backtest loop on the Backtest set, selecting predictors based *only* on the Training leaderboard.
#       4. Potentially using walk-forward optimization for even greater realism (though computationally expensive).
#       The current implementation checks historical regression stability of FINAL predictors, it does NOT simulate real trading.

import logging
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Callable, Union
from pathlib import Path
from datetime import datetime, timezone
import math
import json # Added for parsing params
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Import config for constants
import config
import utils
import data_manager
import indicator_factory
import leaderboard_manager
import sqlite_manager
import predictor # To reuse regression logic and pair generation

logger = logging.getLogger(__name__)

# Get constant from config
MIN_REGRESSION_POINTS = config.DEFAULTS.get("min_regression_points", 30) # Fallback default

class Backtester:
    def __init__(self, data_manager, indicator_factory):
        """Initialize the Backtester with data and indicator managers."""
        self.data_manager = data_manager
        self.indicator_factory = indicator_factory
        self._validate_dependencies()

    def _validate_dependencies(self):
        """Validate that all required dependencies are available."""
        if not predictor.STATSMODELS_AVAILABLE:
            logger.warning("statsmodels not available. Some functionality may be limited.")
        try:
            import scipy
            self.SCIPY_AVAILABLE = True
        except ImportError:
            logger.warning("scipy not available. Some functionality may be limited.")
            self.SCIPY_AVAILABLE = False

    def run_strategy(self, data: pd.DataFrame, strategy_func: Callable, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a trading strategy on the provided data."""
        try:
            # Generate positions using the strategy
            positions = strategy_func(data, params)
            if not isinstance(positions, pd.Series):
                raise ValueError("Strategy must return a pandas Series of positions")
            
            # Size positions
            sized_positions = self.size_positions(positions, method='fixed', size=0.1)
            
            # Calculate returns
            returns = self.calculate_returns(data, sized_positions)
            
            # Calculate equity curve
            equity_curve = (1 + returns).cumprod()
            
            return {
                'positions': positions,
                'sized_positions': sized_positions,
                'returns': returns,
                'equity_curve': equity_curve
            }
        except Exception as e:
            logger.error(f"Error running strategy: {e}")
            raise

    def size_positions(self, positions: pd.Series, method: str = 'fixed', **kwargs) -> pd.Series:
        """Size positions based on the specified method."""
        if method == 'fixed':
            size = kwargs.get('size', 0.1)
            return positions * size
        elif method == 'dynamic':
            data = kwargs.get('data')
            volatility_window = kwargs.get('volatility_window', 20)
            if data is None:
                raise ValueError("Data required for dynamic position sizing")
            # Calculate volatility
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=volatility_window).std()
            # Size inversely proportional to volatility
            target_vol = 0.15  # Target annualized volatility
            position_size = target_vol / (volatility * np.sqrt(252))
            position_size = position_size.clip(0, 1)  # Limit position size
            # Align position_size to positions index
            position_size_aligned = position_size.reindex(positions.index)
            return positions * position_size_aligned
        else:
            raise ValueError(f"Unknown position sizing method: {method}")

    def calculate_returns(self, data: pd.DataFrame, positions: pd.Series, transaction_cost: float = 0.0) -> pd.Series:
        """Calculate strategy returns including transaction costs."""
        # Handle empty data or missing 'close' column
        if data is None or data.empty or 'close' not in data.columns or positions is None or positions.empty:
            return pd.Series(dtype=float)
        # Calculate price returns
        price_returns = data['close'].pct_change()
        
        # Calculate strategy returns
        strategy_returns = positions.shift(1) * price_returns
        
        # Apply transaction costs
        if transaction_cost > 0:
            position_changes = positions.diff().abs()
            transaction_costs = position_changes * transaction_cost
            strategy_returns -= transaction_costs
        
        return strategy_returns

    def calculate_cumulative_returns(self, returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns."""
        return (1 + returns).cumprod()

    def calculate_performance_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate key performance metrics."""
        if not self.SCIPY_AVAILABLE:
            raise ImportError("scipy required for performance metrics")
        
        # Annualized return
        annual_return = returns.mean() * 252
        
        # Annualized volatility
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.02  # Assuming 2% risk-free rate
        sharpe_ratio = (annual_return - risk_free_rate) / annual_vol if annual_vol != 0 else 0
        
        # Maximum drawdown
        cum_returns = self.calculate_cumulative_returns(returns)
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics."""
        if not self.SCIPY_AVAILABLE:
            raise ImportError("scipy required for risk metrics")
        
        # Volatility
        volatility = returns.std() * np.sqrt(252)
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns, 5)
        
        # Conditional Value at Risk (95%)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    def optimize_strategy(self, data: pd.DataFrame, strategy_func: Callable, 
                         param_space: Dict[str, List[Any]], metric: str = 'sharpe_ratio') -> Dict[str, Any]:
        """Optimize strategy parameters using grid search."""
        best_score = float('-inf')
        best_params = None
        results = []
        
        # Generate parameter combinations
        param_names = list(param_space.keys())
        param_values = list(param_space.values())
        param_combinations = [dict(zip(param_names, values)) 
                            for values in np.array(np.meshgrid(*param_values)).T.reshape(-1, len(param_names))]
        
        for params in param_combinations:
            try:
                # Run strategy with current parameters
                strategy_results = self.run_strategy(data, strategy_func, params)
                returns = strategy_results['returns']
                
                # Calculate performance metrics
                metrics = self.calculate_performance_metrics(returns)
                score = metrics[metric]
                
                results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics
                })
                
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception as e:
                logger.warning(f"Strategy failed with params {params}: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }

    def walk_forward_analysis(self, data: pd.DataFrame, strategy_func: Callable, 
                            params: Dict[str, Any], train_size: float = 0.7, 
                            step_size: float = 0.1) -> Dict[str, Any]:
        """Perform walk-forward analysis."""
        # Validate train_size and step_size
        if not (0 < train_size < 1):
            raise ValueError("train_size must be between 0 and 1 (exclusive)")
        if not (0 < step_size < 1):
            raise ValueError("step_size must be between 0 and 1 (exclusive)")
        n_points = len(data)
        train_size_points = int(n_points * train_size)
        step_size_points = int(n_points * step_size)
        
        train_metrics = []
        test_metrics = []
        
        for i in range(0, n_points - train_size_points, step_size_points):
            # Split data
            train_end = i + train_size_points
            test_end = min(train_end + step_size_points, n_points)
            
            train_data = data.iloc[i:train_end]
            test_data = data.iloc[train_end:test_end]
            
            if len(test_data) == 0:
                break
            
            # Run strategy on training data
            train_results = self.run_strategy(train_data, strategy_func, params)
            train_metrics.append(self.calculate_performance_metrics(train_results['returns']))
            
            # Run strategy on test data
            test_results = self.run_strategy(test_data, strategy_func, params)
            test_metrics.append(self.calculate_performance_metrics(test_results['returns']))
        
        # Calculate combined metrics by averaging across all windows
        combined_metrics = {}
        if train_metrics and test_metrics:
            # Average metrics across all windows
            for metric in train_metrics[0].keys():
                train_values = [m[metric] for m in train_metrics]
                test_values = [m[metric] for m in test_metrics]
                combined_metrics[metric] = {
                    'train': np.mean(train_values),
                    'test': np.mean(test_values),
                    'overall': np.mean(train_values + test_values)
                }
        
        return {
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            'combined_metrics': combined_metrics
        }

    def run_monte_carlo_simulation(self, returns: pd.Series, n_simulations: int = 1000, 
                                 time_steps: int = 252) -> Dict[str, Any]:
        """Run Monte Carlo simulation of strategy returns."""
        if not self.SCIPY_AVAILABLE:
            raise ImportError("scipy required for Monte Carlo simulation")
        # Validate parameters
        if n_simulations <= 0 or time_steps <= 0:
            raise ValueError("n_simulations and time_steps must be positive integers")
        # Calculate mean and standard deviation of returns
        mean_return = returns.mean()
        std_return = returns.std()
        
        # Generate simulations
        simulations = np.random.normal(
            mean_return, 
            std_return, 
            (n_simulations, time_steps)
        )
        
        # Calculate cumulative returns for each simulation
        cum_returns = (1 + simulations).cumprod(axis=1)
        
        # Calculate confidence intervals
        confidence_intervals = {
            '95%': np.percentile(cum_returns, [2.5, 97.5], axis=0),
            '68%': np.percentile(cum_returns, [16, 84], axis=0)
        }
        
        return {
            'simulations': cum_returns,
            'confidence_intervals': confidence_intervals
        }

    def plot_equity_curve(self, returns: pd.Series) -> plt.Figure:
        """Plot equity curve."""
        fig, ax = plt.subplots(figsize=(12, 6))
        equity_curve = self.calculate_cumulative_returns(returns)
        equity_curve.plot(ax=ax)
        ax.set_title('Equity Curve')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.grid(True)
        return fig

    def plot_drawdown(self, returns: pd.Series) -> plt.Figure:
        """Plot drawdown chart."""
        fig, ax = plt.subplots(figsize=(12, 6))
        cum_returns = self.calculate_cumulative_returns(returns)
        rolling_max = cum_returns.expanding().max()
        drawdowns = cum_returns / rolling_max - 1
        drawdowns.plot(ax=ax)
        ax.set_title('Drawdown')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown')
        ax.grid(True)
        return fig

    def plot_monthly_returns_heatmap(self, returns: pd.Series) -> plt.Figure:
        """Plot monthly returns heatmap."""
        # Ensure index is DatetimeIndex
        if not isinstance(returns.index, pd.DatetimeIndex):
            # Try to convert from ms since epoch
            returns = returns.copy()
            returns.index = pd.to_datetime(returns.index, unit='ms')
        # Resample returns to monthly frequency
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        # Create a DataFrame with year and month as indices
        monthly_returns_df = pd.DataFrame({
            'year': monthly_returns.index.year,
            'month': monthly_returns.index.month,
            'returns': monthly_returns.values
        })
        # Pivot the data for the heatmap
        heatmap_data = monthly_returns_df.pivot(
            index='year', 
            columns='month', 
            values='returns'
        )
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(heatmap_data, annot=True, fmt='.2%', cmap='RdYlGn', 
                   center=0, ax=ax)
        ax.set_title('Monthly Returns Heatmap')
        return fig

    def generate_performance_report(self, returns: pd.Series) -> Dict[str, Any]:
        """Generate a comprehensive performance report."""
        # Calculate metrics
        performance_metrics = self.calculate_performance_metrics(returns)
        risk_metrics = self.calculate_risk_metrics(returns)
        
        # Generate plots
        equity_plot = self.plot_equity_curve(returns)
        drawdown_plot = self.plot_drawdown(returns)
        heatmap = self.plot_monthly_returns_heatmap(returns)
        
        # Create summary
        summary = {
            'total_return': self.calculate_cumulative_returns(returns).iloc[-1] - 1,
            'annualized_return': performance_metrics['annual_return'],
            'sharpe_ratio': performance_metrics['sharpe_ratio'],
            'max_drawdown': performance_metrics['max_drawdown'],
            'volatility': risk_metrics['volatility']
        }
        
        return {
            'summary': summary,
            'metrics': {**performance_metrics, **risk_metrics},
            'charts': {
                'equity_curve': equity_plot,
                'drawdown': drawdown_plot,
                'monthly_returns': heatmap
            }
        }

    def compare_with_benchmark(self, strategy_returns: pd.Series, 
                             benchmark_returns: pd.Series) -> Dict[str, Any]:
        """Compare strategy performance with a benchmark."""
        # Validate input series lengths
        if len(strategy_returns) != len(benchmark_returns):
            raise ValueError("Strategy returns and benchmark returns must have the same length")
            
        # Calculate metrics for both strategy and benchmark
        strategy_metrics = self.calculate_performance_metrics(strategy_returns)
        benchmark_metrics = self.calculate_performance_metrics(benchmark_returns)
        
        # Calculate relative performance
        relative_returns = strategy_returns - benchmark_returns
        relative_metrics = self.calculate_performance_metrics(relative_returns)
        
        # Calculate correlation
        correlation = strategy_returns.corr(benchmark_returns)
        
        # Calculate information ratio
        tracking_error = relative_returns.std() * np.sqrt(252)
        information_ratio = relative_metrics['annual_return'] / tracking_error if tracking_error != 0 else 0
        
        return {
            'relative_performance': relative_metrics,
            'metrics_comparison': {
                'strategy': strategy_metrics,
                'benchmark': benchmark_metrics
            },
            'correlation': correlation,
            'information_ratio': information_ratio
        }

def run_backtest(
    db_path: Path,
    symbol: str,
    timeframe: str,
    max_lag_backtest: int,
    num_backtest_points: int = 50 # Number of historical points to test per lag
) -> None:
    """
    Performs a simplified "historical predictor check", NOT a true backtest.

    *** CRITICAL WARNING ***
    This function uses the FINAL leaderboard generated from the ENTIRE dataset
    to select predictors. This introduces SEVERE LOOKAHEAD BIAS.
    The results DO NOT represent realistic trading performance or how the
    strategy would have performed historically based only on past data.
    It primarily serves to check the historical stability of the regression
    relationship for the predictors that were ultimately found to be best overall.
    *** CRITICAL WARNING ***

    Args:
        db_path: Path to the database containing historical price data.
        symbol: Trading symbol (e.g., 'BTCUSDT').
        timeframe: Timeframe string (e.g., '1d').
        max_lag_backtest: Maximum lag to check predictors for.
        num_backtest_points: Number of historical points to test per lag.
    """
    utils.clear_screen()
    # --- Enhanced Warnings ---
    print("\n" + "="*60)
    print("--- HISTORICAL PREDICTOR CHECK (NOT A REALISTIC BACKTEST) ---")
    print("="*60)
    print(f"\nSymbol: {symbol} ({timeframe})")
    print(f"Checking last {num_backtest_points} points for lags 1 to {max_lag_backtest}")
    print("\n" + "***" * 20)
    print("*** WARNING: SEVERE LOOKAHEAD BIAS PRESENT ***")
    print("*** Uses FINAL leaderboard (knows future best predictors). ***")
    print("*** Results DO NOT reflect realistic trading performance. ***")
    print("*** Checks historical regression stability of final predictors ONLY. ***")
    print("***" * 20 + "\n")
    logger.critical("Starting Historical Predictor Check - SEVERE LOOKAHEAD BIAS PRESENT.")
    logger.warning(f"Historical Check Parameters: {symbol}/{timeframe}, Lags: 1-{max_lag_backtest}, Points: {num_backtest_points}. Uses FINAL leaderboard.")
    # --- End Enhanced Warnings ---

    if not predictor.STATSMODELS_AVAILABLE:
        print("\nError: This check requires 'statsmodels'.")
        logger.error("Historical Check skipped: statsmodels missing.")
        return

    # 1. Load full historical data
    full_historical_data = data_manager.load_data(db_path)
    # Check data validity thoroughly
    if full_historical_data is None or full_historical_data.empty:
        print(f"Error: Could not load historical data from {db_path}.")
        logger.error(f"Historical Check failed: Cannot load data from {db_path}.")
        return
    required_cols = ['date', 'close', 'open_time'] # Check essential columns
    if not all(col in full_historical_data.columns for col in required_cols):
        missing = [c for c in required_cols if c not in full_historical_data.columns]
        print(f"Error: Historical data missing required columns: {missing}.")
        logger.error(f"Historical Check failed: Data missing columns {missing}.")
        return
    if not pd.api.types.is_numeric_dtype(full_historical_data['close']):
        print("Error: 'close' column is not numeric.")
        logger.error("Historical Check failed: 'close' column not numeric.")
        return
    if len(full_historical_data) < max_lag_backtest + num_backtest_points + MIN_REGRESSION_POINTS:
        print(f"Error: Insufficient historical data ({len(full_historical_data)} rows). Need at least {max_lag_backtest + num_backtest_points + MIN_REGRESSION_POINTS}.")
        logger.error(f"Insufficient data for historical check. Have {len(full_historical_data)}, Need {max_lag_backtest + num_backtest_points + MIN_REGRESSION_POINTS}.")
        return

    # 2. Load final leaderboard data (emphasize this is the source of bias)
    logger.warning("Loading FINAL leaderboard state for historical check (Source of LOOKAHEAD BIAS).")
    final_leaderboard = leaderboard_manager.load_leaderboard()
    if not final_leaderboard:
        print("Error: Could not load FINAL leaderboard data. Cannot proceed.")
        logger.error("Historical Check failed: cannot load final leaderboard.")
        return

    # 3. Get Symbol/Timeframe IDs (Needed for pair generation if not cached)
    conn_ids = sqlite_manager.create_connection(str(db_path)); sym_id = -1; tf_id = -1
    if conn_ids:
        try:
            conn_ids.execute("BEGIN;")
            sym_id = sqlite_manager._get_or_create_id(conn_ids, 'symbols', 'symbol', symbol)
            tf_id = sqlite_manager._get_or_create_id(conn_ids, 'timeframes', 'timeframe', timeframe)
            conn_ids.commit()
        except Exception as id_err:
             logger.error(f"Historical Check: Failed get sym/tf IDs: {id_err}", exc_info=True)
             try: conn_ids.rollback()
             except Exception as rb_err: logger.error(f"Rollback failed: {rb_err}")
        finally:
             if conn_ids: conn_ids.close()
    else: logger.error("Historical Check: Failed connect for sym/tf IDs.")

    if sym_id == -1 or tf_id == -1:
        print("\nError: Failed to get Symbol/Timeframe ID from database.")
        logger.error("Historical Check failed: Could not get Symbol/Timeframe IDs.")
        return

    # --- Historical Check Loop ---
    backtest_results = []
    # Cache indicators calculated during this check (local scope)
    indicator_series_cache_local: Dict[int, pd.DataFrame] = {}

    total_iterations = max_lag_backtest * num_backtest_points
    completed_iterations = 0

    print("\nRunning historical check iterations...")
    for lag in range(1, max_lag_backtest + 1):
        # Find best predictor for this lag ONCE from the FINAL leaderboard
        predictor_key_pos = (lag, 'positive')
        predictor_key_neg = (lag, 'negative')
        best_predictor_info = None
        corr_pos = final_leaderboard.get(predictor_key_pos, {}).get('correlation_value', -np.inf)
        corr_neg = final_leaderboard.get(predictor_key_neg, {}).get('correlation_value', np.inf)

        # Choose based on absolute correlation value from the final leaderboard
        # Ensure values are valid floats before comparing
        valid_pos = isinstance(corr_pos, (int, float)) and pd.notna(corr_pos)
        valid_neg = isinstance(corr_neg, (int, float)) and pd.notna(corr_neg)
        abs_pos = abs(corr_pos) if valid_pos else -np.inf
        abs_neg = abs(corr_neg) if valid_neg else -np.inf

        if abs_pos >= abs_neg and valid_pos:
            best_predictor_info = final_leaderboard.get(predictor_key_pos)
        elif valid_neg:
            best_predictor_info = final_leaderboard.get(predictor_key_neg)

        if best_predictor_info:
            best_predictor_info['lag'] = lag # Add lag info for context

        # Validate predictor info structure
        if (not best_predictor_info or
            not best_predictor_info.get('config_id_source_db') or
            not best_predictor_info.get('indicator_name') or
            not best_predictor_info.get('config_json')):
            logger.warning(f"Historical Check: No valid predictor found for Lag = {lag} in FINAL leaderboard. Skipping lag.")
            completed_iterations += num_backtest_points # Increment progress even if skipped
            continue

        ind_name = best_predictor_info['indicator_name']
        cfg_id = best_predictor_info['config_id_source_db']
        try:
            # Attempt to parse params here. If fails, skip lag.
            params = json.loads(best_predictor_info['config_json'])
            indicator_config = {'indicator_name': ind_name, 'params': params, 'config_id': cfg_id}
        except json.JSONDecodeError:
            logger.error(f"Historical Check: Failed to parse params for predictor CfgID {cfg_id} (Lag {lag}). Skipping lag.")
            completed_iterations += num_backtest_points; continue
        except Exception as e:
             logger.error(f"Historical Check: Error preparing predictor config for CfgID {cfg_id} (Lag {lag}): {e}. Skipping lag.")
             completed_iterations += num_backtest_points; continue

        logger.info(f"Historical Check Lag {lag}: Using Predictor CfgID {cfg_id} ('{ind_name}') from FINAL leaderboard.")

        for i in range(num_backtest_points):
            # t = index for predictor calculation
            # target_idx = index for actual price verification
            # We iterate from the most recent point backwards
            t = len(full_historical_data) - 1 - lag - i
            target_idx = t + lag

            # Basic bounds check
            if t < 0 or target_idx >= len(full_historical_data):
                logger.warning(f"Historical Check: Index out of bounds (t={t}, target={target_idx}). Stopping early for lag {lag}.")
                break # Stop testing this lag if we run out of data

            current_progress = (completed_iterations / total_iterations) * 100 if total_iterations > 0 else 0
            print(f" Progress: {current_progress:.1f}% (Lag {lag}, Point {i+1}/{num_backtest_points})", end='\r')

            try:
                # Data slices
                # Data for regression uses history *up to* point t
                data_for_regression = full_historical_data.iloc[:t+1]
                actual_price = full_historical_data.iloc[target_idx]['close']
                actual_date = full_historical_data.iloc[target_idx]['date']
                predictor_date = full_historical_data.iloc[t]['date']

                # Calculate current indicator value at time t
                # Use the local cache, but calculate if missing
                current_ind_val = None
                indicator_df_cached = indicator_series_cache_local.get(cfg_id)

                if indicator_df_cached is None: # Not cached or previously failed
                    logger.debug(f"Historical Check: Calculating full indicator series for CfgID {cfg_id} (up to point {t})")
                    # Pass copy to avoid modification issues, use full history for stable calculation
                    indicator_df_full_hist = indicator_factory._compute_single_indicator(
                        full_historical_data.copy(), # Use full history for calc
                        indicator_config
                    )
                    if indicator_df_full_hist is not None and not indicator_df_full_hist.empty:
                        indicator_series_cache_local[cfg_id] = indicator_df_full_hist
                        indicator_df_cached = indicator_df_full_hist # Use the newly calculated df
                    else:
                        logger.error(f"Historical Check: Failed compute indicator Cfg {cfg_id} for point {t}. Skipping point.")
                        indicator_series_cache_local[cfg_id] = pd.DataFrame() # Cache empty to avoid re-trying
                        completed_iterations += 1; continue
                # else: logger.debug(f"Historical Check: Using cached indicator series for CfgID {cfg_id}")

                # Get the indicator column name (handles multi-output)
                potential_cols = [col for col in indicator_df_cached.columns if col.startswith(f"{ind_name}_{cfg_id}")]
                if not potential_cols:
                    logger.error(f"Historical Check: No output col found for CfgID {cfg_id}. Skipping point {t}, lag {lag}.")
                    completed_iterations += 1; continue
                current_ind_col = potential_cols[0]

                # Get value at index t from the cached series
                if t < len(indicator_df_cached):
                    current_ind_val = indicator_df_cached[current_ind_col].iloc[t]
                    if pd.isna(current_ind_val):
                         logger.warning(f"Historical Check: Indicator value NaN at index {t} for CfgID {cfg_id}, Lag {lag}. Skipping point.")
                         completed_iterations += 1; continue
                else: # Should not happen if caching logic is correct, but safety check
                    logger.error(f"Historical Check: Index {t} out of bounds for indicator CfgID {cfg_id}. Skipping point.")
                    completed_iterations += 1; continue

                # Get historical pairs using data *up to time t*
                # Pass an empty dict to force recalc on slice (don't use main series cache here)
                hist_pairs = predictor._get_historical_indicator_price_pairs(
                    db_path, sym_id, tf_id, indicator_config, lag,
                    data_for_regression, # Use data only up to t
                    {} # Use a temporary empty cache for pair generation
                )
                if hist_pairs is None or len(hist_pairs) < MIN_REGRESSION_POINTS:
                    logger.warning(f"Historical Check: Insufficient regression pairs ({len(hist_pairs) if hist_pairs is not None else 0}) at point {t}, lag {lag}. Min={MIN_REGRESSION_POINTS}. Skipping.")
                    completed_iterations += 1; continue

                # Perform regression using the historical pairs up to t
                reg_res = predictor._perform_prediction_regression(hist_pairs, current_ind_val, lag)
                if reg_res is None:
                    logger.warning(f"Historical Check: Regression failed at point {t}, lag {lag}. Skipping.")
                    completed_iterations += 1; continue

                # Store result
                predicted_price = reg_res['predicted_value']
                error = predicted_price - actual_price
                pct_error = (error / actual_price) * 100 if actual_price != 0 else np.inf

                backtest_results.append({
                    'Lag': lag,
                    'Test Point Index (i)': i,
                    'Predictor Time (t)': predictor_date, # Use datetime objects
                    'Target Time (t+lag)': actual_date,   # Use datetime objects
                    'Actual Price': actual_price,
                    'Predicted Price': predicted_price,
                    'Error': error,
                    'Percent Error': pct_error,
                    'Predictor CfgID': cfg_id,
                    'Predictor Name': ind_name,
                    'Indicator Value @ t': current_ind_val,
                    'Regression R2': reg_res['r_squared']
                })

            except Exception as iter_err:
                logger.error(f"Historical Check: Error during iteration (Lag {lag}, Point {i}): {iter_err}", exc_info=True)
            finally:
                completed_iterations += 1 # Ensure progress increments even on error within loop

    print("\nHistorical check iterations complete.") # Final newline after progress indicator

    # 4. Analyze and Report Results
    if not backtest_results:
        print("\nHistorical check finished with no results.")
        logger.warning("Historical check completed but no results were generated.")
        return

    results_df = pd.DataFrame(backtest_results)

    # Calculate overall metrics
    mae = np.mean(np.abs(results_df['Error']))
    rmse = np.sqrt(np.mean(results_df['Error']**2))
    # Calculate MAPE carefully, excluding zero actual prices and infs
    valid_pct_err = results_df.loc[(results_df['Actual Price'] != 0) & np.isfinite(results_df['Percent Error']), 'Percent Error']
    mape = np.mean(np.abs(valid_pct_err)) if not valid_pct_err.empty else np.nan

    print("\n--- Historical Predictor Check Overall Summary ---")
    print("REMINDER: Results affected by LOOKAHEAD BIAS in predictor selection.")
    print(f"Total Predictions Tested: {len(results_df)}")
    print(f"MAE (Mean Absolute Error):  {mae:.4f}")
    print(f"RMSE (Root Mean Squared Error): {rmse:.4f}")
    print(f"MAPE (Mean Absolute Percentage Error): {mape:.2f}%" if pd.notna(mape) else "MAPE: N/A")

    # Calculate metrics per lag
    metrics_per_lag = results_df.groupby('Lag').agg(
        Predictions=('Error', 'size'),
        MAE=('Error', lambda x: np.mean(np.abs(x))),
        RMSE=('Error', lambda x: np.sqrt(np.mean(x**2))),
        # Calculate MAPE per group, handling potential infs/NaNs within the lambda
        MAPE=('Percent Error', lambda x: np.mean(np.abs(x.loc[np.isfinite(x) & (x != 0)])) if not x.loc[np.isfinite(x) & (x != 0)].empty else np.nan),
        Mean_R2=('Regression R2', 'mean')
    ).reset_index()

    print("\n--- Historical Predictor Check Metrics Per Lag ---")
    metrics_per_lag['MAPE'] = metrics_per_lag['MAPE'].map(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
    metrics_per_lag['MAE'] = metrics_per_lag['MAE'].map('{:.4f}'.format)
    metrics_per_lag['RMSE'] = metrics_per_lag['RMSE'].map('{:.4f}'.format)
    metrics_per_lag['Mean_R2'] = metrics_per_lag['Mean_R2'].map('{:.3f}'.format)
    # Increase display width for better table formatting
    with pd.option_context('display.width', 1000, 'display.max_columns', None):
        print(metrics_per_lag.to_string(index=False))

    # Save detailed results to CSV
    output_filename = f"{symbol}_{timeframe}_historical_check_details_{max_lag_backtest}lags_{num_backtest_points}pts.csv"
    output_filepath = config.REPORTS_DIR / output_filename
    output_filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        results_df_sorted = results_df.sort_values(by=['Lag', 'Test Point Index (i)']).copy()
        # Format dates for CSV readability (UTC)
        results_df_sorted['Predictor Time (t)'] = results_df_sorted['Predictor Time (t)'].dt.strftime('%Y-%m-%d %H:%M:%S')
        results_df_sorted['Target Time (t+lag)'] = results_df_sorted['Target Time (t+lag)'].dt.strftime('%Y-%m-%d %H:%M:%S')

        results_df_sorted.to_csv(output_filepath, index=False, float_format='%.6f')
        print(f"\nDetailed historical check results saved to: {output_filepath}")
        logger.info(f"Historical Check details saved to: {output_filepath}")
    except Exception as e:
        print(f"\nError saving detailed historical check results: {e}")
        logger.error(f"Failed to save historical check CSV: {e}", exc_info=True)

# Example of how to potentially call it (e.g., from main.py prompt)
# if __name__ == '__main__':
#     print("Backtester (Historical Check) module should be run via main.py.")