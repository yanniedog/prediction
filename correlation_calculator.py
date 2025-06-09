# correlation_calculator.py

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable, Union, Literal
import sqlite3 # For specific error handling if needed
import time
import concurrent.futures
import os # To get CPU count
from datetime import timedelta # For ETA calculation
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import networkx as nx
from matplotlib.figure import Figure

import config
import sqlite_manager
import utils # For ETA formatting
import leaderboard_manager # For periodic reporting function type hint

logger = logging.getLogger(__name__)

# --- Constants ---
ETA_UPDATE_INTERVAL_SECONDS_CORR = config.DEFAULTS.get("eta_update_interval_seconds", 15)
_periodic_report_interval_seconds_corr: int = config.DEFAULTS.get("periodic_report_interval_seconds", 30) # Use config

# --- Core Correlation Calculation Function ---
# Calculates Corr(Indicator[t], Close[t+lag]) by shifting Close backward.
def calculate_correlation_indicator_vs_future_price(
    data: pd.DataFrame, indicator_col: str, lag: int
) -> Optional[float]:
    """
    Calculate Pearson correlation: Indicator[t] vs Close[t+lag].

    Args:
        data (pd.DataFrame): DataFrame with 'close' and indicator column, sorted chronologically.
                             Should have NaNs handled prior to calling if possible.
        indicator_col (str): Name of the indicator column.
        lag (int): Future periods for the price (must be > 0).

    Returns:
        Optional[float]: Pearson correlation, np.nan if impossible, None on error.
    """
    if not all(col in data.columns for col in [indicator_col, 'close']):
        return None
    if lag <= 0:
        return None

    indicator_series = data[indicator_col]
    close_series = data['close']

    # Early exit checks
    if indicator_series.isnull().all() or close_series.isnull().all() or indicator_series.nunique(dropna=True) <= 1:
        return np.nan

    try:
        # Shift Close Price BACKWARD by lag
        shifted_close_future = close_series.shift(-lag)
        # Combine, dropna, and check length before correlating
        combined = pd.concat([indicator_series, shifted_close_future], axis=1).dropna()
        if len(combined) < 2: # Need at least 2 pairs
            return np.nan
        # Calculate correlation using pandas on the cleaned data
        correlation = combined.iloc[:, 0].corr(combined.iloc[:, 1])
        # Return float or nan
        return float(correlation) if pd.notna(correlation) else np.nan
    except Exception as e:
        # logger.error(f"Error calculating correlation for {indicator_col}, lag {lag}: {e}", exc_info=False) # Avoid excessive logging in worker
        return None # Return None on unexpected error


# --- Worker Function for Parallel Processing ---
def _calculate_correlations_for_single_indicator(
    indicator_col_name: str,
    indicator_series: pd.Series, # Pass the series directly
    shifted_closes_future: Dict[int, pd.Series], # Pass the pre-shifted closes
    max_lag: int,
    symbol_id: int,
    timeframe_id: int,
    config_id: int
) -> List[Tuple[int, int, int, int, Optional[float]]]:
    """
    Worker to calculate correlations for all lags for one indicator series.
    """
    results_for_indicator = []
    nan_count = 0
    error_count = 0

    # Basic check before loop
    if indicator_series.isnull().all() or indicator_series.nunique(dropna=True) <= 1:
        is_all_nan = indicator_series.isnull().all()
        logger.debug(f"Worker: Skipping {indicator_col_name} (ConfigID: {config_id}) - {'All NaN' if is_all_nan else 'Constant Value'}.")
        # Return Nones (for DB) for all lags
        return [(symbol_id, timeframe_id, config_id, lag, None) for lag in range(1, max_lag + 1)]

    for lag in range(1, max_lag + 1):
        correlation_value: Optional[float] = None
        db_value: Optional[float] = None
        try:
            shifted_close = shifted_closes_future.get(lag)
            if shifted_close is None:
                correlation_value = None # Or np.nan? Using None for DB consistency
                error_count += 1
            else:
                # Use pandas .corr() - handles alignment and NaN pair removal
                # Combine first to ensure enough pairs after dropping NaNs
                combined = pd.concat([indicator_series, shifted_close], axis=1).dropna()
                if len(combined) < 2: # Check if enough data remains
                     correlation = np.nan
                else:
                     # Correlate the cleaned columns
                     correlation = combined.iloc[:, 0].corr(combined.iloc[:, 1])

                if pd.notna(correlation):
                    correlation_value = float(correlation)
                else:
                    correlation_value = np.nan # Use NaN for calculation failures (insufficient pairs/variance)
                    nan_count += 1
        except Exception as e:
            # logger.error(f"Worker Error ({indicator_col_name}, lag {lag}): {e}") # Can be noisy
            correlation_value = None # Use None for unexpected errors
            error_count += 1

        # Map np.nan -> None for DB storage consistency
        db_value = None if pd.isna(correlation_value) else correlation_value
        results_for_indicator.append((symbol_id, timeframe_id, config_id, lag, db_value))

    if nan_count > 0 or error_count > 0:
         logger.debug(f"Worker ({indicator_col_name}, ConfigID: {config_id}): Completed with {nan_count} NaN results, {error_count} errors.")

    return results_for_indicator


# --- Process Correlations (Accepts global timing info) ---
def process_correlations(
    data: pd.DataFrame,
    db_path: str,
    symbol_id: int,
    timeframe_id: int,
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    # --- New Args for global timing and periodic reports ---
    analysis_start_time_global: float,
    total_analysis_steps_global: int,
    current_step_base: float,
    total_steps_in_phase: float,
    display_progress_func: Callable,
    periodic_report_func: Callable # Function to call for periodic reports (e.g., tally)
) -> bool:
    """
    Calculates correlations using parallel processing, updating GLOBAL progress.
    """
    start_time_internal = time.time() # Internal start for logging duration of this phase only
    num_configs = len(indicator_configs_processed)
    logger.info(f"Starting PARALLELIZED correlation calculation for {num_configs} configurations up to lag {max_lag}...")

    # --- Input Data Validation ---
    if 'close' not in data.columns or data['close'].isnull().all():
        logger.error("'close' column missing or all NaN. Cannot calculate correlations.")
        return False
    if max_lag <= 0:
        logger.error(f"Max lag must be positive, got {max_lag}.")
        return False
    min_required_len = max_lag + config.DEFAULTS.get("min_data_points_for_lag", 1) # Use default, min 1 point after lag
    if len(data) < min_required_len:
         logger.error(f"Input data has insufficient rows ({len(data)}) for max_lag={max_lag} after NaN drop. Need {min_required_len}.");
         return False
    indicator_columns_present = [col for col in data.columns if utils.parse_indicator_column_name(col) is not None]
    if not indicator_columns_present:
        logger.error("No valid indicator columns found in input data for correlation.")
        return False
    num_indicator_cols = len(indicator_columns_present)

    # --- Database Connection ---
    conn = sqlite_manager.create_connection(db_path)
    if not conn: return False

    # --- Local timer for periodic reports within this phase ---
    last_periodic_report_time_corr = time.time()

    try:
        # --- Pre-shift close prices ---
        logger.info(f"Pre-calculating {max_lag} shifted 'close' price series...")
        start_shift_time = time.time()
        close_series = data['close'].astype(float)
        shifted_closes_future = {lag: close_series.shift(-lag).reindex(data.index) for lag in range(1, max_lag + 1)}
        logger.info(f"Pre-calculation of shifted closes complete. Time: {time.time() - start_shift_time:.2f}s.")

        # --- Prepare tasks for parallel execution ---
        tasks = []
        config_details_map = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
        valid_tasks_count = 0
        skipped_task_configs = 0

        for indicator_col_name in indicator_columns_present:
            parsed_info = utils.parse_indicator_column_name(indicator_col_name)
            if parsed_info is None:
                logger.warning(f"Could not parse '{indicator_col_name}'. Skipping."); skipped_task_configs += 1; continue
            base_name, config_id, output_suffix = parsed_info
            if not isinstance(config_id, int):
                logger.warning(f"Parsed non-integer config_id {config_id} from '{indicator_col_name}'. Skipping."); skipped_task_configs += 1; continue
            if config_id not in config_details_map:
                logger.debug(f"ConfigID {config_id} ('{indicator_col_name}') not in processed list map. Skipping task."); skipped_task_configs += 1; continue

            indicator_series = data[indicator_col_name].astype(float) # Ensure float

            tasks.append((
                indicator_col_name, indicator_series, shifted_closes_future,
                max_lag, symbol_id, timeframe_id, config_id
            ))
            valid_tasks_count += 1

        logger.info(f"Prepared {valid_tasks_count} correlation tasks for {num_indicator_cols} indicator columns. Skipped {skipped_task_configs} columns.")
        if not tasks: logger.error("No valid tasks generated."); return False

        # --- Execute in Parallel with GLOBAL Progress Reporting ---
        all_correlation_results: List[Tuple[int, int, int, int, Optional[float]]] = []
        num_cores = os.cpu_count() or 4 # Default to 4 if cannot detect
        max_workers = max(1, num_cores - 1) if num_cores > 1 else 1 # Use N-1 cores, minimum 1
        logger.info(f"Starting parallel execution with up to {max_workers} workers...")
        # start_parallel_time = time.time() # Removed local timer for ETA calculation
        last_progress_update_time = time.time() # Use for throttling print updates
        processed_tasks = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(_calculate_correlations_for_single_indicator, *task_args): task_args for task_args in tasks}

            for future in concurrent.futures.as_completed(future_to_task):
                task_args = future_to_task[future]
                indicator_name_done = task_args[0]
                try:
                    result_list = future.result()
                    if result_list: all_correlation_results.extend(result_list)
                    # else: logger.warning(f"Worker for '{indicator_name_done}' returned no results.") # Can be noisy
                except Exception as exc:
                    logger.error(f"Worker for '{indicator_name_done}' generated exception: {exc}", exc_info=True)

                processed_tasks += 1
                current_time = time.time()

                # --- Update GLOBAL Progress & ETA ---
                # Update progress display periodically or on completion
                if current_time - last_progress_update_time > ETA_UPDATE_INTERVAL_SECONDS_CORR or processed_tasks == valid_tasks_count:
                    # --- Calculate OVERALL Progress Step ---
                    frac_corr_done = processed_tasks / valid_tasks_count if valid_tasks_count > 0 else 1.0
                    current_overall_step = current_step_base + total_steps_in_phase * frac_corr_done
                    # --- Call the main display function ---
                    stage_desc = f"Corr Calc ({processed_tasks}/{valid_tasks_count})"
                    display_progress_func(stage_desc, current_overall_step, total_analysis_steps_global)
                    # --------------------------------------
                    last_progress_update_time = current_time
                    # Log less frequently to file to avoid spamming
                    if processed_tasks % 100 == 0 or processed_tasks == valid_tasks_count:
                        logger.info(f"Parallel Correlation Progress: {processed_tasks}/{valid_tasks_count} tasks ({frac_corr_done*100:.1f}%) complete.")

                # --- Periodic Report Generation ---
                if current_time - last_periodic_report_time_corr > _periodic_report_interval_seconds_corr:
                    try:
                         logger.info("Generating periodic tally report during correlation...")
                         periodic_report_func() # Call the passed function (e.g., leaderboard_manager.generate_leading_indicator_report)
                         last_periodic_report_time_corr = current_time
                    except Exception as report_err:
                         logger.error(f"Error generating periodic report during correlation: {report_err}", exc_info=True)

        # Display function handles the final newline after loop completion
        logger.info(f"Parallel execution finished.")
        logger.info(f"Total correlation records collected: {len(all_correlation_results)}")

        # --- Batch Insert Results ---
        if not all_correlation_results:
            logger.warning("No correlation results generated by parallel workers.")
            # Ensure database connection is closed even if no data to insert
            if conn: conn.close()
            return True # Success, but no data inserted

        logger.info(f"Starting batch insert of {len(all_correlation_results)} records...")
        start_insert_time = time.time()
        # Pass the existing connection to the batch insert function
        batch_success = sqlite_manager.batch_insert_correlations(conn, all_correlation_results)
        insert_duration = time.time() - start_insert_time
        logger.info(f"Batch insertion complete. Success: {batch_success}. Time: {insert_duration:.2f}s.")

        if not batch_success:
            logger.error("Batch insertion of correlations failed.")
            # Connection is closed in finally block
            return False

        total_phase_duration = time.time() - start_time_internal
        logger.info(f"Correlation processing phase finished. Duration: {total_phase_duration:.2f}s.")
        return True

    except Exception as e:
        logger.error(f"An error occurred during correlation processing: {e}", exc_info=True)
        return False
    finally:
        if conn:
            try:
                conn.close()
                logger.debug("Correlation processing DB connection closed.")
            except Exception as close_err:
                logger.error(f"Error closing correlation DB connection: {close_err}")

class CorrelationCalculator:
    """Calculate and analyze correlations between financial indicators and price data."""
    
    def __init__(self):
        """Initialize correlation calculator with improved error handling and dependency checks."""
        self.min_correlation = -1.0
        self.max_correlation = 1.0
        self.min_data_points = 100
        self.max_lag = 100
        # Dependency flags
        try:
            import sklearn
            self.SKLEARN_AVAILABLE = True
        except ImportError:
            self.SKLEARN_AVAILABLE = False
        try:
            import scipy
            self.SCIPY_AVAILABLE = True
        except ImportError:
            self.SCIPY_AVAILABLE = False
        try:
            import networkx
            self.NETWORKX_AVAILABLE = True
        except ImportError:
            self.NETWORKX_AVAILABLE = False
        
    def calculate_correlation(self, data: pd.DataFrame, indicator: str, lag: int) -> Optional[float]:
        """Calculate correlation between price and indicator with improved error handling."""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
                
            if indicator not in data.columns:
                raise ValueError(f"Indicator {indicator} not found in data")
                
            if not isinstance(lag, int) or lag < 0:
                raise ValueError(f"Invalid lag value: {lag}")
                
            if len(data) < self.min_data_points:
                raise ValueError(f"Insufficient data points (minimum {self.min_data_points} required)")
                
            # Get price and indicator series
            price = data['close']
            indicator_series = data[indicator]
            
            # Check for missing values
            if price.isnull().any() or indicator_series.isnull().any():
                raise ValueError("Data contains missing values")
                
            # Calculate correlation
            if lag == 0:
                correlation = price.corr(indicator_series)
            else:
                # Shift price series by lag
                shifted_price = price.shift(-lag)
                # Drop rows with NaN values after shift
                valid_mask = ~(shifted_price.isnull() | indicator_series.isnull())
                if valid_mask.sum() < self.min_data_points:
                    raise ValueError(f"Insufficient valid data points after lag {lag}")
                correlation = shifted_price[valid_mask].corr(indicator_series[valid_mask])
                
            # Validate correlation value
            if not self.min_correlation <= correlation <= self.max_correlation:
                raise ValueError(f"Invalid correlation value: {correlation}")
                
            return correlation
        except Exception as e:
            logging.error(f"Failed to calculate correlation for {indicator} at lag {lag}: {e}")
            return None
            
    def calculate_correlations(self, data: pd.DataFrame, indicator: str, max_lag: int) -> Optional[Dict[int, float]]:
        """Calculate correlations for all lags with improved error handling."""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
                
            if indicator not in data.columns:
                raise ValueError(f"Indicator {indicator} not found in data")
                
            if not isinstance(max_lag, int) or max_lag < 0:
                raise ValueError(f"Invalid max_lag value: {max_lag}")
                
            if max_lag > self.max_lag:
                raise ValueError(f"max_lag exceeds maximum allowed value of {self.max_lag}")
                
            # Calculate correlations for each lag
            correlations = {}
            for lag in range(max_lag + 1):
                correlation = self.calculate_correlation(data, indicator, lag)
                if correlation is not None:
                    correlations[lag] = correlation
                    
            if not correlations:
                raise ValueError(f"No valid correlations calculated for {indicator}")
                
            return correlations
        except Exception as e:
            logging.error(f"Failed to calculate correlations for {indicator}: {e}")
            return None
            
    def process_correlations(self, data: pd.DataFrame, configs: List[Dict[str, Any]], max_lag: int) -> bool:
        """Process correlations for multiple indicators with improved error handling."""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a pandas DataFrame")
                
            if not isinstance(configs, list):
                raise ValueError("Configs must be a list")
                
            if not configs:
                raise ValueError("No configurations provided")
                
            if not isinstance(max_lag, int) or max_lag < 0:
                raise ValueError(f"Invalid max_lag value: {max_lag}")
                
            # Process each configuration
            for config in configs:
                if not isinstance(config, dict):
                    raise ValueError("Invalid configuration format")
                    
                indicator = config.get('indicator_name')
                if not indicator:
                    raise ValueError("Missing indicator_name in configuration")
                    
                if indicator not in data.columns:
                    raise ValueError(f"Indicator {indicator} not found in data")
                    
                # Calculate correlations
                correlations = self.calculate_correlations(data, indicator, max_lag)
                if correlations is None:
                    raise ValueError(f"Failed to calculate correlations for {indicator}")
                    
                # Store correlations
                config['correlations'] = correlations
                
            return True
        except Exception as e:
            logging.error(f"Failed to process correlations: {e}")
            return False

    def calculate_rolling_correlation(self, data1: pd.Series, data2: pd.Series, 
                                    window: int = 20, method: Literal['pearson', 'kendall', 'spearman'] = 'pearson') -> pd.Series:
        """Calculate rolling correlation between two series."""
        if len(data1) != len(data2):
            raise ValueError("Series must have the same length")
        
        # Manually compute rolling correlation to support method argument
        result = []
        for i in range(len(data1)):
            start = max(0, i - window + 1)
            x = data1.iloc[start:i+1]
            y = data2.iloc[start:i+1]
            if x.isnull().all() or y.isnull().all() or len(x.dropna()) < 2 or len(y.dropna()) < 2:
                result.append(np.nan)
            else:
                try:
                    corr = x.corr(y, method=method)
                except Exception:
                    corr = np.nan
                result.append(corr)
        return pd.Series(result, index=data1.index)

    def calculate_correlation_matrix(self, data: pd.DataFrame, method: Literal['pearson', 'kendall', 'spearman'] = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for a DataFrame."""
        if data.empty:
            return pd.DataFrame()
        return data.corr(method=method)

    def test_correlation_significance(self, data1: pd.Series, data2: pd.Series, 
                                    alpha: float = 0.05) -> Dict[str, Any]:
        """Test correlation significance."""
        if not self.SCIPY_AVAILABLE:
            raise ImportError("scipy required for significance testing")
        
        # Handle missing values
        valid_data = pd.concat([data1, data2], axis=1).dropna()
        if len(valid_data) < 2:
            return {
                'correlation': np.nan,
                'p_value': np.nan,
                'significant': False
            }
        
        correlation, p_value = pearsonr(valid_data.iloc[:, 0], valid_data.iloc[:, 1])
        
        return {
            'correlation': float(correlation),
            'p_value': float(p_value),
            'significant': bool(p_value < alpha)
        }

    def plot_correlation_heatmap(self, data: pd.DataFrame) -> Figure:
        """Plot correlation heatmap."""
        if data.empty:
            raise ValueError("Empty DataFrame provided")
            
        corr_matrix = self.calculate_correlation_matrix(data)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Correlation Heatmap')
        return fig

    def plot_correlation_scatter(self, data1: pd.Series, data2: pd.Series) -> Figure:
        """Plot correlation scatter plot."""
        if len(data1) != len(data2):
            raise ValueError("Series must have the same length")
            
        valid_data = pd.concat([data1, data2], axis=1).dropna()
        if len(valid_data) < 2:
            raise ValueError("Insufficient valid data points for plotting")
            
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(valid_data.iloc[:, 0], valid_data.iloc[:, 1], alpha=0.5)
        ax.set_xlabel(str(data1.name) if data1.name is not None else "")
        ax.set_ylabel(str(data2.name) if data2.name is not None else "")
        ax.set_title('Correlation Scatter Plot')
        
        # Add correlation line
        z = np.polyfit(valid_data.iloc[:, 0], valid_data.iloc[:, 1], 1)
        p = np.poly1d(z)
        ax.plot(valid_data.iloc[:, 0], p(valid_data.iloc[:, 0]), "r--", alpha=0.8)
        
        # Add correlation coefficient
        corr = valid_data.iloc[:, 0].corr(valid_data.iloc[:, 1])
        ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', 
                transform=ax.transAxes, verticalalignment='top')
        return fig

    def cluster_correlations(self, data: pd.DataFrame, n_clusters: int = 2) -> Dict[str, Any]:
        """Cluster correlations using hierarchical clustering."""
        if not self.SKLEARN_AVAILABLE:
            raise ImportError("sklearn required for clustering")
            
        if data.empty:
            return {
                'labels': [],
                'centers': [],
                'silhouette_score': np.nan
            }
            
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.metrics import silhouette_score
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(data)
        
        # Convert to distance matrix
        distance_matrix = 1 - np.abs(corr_matrix)
        
        # Perform clustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters, metric='precomputed', linkage='complete')
        labels = clustering.fit_predict(distance_matrix)
        
        # Calculate silhouette score if possible
        try:
            score = silhouette_score(distance_matrix, labels, metric='precomputed')
        except:
            score = np.nan
            
        # Calculate cluster centers
        centers = []
        for i in range(n_clusters):
            cluster_data = corr_matrix[labels == i]
            if not cluster_data.empty:
                centers.append(cluster_data.mean().mean())
            else:
                centers.append(np.nan)
                
        return {
            'labels': labels.tolist(),
            'centers': centers,
            'silhouette_score': float(score)
        }

    def analyze_correlation_network(self, data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """Analyze correlation network."""
        if not self.NETWORKX_AVAILABLE:
            raise ImportError("networkx required for network analysis")
            
        if data.empty:
            return {
                'nodes': [],
                'edges': [],
                'centrality': {
                    'degree': {},
                    'betweenness': {},
                    'eigenvector': {}
                },
                'network': {
                    'nodes': [],
                    'edges': []
                }
            }
            
        import networkx as nx
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(data)
        
        # Create network
        G = nx.Graph()
        
        # Add nodes
        for col in data.columns:
            G.add_node(col)
            
        # Add edges
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns[i+1:], i+1):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    G.add_edge(col1, col2, weight=float(abs(corr_matrix.iloc[i, j])))
                    
        # Calculate centrality measures
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        except:
            degree_centrality = {node: 0.0 for node in G.nodes()}
            betweenness_centrality = {node: 0.0 for node in G.nodes()}
            eigenvector_centrality = {node: 0.0 for node in G.nodes()}
            
        # Return edges as (source, target, weight) tuples
        edge_list = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        node_list = list(G.nodes())
        
        return {
            'nodes': node_list,
            'edges': edge_list,
            'centrality': {
                'degree': degree_centrality,
                'betweenness': betweenness_centrality,
                'eigenvector': eigenvector_centrality
            },
            'network': {
                'nodes': node_list,
                'edges': edge_list
            }
        }

    def generate_correlation_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate a comprehensive correlation report."""
        if data.empty:
            return {
                'summary': {
                    'mean_correlation': np.nan,
                    'max_correlation': np.nan,
                    'min_correlation': np.nan,
                    'std_correlation': np.nan,
                    'total_correlations': 0,
                    'significant_correlations': 0
                },
                'correlation_matrix': pd.DataFrame(),
                'correlations': pd.DataFrame(),
                'significance': {},
                'visualizations': {},
                'analysis': {
                    'decomposition': {},
                    'stability': {}
                }
            }
            
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(data)
        
        # Calculate summary statistics
        corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
        valid_corrs = corr_values[~np.isnan(corr_values)]
        
        summary = {
            'mean_correlation': float(np.mean(valid_corrs)) if len(valid_corrs) > 0 else np.nan,
            'max_correlation': float(np.max(valid_corrs)) if len(valid_corrs) > 0 else np.nan,
            'min_correlation': float(np.min(valid_corrs)) if len(valid_corrs) > 0 else np.nan,
            'std_correlation': float(np.std(valid_corrs)) if len(valid_corrs) > 0 else np.nan,
            'total_correlations': int(len(valid_corrs)),
            'significant_correlations': int(np.sum(np.abs(valid_corrs) > 0.5)) if len(valid_corrs) > 0 else 0
        }
        
        # Generate visualizations
        visualizations = {}
        try:
            fig = self.plot_correlation_heatmap(data)
            visualizations['heatmap'] = fig
        except Exception as e:
            logger.warning(f"Could not generate correlation heatmap: {e}")
        
        # Generate analysis
        analysis = {
            'decomposition': {},
            'stability': {}
        }
        
        if self.SKLEARN_AVAILABLE:
            try:
                # Perform PCA decomposition
                n_components = min(3, len(data.columns))
                pca = PCA(n_components=n_components)
                pca.fit(corr_matrix.fillna(0))
                
                analysis['decomposition'] = {
                    'explained_variance': pca.explained_variance_ratio_.tolist(),
                    'components': pca.components_.tolist(),
                    'loadings': pd.DataFrame(pca.components_.T, 
                                          index=corr_matrix.columns, 
                                          columns=[f'PC{i+1}' for i in range(n_components)]).to_dict()
                }
                
                # Analyze stability
                stability = self.analyze_correlation_stability(data)
                analysis['stability'] = stability
                
            except Exception as e:
                logger.warning(f"Could not perform advanced analysis: {e}")
        
        return {
            'summary': summary,
            'correlation_matrix': corr_matrix,
            'correlations': pd.DataFrame(corr_values, columns=['correlation']),
            'significance': {},
            'visualizations': visualizations,
            'analysis': analysis
        }

    def visualize_correlation(self, data1: pd.Series, data2: pd.Series) -> Figure:
        """Visualize correlation between two series (scatter plot)."""
        return self.plot_correlation_scatter(data1, data2)

    def visualize_correlation_matrix(self, data: pd.DataFrame) -> Figure:
        """Visualize correlation matrix (heatmap)."""
        return self.plot_correlation_heatmap(data)

    def decompose_correlation(self, data: pd.DataFrame, n_components: int = 2) -> Dict[str, Any]:
        """Perform correlation decomposition using PCA.
        Args:
            data (pd.DataFrame): DataFrame to decompose
            n_components (int): Number of principal components to extract
        Returns:
            Dict[str, Any]: Decomposition results
        """
        if not self.SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for correlation decomposition")
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(data)
        # Perform PCA
        pca = PCA(n_components=n_components)
        pca.fit(corr_matrix)
        return {
            'components': pd.DataFrame(pca.components_, columns=corr_matrix.columns),
            'explained_variance': pca.explained_variance_,
            'loadings': pd.DataFrame(pca.components_.T, index=corr_matrix.columns, columns=[f'PC{i+1}' for i in range(n_components)])
        }

    def analyze_correlation_stability(self, data: pd.DataFrame, window_size: int = 20) -> Dict[str, Any]:
        """Analyze correlation stability over time."""
        if len(data) < window_size:
            return {
                'stability_score': 0.0,
                'volatility': 0.0,
                'trend': 'stable'
            }
            
        # Calculate rolling correlations
        rolling_corr = data.rolling(window=window_size, min_periods=1).corr()
        
        # Get the correlation between the first two columns
        corr_series = rolling_corr.iloc[0::2, 1]  # Get every other row, second column
        
        # Calculate stability metrics
        stability_score = float(1.0 - np.nanstd(corr_series))  # Higher std = lower stability
        volatility = float(np.nanstd(corr_series))
        
        # Determine trend
        if len(corr_series.dropna()) < 2:
            trend = 'stable'
        else:
            # Calculate linear regression slope
            x = np.arange(len(corr_series))
            mask = ~np.isnan(corr_series)
            if np.sum(mask) >= 2:
                slope = np.polyfit(x[mask], corr_series[mask], 1)[0]
                if abs(slope) < 0.01:
                    trend = 'stable'
                else:
                    trend = 'increasing' if slope > 0 else 'decreasing'
            else:
                trend = 'stable'
        
        return {
            'stability_score': max(0.0, min(1.0, stability_score)),  # Clamp between 0 and 1
            'volatility': max(0.0, volatility),  # Ensure non-negative
            'trend': trend
        }

    def forecast_correlations(self, data: pd.DataFrame, forecast_horizon: int = 5) -> Dict[str, Any]:
        """Forecast future correlations using simple moving average."""
        # Calculate rolling correlation matrices
        window_size = 20
        n_windows = len(data) - window_size + 1
        corr_matrices = []
        
        for i in range(n_windows):
            window_data = data.iloc[i:i+window_size]
            corr_matrix = self.calculate_correlation_matrix(window_data)
            corr_matrices.append(corr_matrix)
        
        # Convert to 3D array
        corr_array = np.array(corr_matrices)
        
        # Calculate mean and standard deviation
        mean_corr = np.mean(corr_array, axis=0)
        std_corr = np.std(corr_array, axis=0)
        
        # Generate forecast
        forecast_matrix = mean_corr
        confidence_intervals = {
            'lower': mean_corr - 1.96 * std_corr,
            'upper': mean_corr + 1.96 * std_corr
        }
        
        return {
            'forecast': pd.DataFrame(forecast_matrix, 
                                   index=data.columns, 
                                   columns=data.columns),
            'confidence_intervals': {
                'lower': pd.DataFrame(confidence_intervals['lower'], 
                                    index=data.columns, 
                                    columns=data.columns),
                'upper': pd.DataFrame(confidence_intervals['upper'], 
                                    index=data.columns, 
                                    columns=data.columns)
            }
        }

    def detect_correlation_regimes(self, data: pd.DataFrame, n_regimes: int = 2) -> Dict[str, Any]:
        """Detect correlation regimes using clustering."""
        if not self.SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for regime detection")
        
        # Calculate rolling correlation matrices
        window_size = min(20, len(data) - 1)  # Ensure window size is valid
        if window_size < 2:
            return {
                'regime_labels': [],
                'regimes': [],
                'regime_correlations': [],
                'transition_matrix': np.array([]),
                'regime_characteristics': []
            }
            
        n_windows = len(data) - window_size + 1
        corr_features = []
        
        for i in range(n_windows):
            window_data = data.iloc[i:i+window_size]
            corr_matrix = self.calculate_correlation_matrix(window_data)
            if not corr_matrix.empty:
                # Convert to numpy array before getting upper triangle
                corr_array = corr_matrix.values.astype(np.float64)
                upper_tri = corr_array[np.triu_indices_from(corr_array, k=1)]
                corr_features.append(upper_tri)
        
        if not corr_features:
            return {
                'regime_labels': [],
                'regimes': [],
                'regime_correlations': [],
                'transition_matrix': np.array([]),
                'regime_characteristics': []
            }
        
        # Convert to array
        X = np.array(corr_features, dtype=np.float64)
        
        # Adjust number of regimes if needed
        n_regimes = min(n_regimes, len(X) - 1)
        if n_regimes < 2:
            return {
                'regime_labels': [0] * len(X),
                'regimes': [0],
                'regime_correlations': [X.mean(axis=0)],
                'transition_matrix': np.array([[1.0]]),
                'regime_characteristics': [
                    {
                        'mean_correlation': float(X.mean()),
                        'std_correlation': float(X.std()),
                        'stability': float(1.0 - X.std()),
                        'duration': len(X)
                    }
                ]
            }
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Calculate regime correlations and characteristics
        regime_correlations = []
        regime_characteristics = []
        for regime in range(n_regimes):
            regime_indices = np.where(labels == regime)[0]
            if len(regime_indices) > 0:
                regime_data = X[regime_indices]
                regime_corr = np.mean(regime_data, axis=0)
                regime_correlations.append(regime_corr)
                regime_characteristics.append({
                    'mean_correlation': float(np.mean(regime_corr)),
                    'std_correlation': float(np.std(regime_corr)),
                    'stability': float(1.0 - np.std(regime_corr)),
                    'duration': len(regime_indices),
                    'volatility': float(np.std(regime_data)),
                    'trend': 'increasing' if np.mean(np.diff(regime_data.mean(axis=1))) > 0 else 'decreasing'
                })
        
        # Calculate transition matrix
        transition_matrix = np.zeros((n_regimes, n_regimes), dtype=np.float64)
        for i in range(len(labels)-1):
            transition_matrix[labels[i], labels[i+1]] += 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        transition_matrix = np.divide(transition_matrix, row_sums, 
                                    out=np.zeros_like(transition_matrix), 
                                    where=row_sums!=0)
        
        labels_list = list(labels)
        unique_regimes = sorted(set(labels_list))
        return {
            'regime_labels': labels_list,
            'regimes': unique_regimes,
            'regime_correlations': regime_correlations,
            'transition_matrix': transition_matrix,
            'regime_characteristics': regime_characteristics
        }

    def analyze_correlation_network(self, data: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """Analyze correlation network."""
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(data)
        # Create network
        G = nx.Graph()
        # Add nodes
        for col in data.columns:
            G.add_node(col)
        # Add edges
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns[i+1:], i+1):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    G.add_edge(col1, col2, weight=float(abs(corr_matrix.iloc[i, j])))
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
        # Return edges as (source, target, weight) tuples
        edge_list = [(u, v, d['weight']) for u, v, d in G.edges(data=True)]
        node_list = list(G.nodes())
        return {
            'nodes': node_list,
            'edges': edge_list,
            'centrality': {
                'degree': degree_centrality,
                'betweenness': betweenness_centrality,
                'eigenvector': eigenvector_centrality
            },
            'network': {
                'nodes': node_list,
                'edges': edge_list
            }
        }

    def detect_correlation_anomalies(self, data: pd.DataFrame, window_size: int = 20, 
                                   threshold: float = 2.0) -> Dict[str, Any]:
        """Detect correlation anomalies using rolling statistics."""
        # Calculate rolling correlation matrices
        n_windows = len(data) - window_size + 1
        corr_matrices = []
        dates = []
        
        for i in range(n_windows):
            window_data = data.iloc[i:i+window_size]
            corr_matrix = self.calculate_correlation_matrix(window_data)
            corr_matrices.append(corr_matrix)
            dates.append(window_data.index[-1])
        
        # Convert to array
        corr_array = np.array(corr_matrices)
        
        # Calculate mean and standard deviation
        mean_corr = np.mean(corr_array, axis=0)
        std_corr = np.std(corr_array, axis=0)
        
        # Detect anomalies
        anomaly_scores = []
        anomaly_dates = []
        anomaly_correlations = []
        
        for i, corr_matrix in enumerate(corr_matrices):
            z_scores = np.abs((corr_matrix - mean_corr) / std_corr)
            if np.any(z_scores > threshold):
                anomaly_scores.append(np.max(z_scores))
                anomaly_dates.append(dates[i])
                anomaly_correlations.append(corr_matrix)
        
        return {
            'anomaly_scores': anomaly_scores,
            'anomaly_dates': anomaly_dates,
            'anomaly_correlations': anomaly_correlations,
            'anomalies': anomaly_dates,
            'scores': anomaly_scores
        }

    def visualize_correlation(self, data1: pd.Series, data2: pd.Series) -> Figure:
        """Visualize correlation between two series (scatter plot)."""
        return self.plot_correlation_scatter(data1, data2)

    def visualize_correlation_matrix(self, data: pd.DataFrame) -> Figure:
        """Visualize correlation matrix (heatmap)."""
        return self.plot_correlation_heatmap(data)

    def _calculate_distance_matrix(self, correlations_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate distance matrix between correlation series."""
        try:
            # Ensure we have valid data
            if correlations_df.empty:
                logger.warning("Empty correlation dataframe for distance calculation")
                return pd.DataFrame()
            
            # Calculate correlation between correlation series
            distance_matrix = correlations_df.corr(method='pearson')
            
            # Check if we have any valid values before calculating mean
            if distance_matrix.empty or distance_matrix.isna().all().all():
                logger.warning("No valid correlations for distance matrix")
                return pd.DataFrame()
            
            # Convert to numpy array for calculations
            matrix_values = distance_matrix.to_numpy()
            valid_mask = ~np.isnan(matrix_values)
            
            # Calculate mean only if we have valid values
            if np.any(valid_mask):
                mean_val = float(np.nanmean(matrix_values))
            else:
                logger.warning("No valid values for mean calculation in distance matrix")
                mean_val = 0.0
            
            # Fill NaN with mean
            distance_matrix = distance_matrix.fillna(mean_val)
            
            return distance_matrix
        except Exception as e:
            logger.error(f"Error calculating distance matrix: {e}")
            return pd.DataFrame()

    def _calculate_correlation_matrix(self, correlations_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix between correlation series."""
        try:
            if correlations_df.empty:
                return pd.DataFrame()
            
            # Convert to numpy array for calculations
            matrix = correlations_df.to_numpy()
            if matrix.size == 0:
                return pd.DataFrame()
            
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(matrix, rowvar=False)
            if np.isnan(corr_matrix).all():
                return pd.DataFrame()
            
            # Convert back to DataFrame with proper index/columns
            return pd.DataFrame(corr_matrix, 
                              index=correlations_df.columns,
                              columns=correlations_df.columns)
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()

    def _get_triu_indices(self, matrix: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Get upper triangle indices from matrix."""
        try:
            # Convert to numpy array first
            arr = matrix.to_numpy()
            return np.triu_indices_from(arr)
        except Exception as e:
            logger.error(f"Error getting upper triangle indices: {e}")
            return np.array([]), np.array([])

    def _calculate_correlation_stability(self, correlations_df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
        """Calculate correlation stability using rolling windows."""
        try:
            if correlations_df.empty:
                return pd.DataFrame()
                
            # Convert to numeric type explicitly
            numeric_df = correlations_df.astype(float)
            
            # Calculate rolling correlation
            rolling_corr = numeric_df.rolling(window=window_size, min_periods=1).corr()
            
            # Calculate stability metrics
            stability = pd.DataFrame(index=rolling_corr.index)
            stability['mean'] = rolling_corr.mean(axis=1)
            stability['std'] = rolling_corr.std(axis=1)
            stability['stability'] = 1.0 - stability['std']  # Higher is more stable
            
            return stability
        except Exception as e:
            self.logger.error(f"Error calculating correlation stability: {e}")
            return pd.DataFrame()

    def _calculate_correlation_regimes(self, correlations_df: pd.DataFrame, n_regimes: int = 3) -> pd.DataFrame:
        """Detect correlation regimes using clustering."""
        try:
            if correlations_df.empty:
                return pd.DataFrame()
                
            # Convert to numeric type explicitly
            numeric_df = correlations_df.astype(float)
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Convert to numpy array for clustering
            matrix = corr_matrix.to_numpy()
            if matrix.size == 0 or np.isnan(matrix).all():
                return pd.DataFrame()
                
            # Perform clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_regimes, random_state=42)
            labels = kmeans.fit_predict(matrix)
            
            # Create regime DataFrame
            regimes = pd.DataFrame({
                'regime': labels.astype(int),  # Ensure integer type
                'stability': (1.0 - np.std(matrix, axis=1)).astype(float)  # Ensure float type
            }, index=corr_matrix.index)
            
            return regimes
        except Exception as e:
            self.logger.error(f"Error detecting correlation regimes: {e}")
            return pd.DataFrame()

    def _calculate_correlation_network(self, correlations_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """Calculate correlation network edges based on threshold."""
        try:
            if correlations_df.empty:
                return pd.DataFrame()
                
            # Convert to numeric type explicitly
            numeric_df = correlations_df.astype(float)
            
            # Calculate correlation matrix
            corr_matrix = numeric_df.corr()
            
            # Convert to numpy array for calculations
            matrix = corr_matrix.to_numpy()
            if matrix.size == 0:
                return pd.DataFrame()
                
            # Get upper triangle indices
            i, j = np.triu_indices_from(matrix, k=1)  # k=1 to exclude diagonal
            
            # Create edges DataFrame with explicit type conversion
            edges = pd.DataFrame({
                'source': corr_matrix.index[i].astype(str),
                'target': corr_matrix.columns[j].astype(str),
                'weight': matrix[i, j].astype(float)
            })
            
            # Filter by threshold
            edges = edges[edges['weight'].abs() >= float(threshold)]
            
            return edges
        except Exception as e:
            self.logger.error(f"Error calculating correlation network: {e}")
            return pd.DataFrame()

    def _calculate_correlation_forecast(self, correlations_df: pd.DataFrame, forecast_horizon: int = 5) -> pd.DataFrame:
        """Forecast future correlations using simple moving average."""
        try:
            if correlations_df.empty:
                return pd.DataFrame()
                
            # Convert to numeric type explicitly
            numeric_df = correlations_df.astype(float)
            
            # Calculate moving average
            ma = numeric_df.rolling(window=forecast_horizon, min_periods=1).mean()
            
            # Extend forecast with explicit type conversion
            last_values = ma.iloc[-1].astype(float)
            forecast = pd.DataFrame(
                [last_values] * forecast_horizon,
                index=pd.date_range(start=ma.index[-1], periods=forecast_horizon + 1, freq='D')[1:],
                columns=ma.columns
            )
            
            return pd.concat([ma, forecast])
        except Exception as e:
            self.logger.error(f"Error forecasting correlations: {e}")
            return pd.DataFrame()

    def calculate_lag_correlation(self, data1: pd.Series, data2: pd.Series, lags: Union[range, List[int]]) -> pd.Series:
        """
        Calculate correlation between two series at different lags.
        
        Args:
            data1: First data series
            data2: Second data series
            lags: Range or list of lags to calculate correlation for
            
        Returns:
            Series with correlation values for each lag
        """
        if not isinstance(data1, pd.Series) or not isinstance(data2, pd.Series):
            raise TypeError("data1 and data2 must be pandas Series")
            
        correlations = pd.Series(index=lags, dtype=float)
        for lag in lags:
            if lag >= 0:
                # Shift data2 forward for positive lags
                shifted_data2 = data2.shift(-lag)
                correlations[lag] = self.calculate_correlation(data1, shifted_data2)
            else:
                # Shift data1 forward for negative lags
                shifted_data1 = data1.shift(lag)
                correlations[lag] = self.calculate_correlation(shifted_data1, data2)
                
        return correlations

def _calculate_correlation(series1: pd.Series, series2: pd.Series, method: str = 'pearson') -> float:
    """Calculate correlation between two series."""
    if method == 'pearson':
        return series1.corr(series2, method='pearson')
    elif method == 'spearman':
        return series1.corr(series2, method='spearman')
    elif method == 'kendall':
        return series1.corr(series2, method='kendall')
    else:
        raise ValueError(f"Unknown correlation method: {method}")

def _calculate_lag_correlation(series1: pd.Series, series2: pd.Series, lag: int = 1, method: str = 'pearson') -> float:
    """Calculate lagged correlation between two series."""
    if lag == 0:
        return _calculate_correlation(series1, series2, method)
    return _calculate_correlation(series1, series2.shift(-lag), method)

def _calculate_rolling_correlation(series1: pd.Series, series2: pd.Series, window: int = 20, method: str = 'pearson') -> pd.Series:
    """Calculate rolling correlation between two series."""
    return series1.rolling(window).corr(series2, method=method)

def _calculate_cross_correlation(series1: pd.Series, series2: pd.Series, lag: int = 0) -> float:
    """Calculate cross-correlation at a given lag."""
    if lag > 0:
        return np.corrcoef(series1[lag:], series2[:-lag])[0, 1]
    elif lag < 0:
        return np.corrcoef(series1[:lag], series2[-lag:])[0, 1]
    else:
        return np.corrcoef(series1, series2)[0, 1]

def _calculate_autocorrelation(series: pd.Series, lag: int = 1) -> float:
    """Calculate autocorrelation for a given lag."""
    return series.autocorr(lag=lag)

def _calculate_partial_correlation(series1: pd.Series, series2: pd.Series, control: pd.Series) -> float:
    """Calculate partial correlation between two series, controlling for a third."""
    from scipy.stats import linregress
    # Regress out control from both series
    res1 = series1 - linregress(control, series1).slope * control
    res2 = series2 - linregress(control, series2).slope * control
    return _calculate_correlation(res1, res2)

def _calculate_spearman_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """Calculate Spearman correlation."""
    return _calculate_correlation(series1, series2, method='spearman')

def _calculate_kendall_correlation(series1: pd.Series, series2: pd.Series) -> float:
    """Calculate Kendall correlation."""
    return _calculate_correlation(series1, series2, method='kendall')