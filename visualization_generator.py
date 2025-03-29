# visualization_generator.py
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend suitable for scripts
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json # Added for parameter formatting
import re   # Added for filename sanitizing

import config
import utils

logger = logging.getLogger(__name__)

# --- Helper functions _prepare_filenames, _set_axis_intersection_at_zero remain the same ---
def _prepare_filenames(output_dir: Path, file_prefix: str, config_identifier: str, chart_type: str) -> Path:
    """Creates a sanitized filename for the plot."""
    # Basic sanitization: replace common problematic characters
    safe_identifier = re.sub(r'[\\/*?:"<>|\s]+', '_', config_identifier)
    # Limit length
    max_len = 100 # Max length for the config identifier part of the filename
    if len(safe_identifier) > max_len:
        safe_identifier = safe_identifier[:max_len] + "_etc"

    filename = f"{file_prefix}_{safe_identifier}_{chart_type}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename

def _set_axis_intersection_at_zero(ax):
    """Moves the x-axis spine to y=0."""
    try:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # Optional: Adjust label positions slightly if they overlap axes
        # ax.xaxis.set_label_coords(1.05, 0.5)
        # ax.yaxis.set_label_coords(0.05, 1.05)
    except Exception as e:
        logger.warning(f"Could not set axis intersection at zero: {e}")


# --- plot_correlation_lines remains the same ---
def plot_correlation_lines(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str
) -> None:
    """
    Generates separate line charts for each indicator configuration's correlation vs. lag,
    including approximated 95% CI bands based on rolling standard deviation.
    Axes intersect at y=0.
    """
    logger.info(f"Generating {len(correlations_by_config_id)} separate correlation line charts with CI bands...")
    if max_lag <= 0:
         logger.error("Max lag is zero or negative, cannot plot correlation lines.")
         return

    lags = list(range(1, max_lag + 1))
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed}
    rolling_window_ci = 10 # Window for rolling std dev for CI bands
    ci_multiplier = 1.96 # For ~95% CI

    plotted_count = 0
    for config_id, corr_values_full in correlations_by_config_id.items():
        config_info = configs_dict.get(config_id)
        if not config_info:
            logger.warning(f"Configuration info not found for Config ID {config_id}. Skipping plot.")
            continue

        indicator_name = config_info['indicator_name']
        # Create a short, safe representation of parameters for the filename/title
        try:
             params_str_short = "-".join(f"{k}{v}" for k,v in sorted(config_info['params'].items()))
             params_str_title = json.dumps(config_info['params'], separators=(',', ':')) # Full params for title
        except:
             params_str_short = str(config_info['params']) # Fallback
             params_str_title = params_str_short
        # Ensure filename component is safe
        safe_params_str = re.sub(r'[\\/*?:"<>|\s]+', '_', params_str_short)

        config_identifier_base = f"{indicator_name}_{config_id}_{safe_params_str}" # More descriptive identifier

        if corr_values_full is None or not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag:
            logger.warning(f"Insufficient or invalid correlation data length for {config_identifier_base} (Expected {max_lag}, Got {len(corr_values_full) if corr_values_full else 0}). Skipping plot.")
            continue

        corr_values = corr_values_full[:max_lag] # Trim to max_lag

        # Convert to Series for easier manipulation
        corr_series = pd.Series(corr_values, index=lags, dtype=float)
        # Drop NaN values *for plotting the main line* to avoid gaps
        corr_series_cleaned = corr_series.dropna()

        if len(corr_series_cleaned) < 2: # Need at least two points to draw a line
             logger.info(f"Not enough valid correlation data points (< 2) to plot for {config_identifier_base}. Skipping.")
             continue

        plot_lags = corr_series_cleaned.index
        plot_corrs = corr_series_cleaned.values

        logger.debug(f"INDIVIDUAL PLOT: ID={config_id}, Identifier={config_identifier_base}, Lags={list(plot_lags[:5])}, Corrs={list(plot_corrs[:5])}")

        # Calculate rolling standard deviation on the *original* series (with NaNs)
        rolling_std = corr_series.rolling(window=rolling_window_ci, min_periods=2, center=True).std()
        half_width = ci_multiplier * rolling_std

        # Calculate bands based on the *original* series index and values
        upper_band = (corr_series + half_width).clip(upper=1.0)
        lower_band = (corr_series - half_width).clip(lower=-1.0)

        # Plotting
        fig, ax = plt.subplots(figsize=(12, 6), dpi=config.PLOT_DPI)
        # Plot the main correlation line (only where non-NaN)
        ax.plot(plot_lags, plot_corrs, marker='.', linestyle='-', label=f'Correlation (Config {config_id})', zorder=3)

        # Plot the confidence interval band where it's available
        valid_band_lags = rolling_std.dropna().index
        if not valid_band_lags.empty:
            ax.fill_between(valid_band_lags,
                            lower_band.loc[valid_band_lags],
                            upper_band.loc[valid_band_lags],
                            color='skyblue', alpha=0.3, interpolate=True,
                            label=f'Approx. 95% CI (Roll.Win={rolling_window_ci})',
                            zorder=2)
        else:
            logger.debug(f"No valid rolling std dev calculated for CI band for {config_identifier_base}.")


        # Formatting
        ax.set_title(f"Correlation vs. Lag: {indicator_name}\nParams: {params_str_title} (ID: {config_id})", fontsize=10)
        ax.set_xlabel("Lag (Periods)")
        ax.set_ylabel("Pearson Correlation")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, max_lag + 1) # Extend xlim slightly
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)

        _set_axis_intersection_at_zero(ax) # Move axes to zero lines
        ax.legend(fontsize='small')
        fig.tight_layout()

        filepath = _prepare_filenames(output_dir, file_prefix, config_identifier_base, "line_CI")
        try:
            fig.savefig(filepath)
            logger.debug(f"Saved line chart with CI: {filepath.name}")
            plotted_count += 1
        except Exception as e:
            logger.error(f"Failed to save line chart {filepath.name}: {e}", exc_info=True)
        finally:
            plt.close(fig) # Close the figure to free memory

    logger.info(f"Generated {plotted_count} separate line charts with CI bands.")


# --- _get_stats remains the same ---
def _get_stats(corr_list: List[Optional[float]]) -> Dict[str, Optional[float]]:
    """Calculate mean, std, max of absolute correlations, handling NaNs."""
    valid_corrs = [c for c in corr_list if pd.notna(c)]
    if not valid_corrs:
        return {'mean_abs': None, 'std_abs': None, 'max_abs': None, 'peak_lag': None}

    abs_corrs = [abs(c) for c in valid_corrs]
    max_abs_val = np.max(abs_corrs) if abs_corrs else 0.0
    peak_lag_val = None
    if abs_corrs:
         try:
             # Find the first lag where the absolute correlation is close to the max absolute value
             peak_lag_index = np.argmax(np.abs(np.nan_to_num(corr_list, nan=-2.0))) # Use argmax on abs values, handle NaNs
             peak_lag_val = peak_lag_index + 1 # Lags are 1-based
             # Verify the value at peak_lag_val is indeed the max_abs_val (or close)
             if not np.isclose(abs(corr_list[peak_lag_index]), max_abs_val):
                  logger.warning(f"Peak lag identification mismatch: index {peak_lag_index} value {corr_list[peak_lag_index]}, max abs value {max_abs_val}")
                  # Fallback? Maybe find first occurrence instead of argmax? For now, keep argmax result.
         except (ValueError, IndexError) as e:
             logger.warning(f"Could not determine peak lag accurately: {e}")
             peak_lag_val = None


    return {
        'mean_abs': np.mean(abs_corrs) if abs_corrs else None,
        'std_abs': np.std(abs_corrs) if abs_corrs else None,
        'max_abs': max_abs_val if abs_corrs else None,
        'peak_lag': peak_lag_val
    }


# --- generate_combined_correlation_chart remains the same ---
def generate_combined_correlation_chart(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str
) -> None:
    """Generates a combined line chart showing correlations for ALL valid indicator configs. Axes intersect at y=0."""
    logger.info(f"Generating combined correlation chart for ALL configurations...")
    if max_lag <= 0 or not correlations_by_config_id:
         logger.warning("No data or invalid lag for combined chart.")
         return

    lags = list(range(1, max_lag + 1))
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed}

    plot_data = []
    for config_id, corr_values_full in correlations_by_config_id.items():
         if corr_values_full is None or not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag:
             logger.warning(f"Skipping config ID {config_id} in combined chart prep due to invalid/insufficient data length ({len(corr_values_full) if corr_values_full else 0} < {max_lag}).")
             continue

         corr_values = corr_values_full[:max_lag]
         stats = _get_stats(corr_values) # Calculates stats based on absolute values

         # Check if there's any significant correlation signal
         if stats['max_abs'] is not None and stats['max_abs'] > 1e-6: # Use max_abs for filtering weak signals
             config_info = configs_dict.get(config_id)
             if config_info:
                 name = config_info['indicator_name']
                 # Short param string for label
                 params_str_short = "-".join(f"{k}{v}" for k,v in sorted(config_info['params'].items()))
                 safe_params_str = re.sub(r'[\\/*?:"<>|\s]+', '_', params_str_short)
                 identifier = f"{name}_{config_id}_{safe_params_str}" # Include short params in identifier

                 plot_data.append({
                     'config_id': config_id,
                     'identifier': identifier,
                     'correlations': corr_values,
                     'mean_abs': stats['mean_abs'], # Store abs mean for potential sorting
                     'max_abs': stats['max_abs'],   # Store max abs for potential sorting
                     'peak_lag': stats['peak_lag'] if stats['peak_lag'] is not None else max_lag + 1
                 })
         else:
             logger.debug(f"Skipping config ID {config_id} from combined plot as max absolute correlation is near zero or NaN.")

    if not plot_data:
         logger.warning("No valid indicator data to plot for combined chart after filtering weak signals.")
         return

    # Optional sorting (e.g., by max absolute correlation to highlight strongest signals first)
    plot_data.sort(key=lambda x: x.get('max_abs', 0), reverse=True)
    logger.info("Sorted configurations by max absolute correlation (desc) for combined plot.")

    fig, ax = plt.subplots(figsize=(15, 10), dpi=config.PLOT_DPI)

    plotted_count = 0
    # Limit number of lines plotted if it exceeds a threshold (e.g., 50) for clarity
    max_lines_on_combined = config.HEATMAP_MAX_CONFIGS # Reuse config setting
    plot_subset = plot_data[:max_lines_on_combined]
    if len(plot_data) > max_lines_on_combined:
        logger.warning(f"Plotting only the top {max_lines_on_combined} configurations (sorted by max abs corr) on the combined chart due to high count ({len(plot_data)}).")


    for item in plot_subset: # Plot subset
        identifier = item['identifier']
        config_id_plotting = item['config_id']
        corr_values_list = item['correlations']

        # Convert to Series for easier handling of potential NaNs within the list
        corr_series = pd.Series(corr_values_list, index=lags, dtype=float)
        corr_series_cleaned = corr_series.dropna()

        if len(corr_series_cleaned) < 2:
            logger.warning(f"Not enough valid points for {identifier} in combined chart. Skipping line.")
            continue

        plot_lags = corr_series_cleaned.index
        plot_corrs = corr_series_cleaned.values

        logger.debug(f"COMBINED PLOT (Top {max_lines_on_combined}): ID={config_id_plotting}, Identifier={identifier}, Lags={list(plot_lags[:5])}, Corrs={list(plot_corrs[:5])}")

        # Use thinner lines and slightly transparent for better visibility
        ax.plot(plot_lags, plot_corrs, marker='.', markersize=1, linestyle='-', linewidth=0.7, alpha=0.6, label=identifier)
        plotted_count += 1

    if plotted_count == 0:
        logger.warning("No lines were plotted for the combined chart.")
        plt.close(fig)
        return

    chart_title = f"Combined Correlation vs. Lag"
    if len(plot_data) > max_lines_on_combined:
        chart_title += f" (Top {plotted_count} of {len(plot_data)} Configs by Max Abs Corr)"
    else:
        chart_title += f" ({plotted_count} Configurations)"

    ax.set_title(chart_title, fontsize=12)
    ax.set_xlabel("Lag (Periods)")
    ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, max_lag + 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)

    _set_axis_intersection_at_zero(ax)

    # Adjust legend display based on number of lines
    if plotted_count <= 30:
         ax.legend(loc='best', fontsize='xx-small', ncol=2 if plotted_count > 15 else 1)
    else:
         logger.warning(f"Hiding legend for combined chart as there are too many lines ({plotted_count}).")
         # ax.legend().set_visible(False) # Uncomment to explicitly hide

    fig.tight_layout()

    # Adjust filename based on whether it's a subset
    filename_suffix = f"Combined_{plotted_count}Configs" if len(plot_data) > max_lines_on_combined else f"Combined_All_{plotted_count}"
    filepath = _prepare_filenames(output_dir, file_prefix, filename_suffix, "chart")
    try:
        fig.savefig(filepath)
        logger.info(f"Saved combined chart: {filepath.name}")
    except Exception as e:
        logger.error(f"Failed to save combined chart {filepath.name}: {e}", exc_info=True)
    finally:
        plt.close(fig)


# --- generate_enhanced_heatmap remains the same ---
def generate_enhanced_heatmap(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str,
    is_tweak_path: bool # Flag to know if it came from optimizer
) -> None:
    """
    Generates an enhanced heatmap of correlation values.
    Filters to HEATMAP_MAX_CONFIGS if needed (especially for default path).
    Sorts columns by absolute mean correlation.
    """
    logger.info("Generating enhanced correlation heatmap...")
    if max_lag <= 0 or not correlations_by_config_id:
         logger.warning("No data or invalid lag for heatmap.")
         return

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed}
    heatmap_data = {}
    valid_config_ids = list(correlations_by_config_id.keys())

    # Build initial DataFrame with all available data
    for config_id in valid_config_ids:
        corr_values_full = correlations_by_config_id.get(config_id)
        # Ensure we have a list and it's long enough
        if (corr_values_full is None or
            not isinstance(corr_values_full, list) or
            len(corr_values_full) < max_lag):
             logger.warning(f"Skipping config ID {config_id} in heatmap prep due to insufficient data length.")
             continue
        # Trim to max_lag and check if *any* data exists within this range
        corr_values = corr_values_full[:max_lag]
        if all(pd.isna(c) for c in corr_values):
             logger.warning(f"Skipping config ID {config_id} in heatmap prep as all correlations up to lag {max_lag} are NaN.")
             continue

        config_info = configs_dict.get(config_id)
        if config_info:
            name = config_info['indicator_name']
            params_str_short = "-".join(f"{k}{v}" for k,v in sorted(config_info['params'].items()))
            safe_params_str = re.sub(r'[\\/*?:"<>|\s]+', '_', params_str_short)
            # Create a potentially long but descriptive identifier
            identifier = f"{name}_{config_id}_{safe_params_str}"
            heatmap_data[identifier] = corr_values
        else:
             logger.warning(f"Config info not found for ID {config_id} during heatmap data preparation.")

    if not heatmap_data:
         logger.warning("No valid data columns to generate heatmap.")
         return

    # Create DataFrame (Lags as index, Configs as columns)
    corr_df = pd.DataFrame(heatmap_data, index=range(1, max_lag + 1))
    corr_df = corr_df.apply(pd.to_numeric, errors='coerce') # Ensure numeric
    corr_df.replace([np.inf, -np.inf], np.nan, inplace=True) # Handle infinities

    # Drop columns/rows that are entirely NaN
    corr_df.dropna(axis=1, how='all', inplace=True)
    corr_df.dropna(axis=0, how='all', inplace=True)

    if corr_df.empty:
        logger.warning("Heatmap DataFrame is empty after initial NaN drop.")
        return

    # --- Filtering Step ---
    num_columns_before_filter = len(corr_df.columns)
    filtered_df = corr_df.copy()
    # Apply filtering ONLY if the number of columns exceeds the limit
    # Tweak path optimizer should already return the target number, so filtering is mainly for Default path.
    if num_columns_before_filter > config.HEATMAP_MAX_CONFIGS:
        filter_desc = f"Filtering heatmap from {num_columns_before_filter} to top {config.HEATMAP_MAX_CONFIGS}"
        if is_tweak_path: filter_desc += " (Note: Optimizer should have limited already)"
        logger.info(filter_desc)

        try:
            # Calculate score for filtering (e.g., max absolute correlation per config)
            col_scores = filtered_df.abs().max(skipna=True).dropna() # Max abs value in each column
            if not col_scores.empty:
                 col_scores = col_scores.sort_values(ascending=False)
                 top_cols = col_scores.head(config.HEATMAP_MAX_CONFIGS).index
                 filtered_df = filtered_df[top_cols]
                 logger.info(f"Filtered heatmap to {len(filtered_df.columns)} columns based on max abs correlation.")
            else:
                 logger.warning("Could not calculate scores for filtering heatmap columns. Using original set.")
        except Exception as filter_e:
             logger.error(f"Error during heatmap column filtering: {filter_e}. Proceeding with unfiltered columns.", exc_info=True)

    # Drop again after potential filtering
    filtered_df.dropna(axis=1, how='all', inplace=True)
    filtered_df.dropna(axis=0, how='all', inplace=True)

    if filtered_df.empty:
        logger.warning("Heatmap DataFrame is empty after filtering/dropping NaNs.")
        return

    # --- Sorting Step ---
    sorted_df = filtered_df.copy()
    sort_description = "Original Order"
    try:
         # Sort columns by mean absolute correlation (descending) for better visual structure
         sort_metric_values = sorted_df.abs().mean(skipna=True).dropna()
         if not sort_metric_values.empty:
              sorted_cols = sort_metric_values.sort_values(ascending=False).index
              sorted_df = sorted_df[sorted_cols] # Reorder columns
              sort_description = "Sorted by Mean Abs Correlation"
              logger.info("Sorted heatmap columns by mean absolute correlation.")
         else:
              logger.warning("Could not calculate mean correlations for sorting heatmap columns.")
    except Exception as sort_e:
         logger.warning(f"Could not sort heatmap columns: {sort_e}. Using default/filtered order.", exc_info=True)

    if sorted_df.empty:
        logger.warning("Cannot plot heatmap, DataFrame is empty after sorting attempts.")
        return

    # --- Plotting ---
    num_configs_plot = len(sorted_df.columns)
    num_lags_plot = len(sorted_df.index)

    # Dynamically adjust figure size (with limits)
    # More width for lags, more height for configs
    fig_width = max(15, min(60, num_lags_plot * 0.15 + 5))
    fig_height = max(10, min(80, num_configs_plot * 0.25 + 2))

    plt.figure(figsize=(fig_width, fig_height), dpi=config.PLOT_DPI)

    sns.heatmap(
        sorted_df.T, # Transpose so configs are on Y-axis, lags on X-axis
        annot=False, # Annotations usually too dense
        cmap='coolwarm', # Diverging colormap centered at 0
        center=0,
        linewidths=0.1,
        linecolor='lightgrey',
        cbar=True,
        vmin=-1.0, # Ensure colorbar covers full potential range
        vmax=1.0,
        cbar_kws={'shrink': 0.6} # Adjust colorbar size
    )

    plt.title(f"Indicator Correlation with Future Close Price vs. Lag\n({num_configs_plot} Configs, {sort_description})", fontsize=14)
    plt.xlabel("Lag (Periods)", fontsize=12)
    plt.ylabel("Indicator Configuration", fontsize=12)

    # Adjust tick label frequency to prevent overlap
    x_tick_labels = sorted_df.index # Lags
    num_x_labels_desired = 50 # Aim for max 50 labels on x-axis
    x_tick_step = max(1, len(x_tick_labels) // num_x_labels_desired)
    new_x_ticks = np.arange(len(x_tick_labels))[::x_tick_step] # Indices for ticks
    new_x_labels = x_tick_labels[::x_tick_step] # Actual lag values for labels

    y_tick_labels = sorted_df.columns # Config identifiers
    num_y_labels_desired = 60 # Aim for max 60 labels on y-axis
    y_tick_step = max(1, len(y_tick_labels) // num_y_labels_desired)
    new_y_ticks = np.arange(len(y_tick_labels))[::y_tick_step] # Indices for ticks
    new_y_labels = y_tick_labels[::y_tick_step] # Actual config identifiers for labels

    # Adjust font size dynamically based on number of labels (with min/max)
    xtick_fontsize = max(5, min(8, 700 / len(new_x_labels))) if len(new_x_labels) > 0 else 8
    ytick_fontsize = max(5, min(8, 700 / len(new_y_labels))) if len(new_y_labels) > 0 else 8
    plt.xticks(ticks=new_x_ticks + 0.5, labels=new_x_labels, rotation=90, fontsize=xtick_fontsize)
    plt.yticks(ticks=new_y_ticks + 0.5, labels=new_y_labels, rotation=0, fontsize=ytick_fontsize)


    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout

    filepath = _prepare_filenames(output_dir, file_prefix, "Heatmap_" + sort_description.replace('|','').replace(' ',''), "heatmap")
    try:
        plt.savefig(filepath)
        logger.info(f"Saved heatmap: {filepath.name}")
    except Exception as e:
        logger.error(f"Failed to save heatmap {filepath.name}: {e}", exc_info=True)
    finally:
        plt.close() # Close the figure


# --- generate_correlation_envelope_chart remains the same ---
def generate_correlation_envelope_chart(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]], # Needed to map config_id back if required
    max_lag: int,
    output_dir: Path,
    file_prefix: str
) -> None:
    """
    Generates an area chart showing the max positive (green area from 0 up)
    and min negative (red area from 0 down) correlation across all valid indicators
    for each lag. Axes intersect at y=0.
    """
    logger.info("Generating correlation envelope area chart...")
    if max_lag <= 0 or not correlations_by_config_id:
        logger.warning("No data or invalid lag for envelope chart.")
        return

    # --- 1. Data Transformation ---
    # Create a DataFrame where columns are configs and rows are lags
    valid_data = {}
    for config_id, corr_values_full in correlations_by_config_id.items():
        if (corr_values_full is not None and
            isinstance(corr_values_full, list) and
            len(corr_values_full) >= max_lag):
            # Only include configs with data up to max_lag
            corr_values = corr_values_full[:max_lag]
            # Only include if there's at least one non-NaN value
            if not all(pd.isna(c) for c in corr_values):
                 valid_data[config_id] = corr_values # Use config_id as key for now
        else:
             logger.debug(f"Excluding Config ID {config_id} from envelope chart due to insufficient data length.")

    if not valid_data:
         logger.warning("No valid configurations with sufficient data to generate envelope chart.")
         return

    corr_df = pd.DataFrame(valid_data, index=range(1, max_lag + 1))
    corr_df = corr_df.apply(pd.to_numeric, errors='coerce') # Ensure numeric
    corr_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if corr_df.empty or corr_df.shape[1] == 0:
        logger.warning("Envelope chart DataFrame is empty after initial processing.")
        return

    # --- 2. Calculate Envelope Lines ---
    # Calculate max and min correlation across all configs for each lag, ignoring NaNs
    max_positive_corr = corr_df.max(axis=1, skipna=True)
    min_negative_corr = corr_df.min(axis=1, skipna=True)

    # Check if any valid envelope data exists
    if max_positive_corr.isna().all() and min_negative_corr.isna().all():
        logger.warning("Max positive and min negative correlations are all NaN. Cannot plot envelope chart.")
        return

    # --- 3. Plotting as Area Chart ---
    lags = corr_df.index
    fig, ax = plt.subplots(figsize=(15, 8), dpi=config.PLOT_DPI)

    # Fill positive area: Use .where() to fill only when max_positive_corr >= 0. Fill with 0 otherwise.
    # Use fillna(0) to handle cases where a lag might have only NaNs or only negative values
    ax.fill_between(lags, 0, max_positive_corr.where(max_positive_corr >= 0, 0).fillna(0),
                    facecolor='green', alpha=0.4, interpolate=True, label='Max Positive Correlation Range', zorder=2)

    # Fill negative area: Use .where() to fill only when min_negative_corr <= 0. Fill with 0 otherwise.
    ax.fill_between(lags, 0, min_negative_corr.where(min_negative_corr <= 0, 0).fillna(0),
                    facecolor='red', alpha=0.4, interpolate=True, label='Min Negative Correlation Range', zorder=2)

    # Optionally plot the boundary lines themselves for clarity, handling NaNs
    ax.plot(lags, max_positive_corr.fillna(method='ffill').fillna(0), color='darkgreen', linewidth=0.8, alpha=0.7, label='Max Positive Envelope', zorder=3) # Fill NaNs for line
    ax.plot(lags, min_negative_corr.fillna(method='ffill').fillna(0), color='darkred', linewidth=0.8, alpha=0.7, label='Min Negative Envelope', zorder=3) # Fill NaNs for line

    # --- Formatting ---
    ax.set_title(f'Correlation Envelope vs. Lag ({corr_df.shape[1]} Configs)', fontsize=14)
    ax.set_xlabel('Lag (Periods)', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, max_lag + 1)

    _set_axis_intersection_at_zero(ax) # Apply axis change
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1) # Send grid behind areas
    ax.legend(loc='lower right', fontsize='small')
    fig.tight_layout()

    # --- Saving ---
    filepath = _prepare_filenames(output_dir, file_prefix, "Correlation_Envelope_Area", "chart")
    try:
        fig.savefig(filepath)
        logger.info(f"Saved correlation envelope area chart: {filepath.name}")
    except Exception as e:
        logger.error(f"Failed to save correlation envelope area chart {filepath.name}: {e}", exc_info=True)
    finally:
        plt.close(fig) # Close the figure

# --- NEW FUNCTION ---
def generate_peak_correlation_report(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str
) -> None:
    """
    Generates a summary report (CSV and console table) of peak correlations.
    """
    logger.info("Generating peak correlation summary report...")
    if not correlations_by_config_id or max_lag <= 0:
        logger.warning("No correlation data or invalid max_lag provided for report.")
        return

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed}
    report_data = []

    for config_id, corr_values_full in correlations_by_config_id.items():
        config_info = configs_dict.get(config_id)
        if not config_info:
            logger.warning(f"Config info missing for ID {config_id} in report generation.")
            continue

        indicator_name = config_info['indicator_name']
        params = config_info.get('params', {})
        params_str = json.dumps(params, separators=(',',':')) # Compact JSON string

        if corr_values_full is None or not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag:
            logger.warning(f"Insufficient correlation data length for {indicator_name} (ID {config_id}). Skipping report entry.")
            continue

        corr_values = corr_values_full[:max_lag]
        corr_array = np.array(corr_values, dtype=float) # Convert Nones to NaN automatically

        # Skip if all correlations are NaN for this config
        if np.isnan(corr_array).all():
            logger.info(f"Skipping report entry for {indicator_name} (ID {config_id}) as all correlations are NaN.")
            report_data.append({
                'Config ID': config_id,
                'Indicator': indicator_name,
                'Parameters': params_str,
                'Peak Positive Corr': np.nan, 'Peak Positive Lag': np.nan,
                'Peak Negative Corr': np.nan, 'Peak Negative Lag': np.nan,
                'Peak Absolute Corr': np.nan, 'Peak Absolute Lag': np.nan,
            })
            continue

        # Calculate Peak Positive
        peak_pos_corr = np.nanmax(corr_array) if not np.isnan(corr_array).all() else np.nan
        peak_pos_idx = np.nanargmax(corr_array) if not np.isnan(peak_pos_corr) else -1
        peak_pos_lag = peak_pos_idx + 1 if peak_pos_idx != -1 else np.nan

        # Calculate Peak Negative
        peak_neg_corr = np.nanmin(corr_array) if not np.isnan(corr_array).all() else np.nan
        peak_neg_idx = np.nanargmin(corr_array) if not np.isnan(peak_neg_corr) else -1
        peak_neg_lag = peak_neg_idx + 1 if peak_neg_idx != -1 else np.nan

        # Calculate Peak Absolute
        abs_corr_array = np.abs(corr_array)
        peak_abs_corr = np.nanmax(abs_corr_array) if not np.isnan(abs_corr_array).all() else np.nan
        peak_abs_idx = np.nanargmax(abs_corr_array) if not np.isnan(peak_abs_corr) else -1
        peak_abs_lag = peak_abs_idx + 1 if peak_abs_idx != -1 else np.nan

        report_data.append({
            'Config ID': config_id,
            'Indicator': indicator_name,
            'Parameters': params_str,
            'Peak Positive Corr': peak_pos_corr,
            'Peak Positive Lag': peak_pos_lag,
            'Peak Negative Corr': peak_neg_corr,
            'Peak Negative Lag': peak_neg_lag,
            'Peak Absolute Corr': peak_abs_corr,
            'Peak Absolute Lag': peak_abs_lag,
        })

    if not report_data:
        logger.warning("No data available to generate peak correlation report.")
        return

    # Create DataFrame and Sort
    report_df = pd.DataFrame(report_data)
    report_df.sort_values('Peak Absolute Corr', ascending=False, inplace=True, na_position='last')

    # --- Print to Console ---
    print("\n\n--- Peak Correlation Summary ---")
    # Format for printing
    report_df_print = report_df.copy()
    float_cols = ['Peak Positive Corr', 'Peak Negative Corr', 'Peak Absolute Corr']
    lag_cols = ['Peak Positive Lag', 'Peak Negative Lag', 'Peak Absolute Lag']
    for col in float_cols: report_df_print[col] = report_df_print[col].map('{:.4f}'.format)
    for col in lag_cols: report_df_print[col] = report_df_print[col].map('{:.0f}'.format).replace('nan', 'N/A') # Format lag as int, handle NaN

    # Limit parameter string length for console display
    max_param_len = 50
    report_df_print['Parameters'] = report_df_print['Parameters'].apply(lambda x: x if len(x) <= max_param_len else x[:max_param_len-3] + '...')

    try:
         # Use pandas to_string for simple table output
         print(report_df_print.to_string(index=False, max_rows=None)) # Print all rows
    except Exception as print_e:
         logger.error(f"Error printing report to console: {print_e}")
         # Fallback: print basic info
         print(report_df_print.head())

    # --- Save to CSV ---
    csv_filename = f"{file_prefix}_peak_correlation_report.csv"
    csv_filepath = output_dir / csv_filename
    try:
        # Save the original DataFrame with full precision
        report_df.to_csv(csv_filepath, index=False, float_format='%.6f')
        logger.info(f"Saved peak correlation report to: {csv_filepath}")
        print(f"\nPeak correlation report saved to: {csv_filepath}")
    except Exception as e:
        logger.error(f"Failed to save peak correlation report CSV to {csv_filepath}: {e}", exc_info=True)
        print(f"Error: Failed to save report CSV file.")