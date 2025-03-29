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
import json
import re

import config
import utils
# Need these for getting default params/config ID in tweak mode
import indicator_factory
# import sqlite_manager # No longer needed here for default ID lookup

logger = logging.getLogger(__name__)

# --- _prepare_filenames, _set_axis_intersection_at_zero ---
# --- plot_correlation_lines ---
# --- _get_stats ---
# --- generate_combined_correlation_chart ---
# --- generate_enhanced_heatmap ---
# (Keep existing functions - no changes needed here)
def _prepare_filenames(output_dir: Path, file_prefix: str, config_identifier: str, chart_type: str) -> Path:
    """Creates a sanitized filename for the plot."""
    safe_identifier = re.sub(r'[\\/*?:"<>|\s]+', '_', config_identifier)
    max_len = 100
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
    except Exception as e:
        logger.warning(f"Could not set axis intersection at zero: {e}")

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
    rolling_window_ci = 10
    ci_multiplier = 1.96

    plotted_count = 0
    for config_id, corr_values_full in correlations_by_config_id.items():
        config_info = configs_dict.get(config_id)
        if not config_info:
            logger.warning(f"Configuration info not found for Config ID {config_id}. Skipping plot.")
            continue

        indicator_name = config_info['indicator_name']
        try:
             params_str_short = "-".join(f"{k}{v}" for k,v in sorted(config_info.get('params',{}).items()))
             params_str_title = json.dumps(config_info.get('params',{}), separators=(',', ':'))
        except:
             params_str_short = str(config_info.get('params',{}))
             params_str_title = params_str_short
        safe_params_str = re.sub(r'[\\/*?:"<>|\s]+', '_', params_str_short)
        config_identifier_base = f"{indicator_name}_{config_id}_{safe_params_str}"

        if corr_values_full is None or not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag:
            logger.warning(f"Insufficient or invalid correlation data length for {config_identifier_base} (Expected {max_lag}, Got {len(corr_values_full) if corr_values_full else 0}). Skipping plot.")
            continue

        corr_values = corr_values_full[:max_lag]
        corr_series = pd.Series(corr_values, index=lags, dtype=float)
        corr_series_cleaned = corr_series.dropna()

        if len(corr_series_cleaned) < 2:
             logger.info(f"Not enough valid correlation data points (< 2) to plot for {config_identifier_base}. Skipping.")
             continue

        plot_lags = corr_series_cleaned.index
        plot_corrs = corr_series_cleaned.values
        logger.debug(f"INDIVIDUAL PLOT: ID={config_id}, Identifier={config_identifier_base}, Lags={list(plot_lags[:5])}, Corrs={list(plot_corrs[:5])}")

        rolling_std = corr_series.rolling(window=rolling_window_ci, min_periods=2, center=True).std()
        half_width = ci_multiplier * rolling_std
        upper_band = (corr_series + half_width).clip(upper=1.0)
        lower_band = (corr_series - half_width).clip(lower=-1.0)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=config.PLOT_DPI)
        ax.plot(plot_lags, plot_corrs, marker='.', linestyle='-', label=f'Correlation (Config {config_id})', zorder=3)

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

        ax.set_title(f"Correlation vs. Lag: {indicator_name}\nParams: {params_str_title} (ID: {config_id})", fontsize=10)
        ax.set_xlabel("Lag (Periods)")
        ax.set_ylabel("Pearson Correlation")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, max_lag + 1)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        _set_axis_intersection_at_zero(ax)
        ax.legend(fontsize='small')
        fig.tight_layout()

        filepath = _prepare_filenames(output_dir, file_prefix, config_identifier_base, "line_CI")
        try:
            fig.savefig(filepath); plotted_count += 1
            logger.debug(f"Saved line chart with CI: {filepath.name}")
        except Exception as e: logger.error(f"Failed to save line chart {filepath.name}: {e}", exc_info=True)
        finally: plt.close(fig)
    logger.info(f"Generated {plotted_count} separate line charts with CI bands.")

def _get_stats(corr_list: List[Optional[float]]) -> Dict[str, Optional[float]]:
    """Calculate mean, std, max of absolute correlations, handling NaNs."""
    valid_corrs = [c for c in corr_list if pd.notna(c)]
    if not valid_corrs: return {'mean_abs': None, 'std_abs': None, 'max_abs': None, 'peak_lag': None}
    abs_corrs = [abs(c) for c in valid_corrs]
    max_abs_val = np.max(abs_corrs) if abs_corrs else 0.0
    peak_lag_val = None
    if abs_corrs:
         try:
             # Use nanargmax on the absolute values after converting None/NaN to a value that won't be chosen
             corr_array_for_abs_max = np.array(corr_list, dtype=float) # Convert Nones to NaN
             peak_lag_index = np.nanargmax(np.abs(corr_array_for_abs_max))
             peak_lag_val = peak_lag_index + 1
             # Verify the value at peak_lag_val is indeed the max_abs_val (or close)
             if not np.isclose(abs(corr_list[peak_lag_index]), max_abs_val):
                  logger.warning(f"Peak lag identification mismatch: index {peak_lag_index} value {corr_list[peak_lag_index]}, max abs value {max_abs_val}. Re-checking...")
                  first_max_index = np.where(np.isclose(np.abs(corr_array_for_abs_max), max_abs_val))[0]
                  if len(first_max_index) > 0:
                      peak_lag_index = first_max_index[0]
                      peak_lag_val = peak_lag_index + 1
                  else: peak_lag_val = None
         except (ValueError, IndexError) as e:
             logger.warning(f"Could not determine peak lag accurately: {e}")
             peak_lag_val = None
    return {'mean_abs': np.mean(abs_corrs) if abs_corrs else None, 'std_abs': np.std(abs_corrs) if abs_corrs else None, 'max_abs': max_abs_val if abs_corrs else None, 'peak_lag': peak_lag_val }

def generate_combined_correlation_chart(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str
) -> None:
    """Generates a combined line chart showing correlations for ALL valid indicator configs. Axes intersect at y=0."""
    logger.info(f"Generating combined correlation chart for ALL configurations...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data or invalid lag for combined chart."); return

    lags = list(range(1, max_lag + 1))
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed}
    plot_data = []
    for config_id, corr_values_full in correlations_by_config_id.items():
         if corr_values_full is None or not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag:
             logger.warning(f"Skipping config ID {config_id} in combined chart prep due to invalid/insufficient data length ({len(corr_values_full) if corr_values_full else 0} < {max_lag}).")
             continue
         corr_values = corr_values_full[:max_lag]
         stats = _get_stats(corr_values)
         if stats['max_abs'] is not None and stats['max_abs'] > 1e-6:
             config_info = configs_dict.get(config_id)
             if config_info:
                 name = config_info['indicator_name']
                 params_str_short = "-".join(f"{k}{v}" for k,v in sorted(config_info.get('params',{}).items()))
                 safe_params_str = re.sub(r'[\\/*?:"<>|\s]+', '_', params_str_short)
                 identifier = f"{name}_{config_id}_{safe_params_str}"
                 plot_data.append({ 'config_id': config_id, 'identifier': identifier, 'correlations': corr_values, 'mean_abs': stats['mean_abs'], 'max_abs': stats['max_abs'], 'peak_lag': stats['peak_lag'] if stats['peak_lag'] is not None else max_lag + 1 })
         else: logger.debug(f"Skipping config ID {config_id} from combined plot as max absolute correlation is near zero or NaN.")
    if not plot_data: logger.warning("No valid indicator data to plot for combined chart after filtering weak signals."); return

    plot_data.sort(key=lambda x: x.get('max_abs', 0), reverse=True)
    logger.info("Sorted configurations by max absolute correlation (desc) for combined plot.")
    fig, ax = plt.subplots(figsize=(15, 10), dpi=config.PLOT_DPI)
    plotted_count = 0
    max_lines_on_combined = config.HEATMAP_MAX_CONFIGS
    plot_subset = plot_data[:max_lines_on_combined]
    if len(plot_data) > max_lines_on_combined: logger.warning(f"Plotting only the top {max_lines_on_combined} configurations (sorted by max abs corr) on the combined chart due to high count ({len(plot_data)}).")

    for item in plot_subset:
        identifier = item['identifier']
        config_id_plotting = item['config_id']
        corr_values_list = item['correlations']
        corr_series = pd.Series(corr_values_list, index=lags, dtype=float)
        corr_series_cleaned = corr_series.dropna()
        if len(corr_series_cleaned) < 2: logger.warning(f"Not enough valid points for {identifier} in combined chart. Skipping line."); continue
        plot_lags = corr_series_cleaned.index
        plot_corrs = corr_series_cleaned.values
        logger.debug(f"COMBINED PLOT (Top {max_lines_on_combined}): ID={config_id_plotting}, Identifier={identifier}, Lags={list(plot_lags[:5])}, Corrs={list(plot_corrs[:5])}")
        ax.plot(plot_lags, plot_corrs, marker='.', markersize=1, linestyle='-', linewidth=0.7, alpha=0.6, label=identifier)
        plotted_count += 1
    if plotted_count == 0: logger.warning("No lines were plotted for the combined chart."); plt.close(fig); return

    chart_title = f"Combined Correlation vs. Lag"
    if len(plot_data) > max_lines_on_combined: chart_title += f" (Top {plotted_count} of {len(plot_data)} Configs by Max Abs Corr)"
    else: chart_title += f" ({plotted_count} Configurations)"
    ax.set_title(chart_title, fontsize=12)
    ax.set_xlabel("Lag (Periods)"); ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, max_lag + 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
    _set_axis_intersection_at_zero(ax)
    if plotted_count <= 30: ax.legend(loc='best', fontsize='xx-small', ncol=2 if plotted_count > 15 else 1)
    else: logger.warning(f"Hiding legend for combined chart as there are too many lines ({plotted_count}).")
    fig.tight_layout()

    filename_suffix = f"Combined_{plotted_count}Configs" if len(plot_data) > max_lines_on_combined else f"Combined_All_{plotted_count}"
    filepath = _prepare_filenames(output_dir, file_prefix, filename_suffix, "chart")
    try:
        fig.savefig(filepath); logger.info(f"Saved combined chart: {filepath.name}")
    except Exception as e: logger.error(f"Failed to save combined chart {filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig)

def generate_enhanced_heatmap(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str,
    is_tweak_path: bool
) -> None:
    """
    Generates an enhanced heatmap of correlation values.
    Filters to HEATMAP_MAX_CONFIGS if needed. Sorts columns by absolute mean correlation.
    """
    logger.info("Generating enhanced correlation heatmap...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data or invalid lag for heatmap."); return

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed}
    heatmap_data = {}
    valid_config_ids = list(correlations_by_config_id.keys())
    for config_id in valid_config_ids:
        corr_values_full = correlations_by_config_id.get(config_id)
        if (corr_values_full is None or not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag):
             logger.warning(f"Skipping config ID {config_id} in heatmap prep due to insufficient data length."); continue
        corr_values = corr_values_full[:max_lag]
        if all(pd.isna(c) for c in corr_values):
             logger.warning(f"Skipping config ID {config_id} in heatmap prep as all correlations up to lag {max_lag} are NaN."); continue
        config_info = configs_dict.get(config_id)
        if config_info:
            name = config_info['indicator_name']
            params_str_short = "-".join(f"{k}{v}" for k,v in sorted(config_info.get('params',{}).items()))
            safe_params_str = re.sub(r'[\\/*?:"<>|\s]+', '_', params_str_short)
            identifier = f"{name}_{config_id}_{safe_params_str}"
            heatmap_data[identifier] = corr_values
        else: logger.warning(f"Config info not found for ID {config_id} during heatmap data preparation.")
    if not heatmap_data: logger.warning("No valid data columns to generate heatmap."); return

    corr_df = pd.DataFrame(heatmap_data, index=range(1, max_lag + 1))
    corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
    corr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    corr_df.dropna(axis=1, how='all', inplace=True)
    corr_df.dropna(axis=0, how='all', inplace=True)
    if corr_df.empty: logger.warning("Heatmap DataFrame is empty after initial NaN drop."); return

    num_columns_before_filter = len(corr_df.columns)
    filtered_df = corr_df.copy()
    if num_columns_before_filter > config.HEATMAP_MAX_CONFIGS:
        filter_desc = f"Filtering heatmap from {num_columns_before_filter} to top {config.HEATMAP_MAX_CONFIGS}"
        # if is_tweak_path: filter_desc += " (Note: Optimizer results)"
        logger.info(filter_desc)
        try:
            col_scores = filtered_df.abs().max(skipna=True).dropna()
            if not col_scores.empty:
                 col_scores = col_scores.sort_values(ascending=False)
                 top_cols = col_scores.head(config.HEATMAP_MAX_CONFIGS).index
                 filtered_df = filtered_df[top_cols]
                 logger.info(f"Filtered heatmap to {len(filtered_df.columns)} columns based on max abs correlation.")
            else: logger.warning("Could not calculate scores for filtering heatmap columns. Using original set.")
        except Exception as filter_e: logger.error(f"Error during heatmap column filtering: {filter_e}. Proceeding with unfiltered columns.", exc_info=True)
    filtered_df.dropna(axis=1, how='all', inplace=True)
    filtered_df.dropna(axis=0, how='all', inplace=True)
    if filtered_df.empty: logger.warning("Heatmap DataFrame is empty after filtering/dropping NaNs."); return

    sorted_df = filtered_df.copy()
    sort_description = "Original Order"
    try:
         sort_metric_values = sorted_df.abs().mean(skipna=True).dropna()
         if not sort_metric_values.empty:
              sorted_cols = sort_metric_values.sort_values(ascending=False).index
              sorted_df = sorted_df[sorted_cols]
              sort_description = "Sorted by Mean Abs Correlation"
              logger.info("Sorted heatmap columns by mean absolute correlation.")
         else: logger.warning("Could not calculate mean correlations for sorting heatmap columns.")
    except Exception as sort_e: logger.warning(f"Could not sort heatmap columns: {sort_e}. Using default/filtered order.", exc_info=True)
    if sorted_df.empty: logger.warning("Cannot plot heatmap, DataFrame is empty after sorting attempts."); return

    num_configs_plot = len(sorted_df.columns)
    num_lags_plot = len(sorted_df.index)
    fig_width = max(15, min(60, num_lags_plot * 0.15 + 5))
    fig_height = max(10, min(80, num_configs_plot * 0.25 + 2))
    plt.figure(figsize=(fig_width, fig_height), dpi=config.PLOT_DPI)

    sns.heatmap( sorted_df.T, annot=False, cmap='coolwarm', center=0, linewidths=0.1, linecolor='lightgrey', cbar=True, vmin=-1.0, vmax=1.0, cbar_kws={'shrink': 0.6} )
    plt.title(f"Indicator Correlation with Future Close Price vs. Lag\n({num_configs_plot} Configs, {sort_description})", fontsize=14)
    plt.xlabel("Lag (Periods)", fontsize=12)
    plt.ylabel("Indicator Configuration", fontsize=12)

    x_tick_labels = sorted_df.index; num_x_labels_desired = 50
    x_tick_step = max(1, len(x_tick_labels) // num_x_labels_desired)
    new_x_ticks = np.arange(len(x_tick_labels))[::x_tick_step]
    new_x_labels = x_tick_labels[::x_tick_step]
    y_tick_labels = sorted_df.columns; num_y_labels_desired = 60
    y_tick_step = max(1, len(y_tick_labels) // num_y_labels_desired)
    new_y_ticks = np.arange(len(y_tick_labels))[::y_tick_step]
    new_y_labels = y_tick_labels[::y_tick_step]
    xtick_fontsize = max(5, min(8, 700 / len(new_x_labels))) if len(new_x_labels) > 0 else 8
    ytick_fontsize = max(5, min(8, 700 / len(new_y_labels))) if len(new_y_labels) > 0 else 8
    plt.xticks(ticks=new_x_ticks + 0.5, labels=new_x_labels, rotation=90, fontsize=xtick_fontsize)
    plt.yticks(ticks=new_y_ticks + 0.5, labels=new_y_labels, rotation=0, fontsize=ytick_fontsize)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    filepath = _prepare_filenames(output_dir, file_prefix, "Heatmap_" + sort_description.replace('|','').replace(' ',''), "heatmap")
    try:
        plt.savefig(filepath); logger.info(f"Saved heatmap: {filepath.name}")
    except Exception as e: logger.error(f"Failed to save heatmap {filepath.name}: {e}", exc_info=True)
    finally: plt.close()


# --- UPDATED FUNCTION ---
def generate_correlation_envelope_chart(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]], # List of configs passed for visualization
    max_lag: int,
    output_dir: Path,
    file_prefix: str,
    is_tweak_path: bool
) -> None:
    """
    Generates an area chart showing the max positive (green area >= 0)
    and min negative (red area <= 0) correlation across all valid indicators
    for each lag. Lines also only plot where sign condition is met.
    Optionally plots the default config's correlation line in tweak mode.
    """
    logger.info("Generating correlation envelope area chart...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data or invalid lag for envelope chart."); return

    valid_data = {}
    for config_id, corr_values_full in correlations_by_config_id.items():
        if (corr_values_full is not None and isinstance(corr_values_full, list) and len(corr_values_full) >= max_lag):
            corr_values = corr_values_full[:max_lag]
            if not all(pd.isna(c) for c in corr_values): valid_data[config_id] = corr_values
        else: logger.debug(f"Excluding Config ID {config_id} from envelope chart due to insufficient data length.")
    if not valid_data: logger.warning("No valid configurations with sufficient data to generate envelope chart."); return

    corr_df = pd.DataFrame(valid_data, index=range(1, max_lag + 1))
    corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
    corr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if corr_df.empty or corr_df.shape[1] == 0: logger.warning("Envelope chart DataFrame is empty after initial processing."); return

    max_positive_corr = corr_df.max(axis=1, skipna=True)
    min_negative_corr = corr_df.min(axis=1, skipna=True)
    if max_positive_corr.isna().all() and min_negative_corr.isna().all(): logger.warning("Max positive and min negative correlations are all NaN. Cannot plot envelope chart."); return

    default_corr_series = None; default_config_id = None; tweaked_indicator_name = None; default_params = None
    if is_tweak_path and indicator_configs_processed:
        try:
            tweaked_indicator_name = indicator_configs_processed[0]['indicator_name']
            indicator_factory._load_indicator_definitions()
            indicator_def = indicator_factory._get_indicator_definition(tweaked_indicator_name)
            if indicator_def:
                default_params = {k: v.get('default') for k, v in indicator_def.get('parameters', {}).items() if 'default' in v}
                if default_params:
                    found_default_id = None
                    # Search the list passed to this function
                    for cfg in indicator_configs_processed:
                        if utils.compare_param_dicts(cfg.get('params',{}), default_params):
                             found_default_id = cfg.get('config_id')
                             break
                    if found_default_id and found_default_id in correlations_by_config_id:
                        default_config_id = found_default_id
                        default_corr_list = correlations_by_config_id[default_config_id][:max_lag]
                        if not all(pd.isna(c) for c in default_corr_list):
                             default_corr_series = pd.Series(default_corr_list, index=range(1, max_lag + 1))
                             logger.info(f"Found correlation data for default config (ID: {default_config_id}) of '{tweaked_indicator_name}'.")
                        else: logger.warning(f"Default config (ID: {default_config_id}) for '{tweaked_indicator_name}' has only NaN correlations.")
                    # This warning was the source of the previous confusion - it might fire if default wasn't in top N
                    else: logger.warning(f"Default config for '{tweaked_indicator_name}' (Params: {default_params}) not found in the provided list/correlation data for visualization.")
                else: logger.warning(f"Could not determine default parameters for '{tweaked_indicator_name}'.")
            else: logger.warning(f"Could not find definition for tweaked indicator '{tweaked_indicator_name}'.")
        except Exception as e: logger.error(f"Error retrieving default config data for tweak mode: {e}", exc_info=True)

    lags = corr_df.index
    fig, ax = plt.subplots(figsize=(15, 8), dpi=config.PLOT_DPI)
    ax.fill_between(lags, 0, max_positive_corr.where(max_positive_corr >= 0, 0).fillna(0),
                    facecolor='mediumseagreen', alpha=0.4, interpolate=True, label='Max Positive Correlation Range', zorder=2)
    ax.fill_between(lags, 0, min_negative_corr.where(min_negative_corr <= 0, 0).fillna(0),
                    facecolor='lightcoral', alpha=0.4, interpolate=True, label='Min Negative Correlation Range', zorder=2)
    ax.plot(lags, max_positive_corr.where(max_positive_corr >= 0), color='darkgreen', linewidth=0.8, alpha=0.7, label='Max Positive Envelope', zorder=3)
    ax.plot(lags, min_negative_corr.where(min_negative_corr <= 0), color='darkred', linewidth=0.8, alpha=0.7, label='Min Negative Envelope', zorder=3)

    if default_corr_series is not None:
        default_corr_cleaned = default_corr_series.dropna()
        if len(default_corr_cleaned) >= 2:
            ax.plot(default_corr_cleaned.index, default_corr_cleaned.values,
                    color='darkblue', linestyle='--', linewidth=1.0, alpha=0.8,
                    label=f"Default '{tweaked_indicator_name}' Config (ID: {default_config_id})", zorder=4)
        else: logger.warning(f"Not enough valid points to plot default config line for {tweaked_indicator_name}.")

    title = f'Correlation Envelope vs. Lag ({corr_df.shape[1]} Configs)'
    if is_tweak_path and tweaked_indicator_name: title += f" - Optimized: {tweaked_indicator_name}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Lag (Periods)', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, max_lag + 1)
    _set_axis_intersection_at_zero(ax)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
    ax.legend(loc='lower right', fontsize='small')
    fig.tight_layout()

    filepath = _prepare_filenames(output_dir, file_prefix, "Correlation_Envelope_Area", "chart")
    try:
        fig.savefig(filepath)
        logger.info(f"Saved correlation envelope area chart: {filepath.name}")
    except Exception as e: logger.error(f"Failed to save correlation envelope area chart {filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig)


# --- generate_peak_correlation_report unchanged ---
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
    if not correlations_by_config_id or max_lag <= 0: logger.warning("No correlation data or invalid max_lag provided for report."); return

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed}
    report_data = []
    for config_id, corr_values_full in correlations_by_config_id.items():
        config_info = configs_dict.get(config_id)
        if not config_info: logger.warning(f"Config info missing for ID {config_id} in report generation."); continue
        indicator_name = config_info['indicator_name']
        params = config_info.get('params', {}); params_str = json.dumps(params, separators=(',',':'))
        if corr_values_full is None or not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag: logger.warning(f"Insufficient correlation data length for {indicator_name} (ID {config_id}). Skipping report entry."); continue
        corr_values = corr_values_full[:max_lag]
        corr_array = np.array(corr_values, dtype=float)
        if np.isnan(corr_array).all():
            logger.info(f"Skipping report entry for {indicator_name} (ID {config_id}) as all correlations are NaN.")
            report_data.append({'Config ID': config_id, 'Indicator': indicator_name, 'Parameters': params_str, 'Peak Positive Corr': np.nan, 'Peak Positive Lag': np.nan, 'Peak Negative Corr': np.nan, 'Peak Negative Lag': np.nan, 'Peak Absolute Corr': np.nan, 'Peak Absolute Lag': np.nan, }); continue

        peak_pos_corr = np.nanmax(corr_array) if not np.isnan(corr_array).all() else np.nan
        peak_pos_idx = np.nanargmax(corr_array) if pd.notna(peak_pos_corr) else -1
        peak_pos_lag = peak_pos_idx + 1 if peak_pos_idx != -1 else np.nan
        peak_neg_corr = np.nanmin(corr_array) if not np.isnan(corr_array).all() else np.nan
        peak_neg_idx = np.nanargmin(corr_array) if pd.notna(peak_neg_corr) else -1
        peak_neg_lag = peak_neg_idx + 1 if peak_neg_idx != -1 else np.nan
        abs_corr_array = np.abs(corr_array)
        peak_abs_corr = np.nanmax(abs_corr_array) if not np.isnan(abs_corr_array).all() else np.nan
        peak_abs_idx = np.nanargmax(abs_corr_array) if pd.notna(peak_abs_corr) else -1
        peak_abs_lag = peak_abs_idx + 1 if peak_abs_idx != -1 else np.nan
        report_data.append({ 'Config ID': config_id, 'Indicator': indicator_name, 'Parameters': params_str, 'Peak Positive Corr': peak_pos_corr, 'Peak Positive Lag': peak_pos_lag, 'Peak Negative Corr': peak_neg_corr, 'Peak Negative Lag': peak_neg_lag, 'Peak Absolute Corr': peak_abs_corr, 'Peak Absolute Lag': peak_abs_lag, })
    if not report_data: logger.warning("No data available to generate peak correlation report."); return

    report_df = pd.DataFrame(report_data)
    report_df.sort_values('Peak Absolute Corr', ascending=False, inplace=True, na_position='last')
    print("\n\n--- Peak Correlation Summary ---")
    report_df_print = report_df.copy()
    float_cols = ['Peak Positive Corr', 'Peak Negative Corr', 'Peak Absolute Corr']; lag_cols = ['Peak Positive Lag', 'Peak Negative Lag', 'Peak Absolute Lag']
    for col in float_cols: report_df_print[col] = report_df_print[col].map('{:.4f}'.format)
    for col in lag_cols: report_df_print[col] = report_df_print[col].map('{:.0f}'.format).replace('nan', 'N/A')
    max_param_len = 50
    report_df_print['Parameters'] = report_df_print['Parameters'].apply(lambda x: x if len(x) <= max_param_len else x[:max_param_len-3] + '...')
    try: print(report_df_print.to_string(index=False, max_rows=None))
    except Exception as print_e: logger.error(f"Error printing report to console: {print_e}"); print(report_df_print.head())
    csv_filename = f"{file_prefix}_peak_correlation_report.csv"
    csv_filepath = output_dir / csv_filename
    try:
        report_df.to_csv(csv_filepath, index=False, float_format='%.6f')
        logger.info(f"Saved peak correlation report to: {csv_filepath}")
        print(f"\nPeak correlation report saved to: {csv_filepath}")
    except Exception as e: logger.error(f"Failed to save peak correlation report CSV to {csv_filepath}: {e}", exc_info=True); print(f"Error: Failed to save report CSV file.")