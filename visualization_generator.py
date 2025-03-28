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

import config
import utils

logger = logging.getLogger(__name__)

def _prepare_filenames(output_dir: Path, file_prefix: str, config_identifier: str, chart_type: str) -> Path:
    """Creates a sanitized filename for the plot."""
    safe_identifier = config_identifier.replace(' ', '_').replace('/', '-').replace('\\', '-')
    max_len = 100
    if len(safe_identifier) > max_len:
        safe_identifier = safe_identifier[:max_len] + "_etc"

    filename = f"{file_prefix}_{safe_identifier}_{chart_type}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename

def _set_axis_intersection_at_zero(ax):
    """Moves the x-axis spine to y=0."""
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

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
             import json
             params_str = json.dumps(config_info['params'], separators=(',', ':'))
        except:
             params_str = str(config_info['params'])
        config_identifier_base = utils.get_config_identifier(indicator_name, config_id, None)

        if corr_values_full is None or len(corr_values_full) < max_lag:
            logger.warning(f"Insufficient correlation data length for {config_identifier_base} (Expected {max_lag}, Got {len(corr_values_full) if corr_values_full else 0}). Skipping plot.")
            continue
        corr_values = corr_values_full[:max_lag]

        corr_series = pd.Series(corr_values, index=lags, dtype=float)
        corr_series_cleaned = corr_series.dropna()

        if len(corr_series_cleaned) < 2:
             logger.info(f"Not enough valid correlation data points to plot for {config_identifier_base}. Skipping.")
             continue

        plot_lags = corr_series_cleaned.index
        plot_corrs = corr_series_cleaned.values

        logger.debug(f"INDIVIDUAL PLOT: ID={config_id}, Identifier={config_identifier_base}, Lags={list(plot_lags[:5])}, Corrs={list(plot_corrs[:5])}")

        rolling_std = corr_series.rolling(window=rolling_window_ci, min_periods=2).std()
        half_width = ci_multiplier * rolling_std
        upper_band = (corr_series + half_width).clip(upper=1.0)
        lower_band = (corr_series - half_width).clip(lower=-1.0)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=config.PLOT_DPI)
        ax.plot(plot_lags, plot_corrs, marker='.', linestyle='-', label=f'Correlation (Config {config_id})', zorder=3)

        valid_band_lags = rolling_std.dropna().index
        ax.fill_between(valid_band_lags,
                         lower_band.loc[valid_band_lags],
                         upper_band.loc[valid_band_lags],
                         color='skyblue', alpha=0.3,
                         label=f'± {ci_multiplier} * {rolling_window_ci}-Lag Rolling SD',
                         zorder=2)

        ax.set_title(f"Correlation vs. Lag: {indicator_name}\nParams: {params_str} (ID: {config_id})", fontsize=10)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Pearson Correlation with Future Close Price")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlim(0, max_lag)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        _set_axis_intersection_at_zero(ax)
        ax.legend()
        fig.tight_layout()

        filepath = _prepare_filenames(output_dir, file_prefix, config_identifier_base, "line_CI")
        try:
            fig.savefig(filepath)
            logger.debug(f"Saved line chart with CI: {filepath.name}")
            plotted_count += 1
        except Exception as e:
            logger.error(f"Failed to save line chart {filepath.name}: {e}", exc_info=True)
        finally:
            plt.close(fig)

    logger.info(f"Generated {plotted_count} line charts with CI bands.")


def _get_stats(corr_list: List[Optional[float]]) -> Dict[str, Optional[float]]:
    """Calculate mean, std, max of a list of correlations, handling NaNs."""
    valid_corrs = [c for c in corr_list if pd.notna(c)]
    if not valid_corrs:
        return {'mean': None, 'std': None, 'max': None, 'abs_max': None, 'peak_lag': None}

    abs_max_val = max(abs(c) for c in valid_corrs)
    peak_lag_val = None
    if abs_max_val > 1e-9:
         try:
             peak_lag_val = next(i + 1 for i, c in enumerate(corr_list) if pd.notna(c) and abs(abs(c) - abs_max_val) < 1e-9)
         except StopIteration:
             peak_lag_val = None

    return {
        'mean': np.mean(valid_corrs),
        'std': np.std(valid_corrs),
        'max': np.max(valid_corrs),
        'abs_max': abs_max_val,
        'peak_lag': peak_lag_val
    }

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
         if corr_values_full is None or len(corr_values_full) < max_lag:
             logger.warning(f"Skipping config ID {config_id} in combined chart prep due to invalid/insufficient data.")
             continue
         corr_values = corr_values_full[:max_lag]
         stats = _get_stats(corr_values)
         if stats['abs_max'] is not None and stats['abs_max'] > 1e-9:
             config_info = configs_dict.get(config_id)
             if config_info:
                 name = config_info['indicator_name']
                 identifier = f"{name}_{config_id}"
                 plot_data.append({
                     'config_id': config_id,
                     'identifier': identifier,
                     'correlations': corr_values,
                     'mean': stats['mean'],
                     'peak_lag': stats['peak_lag'] if stats['peak_lag'] is not None else max_lag + 1
                 })
         else:
             logger.debug(f"Skipping config ID {config_id} from combined plot as all correlations are zero or NaN.")

    if not plot_data:
         logger.warning("No valid indicator data to plot for combined chart.")
         return

    # Optional sorting can remain if helpful, even when plotting all
    # plot_data.sort(key=lambda x: (abs(x['mean']), x['peak_lag']), reverse=False)
    # plot_data.sort(key=lambda x: abs(x['mean']), reverse=True)
    # logger.info("Sorted all plottable configurations by absolute mean correlation (desc).")

    fig, ax = plt.subplots(figsize=(15, 10), dpi=config.PLOT_DPI)

    plotted_count = 0
    for item in plot_data: # Plot all items
        identifier = item['identifier']
        config_id_plotting = item['config_id']
        corr_values_list = item['correlations']

        corr_series = pd.Series(corr_values_list, index=lags, dtype=float)
        corr_series_cleaned = corr_series.dropna()

        if len(corr_series_cleaned) < 2:
            logger.warning(f"Not enough valid points for {identifier} in combined chart. Skipping line.")
            continue

        plot_lags = corr_series_cleaned.index
        plot_corrs = corr_series_cleaned.values

        logger.debug(f"COMBINED PLOT: ID={config_id_plotting}, Identifier={identifier}, Lags={list(plot_lags[:5])}, Corrs={list(plot_corrs[:5])}")

        ax.plot(plot_lags, plot_corrs, marker='.', markersize=1, linestyle='-', linewidth=0.8, label=identifier)
        plotted_count += 1

    if plotted_count == 0:
        logger.warning("No lines were plotted for the combined chart.")
        plt.close(fig)
        return

    ax.set_title(f"Combined Correlation vs. Lag (All {plotted_count} Configurations)", fontsize=12)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Pearson Correlation with Future Close Price")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, max_lag)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    _set_axis_intersection_at_zero(ax)

    if plotted_count <= 30:
         ax.legend(loc='best', fontsize='xx-small', ncol=2 if plotted_count > 15 else 1)
    else:
         logger.warning(f"Hiding legend for combined chart as there are too many lines ({plotted_count}).")
         # ax.legend().set_visible(False)

    fig.tight_layout()

    filepath = _prepare_filenames(output_dir, file_prefix, "Combined_All_" + str(plotted_count), "chart")
    try:
        fig.savefig(filepath)
        logger.info(f"Saved combined chart: {filepath.name}")
    except Exception as e:
        logger.error(f"Failed to save combined chart {filepath.name}: {e}", exc_info=True)
    finally:
        plt.close(fig)

def generate_enhanced_heatmap(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str,
    is_tweak_path: bool
) -> None:
    """Generates an enhanced heatmap of correlation values."""
    # (Heatmap logic remains the same)
    logger.info("Generating enhanced correlation heatmap...")
    if max_lag <= 0 or not correlations_by_config_id:
         logger.warning("No data or invalid lag for heatmap.")
         return

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed}
    heatmap_data = {}
    valid_config_ids = list(correlations_by_config_id.keys())

    for config_id in valid_config_ids:
        corr_values_full = correlations_by_config_id.get(config_id)
        if (corr_values_full is None or
            not isinstance(corr_values_full, (list, np.ndarray)) or
            len(corr_values_full) < max_lag or
            all(pd.isna(c) for c in corr_values_full[:max_lag])):
             logger.warning(f"Skipping config ID {config_id} in heatmap due to invalid/insufficient data.")
             continue
        corr_values = corr_values_full[:max_lag]

        config_info = configs_dict.get(config_id)
        if config_info:
            name = config_info['indicator_name']
            identifier = utils.get_config_identifier(name, config_id, None)
            heatmap_data[identifier] = corr_values
        else:
             logger.warning(f"Config info not found for ID {config_id} during heatmap data preparation.")

    if not heatmap_data:
         logger.warning("No valid data to generate heatmap.")
         return

    corr_df = pd.DataFrame(heatmap_data, index=range(1, max_lag + 1))
    corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
    corr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    corr_df.dropna(axis=1, how='all', inplace=True)
    corr_df.dropna(axis=0, how='all', inplace=True)

    if corr_df.empty:
        logger.warning("Heatmap DataFrame is empty after initial NaN drop.")
        return

    num_columns_before_filter = len(corr_df.columns)
    filtered_cols_df = corr_df.copy()
    if is_tweak_path and num_columns_before_filter > config.HEATMAP_MAX_CONFIGS:
        logger.info(f"Tweak path: Filtering heatmap from {num_columns_before_filter} to top {config.HEATMAP_MAX_CONFIGS} configs by absolute mean correlation.")
        try:
            col_means = filtered_cols_df.abs().mean(skipna=True).dropna()
            if not col_means.empty:
                 col_means = col_means.sort_values(ascending=False)
                 top_cols = col_means.head(config.HEATMAP_MAX_CONFIGS).index
                 filtered_cols_df = filtered_cols_df[top_cols]
                 logger.info(f"Filtered heatmap to {len(filtered_cols_df.columns)} columns.")
            else:
                 logger.warning("Could not calculate mean correlations for filtering heatmap columns.")
        except Exception as filter_e:
             logger.error(f"Error during heatmap column filtering: {filter_e}. Proceeding with unfiltered columns.", exc_info=True)

    filtered_cols_df.dropna(axis=1, how='all', inplace=True)
    filtered_cols_df.dropna(axis=0, how='all', inplace=True)

    if filtered_cols_df.empty:
        logger.warning("Heatmap DataFrame is empty after filtering/dropping NaNs.")
        return

    sorted_df = filtered_cols_df.copy()
    sort_description = "Original Order"
    try:
         sort_metric_values = sorted_df.abs().mean(skipna=True).dropna()
         if not sort_metric_values.empty:
              sort_metric = sort_metric_values.sort_values(ascending=False).index
              sorted_df = sorted_df[sort_metric]
              sort_description = "Sorted by |Mean Correlation|"
         else:
              logger.warning("Could not calculate mean correlations for sorting heatmap columns.")
    except Exception as sort_e:
         logger.warning(f"Could not sort heatmap columns: {sort_e}. Using default order.", exc_info=True)

    if sorted_df.empty:
        logger.warning("Cannot plot heatmap, DataFrame is empty after sorting attempts.")
        return

    num_rows_plot = len(sorted_df.columns)
    num_cols_plot = len(sorted_df.index)
    fig_width = max(15, num_cols_plot * 0.25)
    fig_height = max(10, num_rows_plot * 0.2)
    fig_width = min(fig_width, 60)
    fig_height = min(fig_height, 80)

    plt.figure(figsize=(fig_width, fig_height), dpi=config.PLOT_DPI)

    sns.heatmap(
        sorted_df.T,
        annot=False,
        cmap='coolwarm',
        center=0,
        linewidths=0.1,
        linecolor='lightgrey',
        cbar=True,
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={'shrink': 0.5}
    )

    plt.title(f"Indicator Correlation with Future Close Price vs. Lag\n({sort_description})", fontsize=14)
    plt.xlabel("Lag", fontsize=12)
    plt.ylabel("Indicator Configuration", fontsize=12)

    x_tick_labels = sorted_df.index
    num_x_labels_desired = 50
    x_tick_step = max(1, len(x_tick_labels) // num_x_labels_desired)
    new_x_ticks = np.arange(0, len(x_tick_labels), x_tick_step) + 0.5
    new_x_labels = x_tick_labels[::x_tick_step]

    y_tick_labels = sorted_df.columns
    num_y_labels_desired = 60
    y_tick_step = max(1, len(y_tick_labels) // num_y_labels_desired)
    new_y_ticks = np.arange(0, len(y_tick_labels), y_tick_step) + 0.5
    new_y_labels = y_tick_labels[::y_tick_step]

    plt.xticks(ticks=new_x_ticks, labels=new_x_labels, rotation=90, fontsize=max(5, min(8, 500 / len(new_x_labels))))
    plt.yticks(ticks=new_y_ticks, labels=new_y_labels, rotation=0, fontsize=max(5, min(8, 500 / len(new_y_labels))))

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    filepath = _prepare_filenames(output_dir, file_prefix, "Heatmap_" + sort_description.replace('|','').replace(' ',''), "heatmap")
    try:
        plt.savefig(filepath)
        logger.info(f"Saved heatmap: {filepath.name}")
    except Exception as e:
        logger.error(f"Failed to save heatmap {filepath.name}: {e}", exc_info=True)
    finally:
        plt.close()


# --- MODIFIED Envelope Chart (Area Chart, No CI, Conditional Fill/Line) ---
def generate_correlation_envelope_chart(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int,
    output_dir: Path,
    file_prefix: str
) -> None:
    """
    Generates an area chart showing the max positive (green area from 0 up)
    and min negative (red area from 0 down) correlation across all indicators
    for each lag. Axes intersect at y=0.
    """
    logger.info("Generating correlation envelope area chart...")
    if max_lag <= 0 or not correlations_by_config_id:
        logger.warning("No data or invalid lag for envelope chart.")
        return

    # --- 1. Data Transformation ---
    heatmap_data = {}
    valid_config_ids = list(correlations_by_config_id.keys())
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed}

    for config_id in valid_config_ids:
        corr_values_full = correlations_by_config_id.get(config_id)
        if (corr_values_full is None or
            not isinstance(corr_values_full, (list, np.ndarray)) or
            len(corr_values_full) < max_lag or
            all(pd.isna(c) for c in corr_values_full[:max_lag])):
            logger.warning(f"Skipping config ID {config_id} in envelope chart due to invalid/insufficient data.")
            continue
        corr_values = corr_values_full[:max_lag]

        config_info = configs_dict.get(config_id)
        if config_info:
            name = config_info['indicator_name']
            identifier = utils.get_config_identifier(name, config_id, None)
            heatmap_data[identifier] = corr_values
        else:
             logger.warning(f"Config info not found for ID {config_id} during envelope data preparation.")

    if not heatmap_data:
         logger.warning("No valid data to generate envelope chart.")
         return

    corr_df = pd.DataFrame(heatmap_data, index=range(1, max_lag + 1))
    corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
    corr_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    corr_df.dropna(axis=1, how='all', inplace=True)

    if corr_df.empty or corr_df.shape[1] == 0:
        logger.warning("Envelope chart DataFrame is empty after dropping all-NaN columns.")
        return

    # --- 2. Calculate Envelope Lines ---
    max_positive_corr = corr_df.max(axis=1)
    min_negative_corr = corr_df.min(axis=1)

    if max_positive_corr.isna().all() and min_negative_corr.isna().all():
        logger.warning("Max positive and min negative correlations are all NaN. Cannot plot envelope chart.")
        return

    # --- 3. Plotting as Area Chart ---
    lags = corr_df.index
    fig, ax = plt.subplots(figsize=(15, 8), dpi=config.PLOT_DPI)

    # Fill positive area: Use .where() to fill only when max_positive_corr >= 0
    ax.fill_between(lags, 0, max_positive_corr.where(max_positive_corr >= 0, 0),
                    facecolor='green', alpha=0.5, interpolate=True, label='Max Positive Correlation Range')

    # Fill negative area: Use .where() to fill only when min_negative_corr <= 0
    ax.fill_between(lags, 0, min_negative_corr.where(min_negative_corr <= 0, 0),
                    facecolor='red', alpha=0.5, interpolate=True, label='Min Negative Correlation Range')

    # --- REMOVED explicit boundary lines ---
    # ax.plot(lags, max_positive_corr, color='darkgreen', linewidth=0.75, alpha=0.8)
    # ax.plot(lags, min_negative_corr, color='darkred', linewidth=0.75, alpha=0.8)
    # --- END REMOVAL ---

    # --- Formatting ---
    ax.set_title('Correlation Envelope Area Chart vs. Lag', fontsize=14)
    ax.set_xlabel('Lag', fontsize=12)
    ax.set_ylabel('Pearson Correlation with Future Close Price', fontsize=12)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(0, max_lag)

    _set_axis_intersection_at_zero(ax) # Apply axis change
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right')
    fig.tight_layout()

    # --- Saving ---
    filepath = _prepare_filenames(output_dir, file_prefix, "Correlation_Envelope_Area", "chart")
    try:
        fig.savefig(filepath)
        logger.info(f"Saved correlation envelope area chart: {filepath.name}")
    except Exception as e:
        logger.error(f"Failed to save correlation envelope area chart {filepath.name}: {e}", exc_info=True)
    finally:
        plt.close(fig)