# visualization_generator.py
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates # Keep for potential future date axes
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import re

import config # Import the config module
import utils
import indicator_factory # For loading defs if needed (e.g., for default line)

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _prepare_filenames(output_dir: Path, file_prefix: str, config_identifier: str, chart_type: str) -> Path:
    """Creates a sanitized filename for the plot."""
    # Sanitize identifier: remove problematic chars, limit length
    safe_identifier = re.sub(r'[\\/*?:"<>|\s]+', '_', str(config_identifier))
    max_total_len = 200 # Max path length often around 260, leave buffer
    base_len = len(str(output_dir)) + len(file_prefix) + len(chart_type) + 5 # Path, prefix, type, separators, extension
    max_safe_id_len = max(10, max_total_len - base_len) # Ensure at least 10 chars for ID part
    if len(safe_identifier) > max_safe_id_len:
        # Simple truncation, could use hashing for very long IDs if needed
        safe_identifier = safe_identifier[:max_safe_id_len] + "_etc"

    filename = f"{file_prefix}_{safe_identifier}_{chart_type}.png"
    # Ensure directory exists *before* returning path
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create directory {output_dir}: {e}")
        # Fallback: try saving in main reports dir? Or raise error?
        # For now, just log error and proceed, saving might fail.
    return output_dir / filename

def _set_axis_intersection_at_zero(ax):
    """Moves the x-axis spine to y=0."""
    try:
        ax.spines['left'].set_position('zero')
        ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # Add small padding to prevent ticks overlapping axis line
        ax.xaxis.set_tick_params(pad=5)
        ax.yaxis.set_tick_params(pad=5)
    except Exception as e: logger.warning(f"Could not set axis intersection at zero: {e}")

# --- Plotting Functions ---
def plot_correlation_lines(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str
) -> None:
    """Generates separate line charts with CI bands for each indicator config."""
    num_configs = len(correlations_by_config_id)
    logger.info(f"Generating {num_configs} separate correlation line charts...")
    if max_lag <= 0: logger.error("Max lag <= 0, cannot plot lines."); return
    if num_configs == 0: logger.warning("No correlation data provided for line plots."); return

    lags = list(range(1, max_lag + 1))
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    # Adaptive rolling window for CI based on max_lag
    rolling_window_ci = max(3, min(15, max_lag // 5 + 1))
    ci_multiplier = 1.96 # 95% CI
    plotted_count = 0
    plot_dpi = config.DEFAULTS.get("plot_dpi", 150)

    for config_id, corr_values_full in correlations_by_config_id.items():
        config_info = configs_dict.get(config_id)
        if not config_info: logger.warning(f"Config info missing for ID {config_id}. Skipping plot."); continue
        name = config_info.get('indicator_name', 'Unk'); params = config_info.get('params',{})
        try:
             # Create a shorter, safer identifier for filename part
             params_short = "-".join(f"{k}{v}" for k,v in sorted(params.items()))
             params_title = json.dumps(params, separators=(',', ':')) # Full params for title
        except TypeError: params_short = str(params); params_title = params_short
        safe_params = re.sub(r'[\\/*?:"<>|\s]+', '_', params_short)[:50] # Limit param str length
        config_id_base = f"{name}_{config_id}_{safe_params}" # Used for filename

        if corr_values_full is None or not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag:
            logger.debug(f"Invalid/short corr data for {config_id_base}. Skipping plot."); continue

        corr_values = corr_values_full[:max_lag]
        corr_series = pd.Series(corr_values, index=lags).astype(float) # Ensure float

        if corr_series.isnull().all(): logger.debug(f"Corr data all NaN for {config_id_base}. Skipping plot."); continue
        corr_series_clean = corr_series.dropna()
        if len(corr_series_clean) < 2: logger.debug(f"Not enough valid points (need >=2) for {config_id_base}. Skipping plot."); continue

        plot_lags = corr_series_clean.index; plot_corrs = corr_series_clean.values
        # logger.debug(f"INDIV PLOT: ID={config_id}, Lags={list(plot_lags[:5])}, Corrs={list(plot_corrs[:5])}")

        # Calculate CI based on the *full* series (including NaNs) for better windowing
        rolling_std = corr_series.rolling(window=rolling_window_ci, min_periods=2, center=True).std()
        half_width = ci_multiplier * rolling_std
        # Ensure bands don't have NaNs where original data exists
        upper_band = (corr_series + half_width).clip(upper=1.0).combine_first(corr_series)
        lower_band = (corr_series - half_width).clip(lower=-1.0).combine_first(corr_series)

        fig, ax = plt.subplots(figsize=(12, 6), dpi=plot_dpi)

        # Plot only the non-NaN points
        ax.plot(plot_lags, plot_corrs, marker='.', linestyle='-', label=f'Corr (ID {config_id})', zorder=3)

        # Plot CI band where valid std dev exists
        valid_band_lags = rolling_std.dropna().index
        if not valid_band_lags.empty:
            # Ensure we only plot where upper/lower bands are also valid
            valid_plot_indices = valid_band_lags.intersection(lower_band.dropna().index).intersection(upper_band.dropna().index)
            if not valid_plot_indices.empty:
                ax.fill_between(valid_plot_indices, lower_band.loc[valid_plot_indices], upper_band.loc[valid_plot_indices],
                                color='skyblue', alpha=0.3, interpolate=True, label=f'~95% CI (Win={rolling_window_ci})', zorder=2)
        # else: logger.debug(f"No valid CI band for {config_id_base}.")

        ax.set_title(f"Correlation vs. Lag: {name}\nParams: {params_title} (ID: {config_id})", fontsize=10)
        ax.set_xlabel("Lag (Periods)"); ax.set_ylabel("Pearson Correlation")
        ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, max_lag + 1)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        _set_axis_intersection_at_zero(ax)
        ax.legend(fontsize='small'); fig.tight_layout()
        filepath = _prepare_filenames(output_dir, file_prefix, config_id_base, "line_CI")
        try: fig.savefig(filepath); plotted_count += 1; # logger.debug(f"Saved line chart: {filepath.name}")
        except Exception as e: logger.error(f"Failed save line chart {filepath.name}: {e}", exc_info=True)
        finally: plt.close(fig) # Ensure figure is closed

    logger.info(f"Generated {plotted_count}/{num_configs} separate line charts with CI bands.")


def _get_stats(corr_list: List[Optional[float]], lag_offset: int = 1) -> Dict[str, Optional[Any]]:
    """Calculate stats (mean_abs, std_abs, max_abs, peak_lag) from correlation list."""
    # Ensure input is a list
    if not isinstance(corr_list, list):
        return {'mean_abs': np.nan, 'std_abs': np.nan, 'max_abs': np.nan, 'peak_lag': None}

    corr_array = np.array(corr_list, dtype=float)
    if np.isnan(corr_array).all():
        return {'mean_abs': np.nan, 'std_abs': np.nan, 'max_abs': np.nan, 'peak_lag': None}

    abs_corrs = np.abs(corr_array)
    max_abs_val = np.nanmax(abs_corrs)
    peak_lag_val = None

    if pd.notna(max_abs_val):
         try:
             # Find all indices matching max_abs_val within tolerance
             indices = np.where(np.isclose(abs_corrs, max_abs_val))[0]
             if len(indices) > 0:
                 peak_lag_val = indices[0] + lag_offset # Report first peak lag
         except (ValueError, IndexError) as e:
             logger.warning(f"Cannot determine peak lag: {e}")
             peak_lag_val = None

    return {
        'mean_abs': np.nanmean(abs_corrs),
        'std_abs': np.nanstd(abs_corrs), # Std dev of absolute values or original? Let's use original std dev
        'std_corr': np.nanstd(corr_array), # Std dev of original correlations
        'max_abs': max_abs_val,
        'peak_lag': peak_lag_val
    }

def generate_combined_correlation_chart(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str
) -> None:
    """Generates a combined line chart for multiple indicator configs."""
    logger.info("Generating combined correlation chart...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data/invalid lag for combined chart."); return

    lags = list(range(1, max_lag + 1))
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    plot_data = []
    plot_dpi = config.DEFAULTS.get("plot_dpi", 150)
    max_lines = config.DEFAULTS.get("heatmap_max_configs", 50) # Use same limit as heatmap

    for cfg_id, corrs_full in correlations_by_config_id.items():
         if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag:
             logger.debug(f"Skipping ID {cfg_id} in combined chart prep: invalid/short data."); continue

         corrs = corrs_full[:max_lag]
         stats = _get_stats(corrs)

         # Filter out configs with near-zero or NaN max absolute correlation
         if stats['max_abs'] is not None and pd.notna(stats['max_abs']) and stats['max_abs'] > 1e-6:
             info = configs_dict.get(cfg_id)
             if info:
                 name = info.get('indicator_name', 'Unk'); params = info.get('params', {})
                 params_short = "-".join(f"{k}{v}" for k,v in sorted(params.items())); safe_params = re.sub(r'[\\/*?:"<>|\s]+', '_', params_short)[:50]
                 identifier = f"{name}_{cfg_id}_{safe_params}" # Identifier for legend/sorting
                 plot_data.append({'config_id': cfg_id, 'identifier': identifier, 'correlations': corrs, 'max_abs': stats['max_abs'], 'peak_lag': stats['peak_lag'] if stats['peak_lag'] else max_lag + 1})
         else: logger.debug(f"Skipping ID {cfg_id} from combined plot: max abs corr near zero/NaN.")

    if not plot_data: logger.warning("No valid data for combined chart after filtering."); return

    # Sort by max absolute correlation descending
    plot_data.sort(key=lambda x: x.get('max_abs', 0), reverse=True)
    logger.info(f"Prepared {len(plot_data)} configs for combined plot.")

    fig, ax = plt.subplots(figsize=(15, 10), dpi=plot_dpi)

    plotted_count = 0
    plot_subset = plot_data
    title_suffix = f" ({len(plot_data)} Configs)"
    if len(plot_data) > max_lines:
        plot_subset = plot_data[:max_lines]
        title_suffix = f" (Top {max_lines} of {len(plot_data)} by Max Abs Corr)"
        logger.info(f"Plotting only top {max_lines} configs on combined chart.")

    for item in plot_subset:
        identifier = item['identifier']
        corr_series = pd.Series(item['correlations'], index=lags).astype(float)
        corr_clean = corr_series.dropna()
        if len(corr_clean) < 2: logger.debug(f"Not enough points for {identifier} in combined. Skipping."); continue

        plot_lags = corr_clean.index
        plot_corrs = corr_clean.values
        # logger.debug(f"COMBINED PLOT (Limit {max_lines}): ID={item['config_id']}, Identifier={identifier}, Lags={list(plot_lags[:5])}")
        ax.plot(plot_lags, plot_corrs, marker='.', markersize=1, linestyle='-', linewidth=0.7, alpha=0.6, label=identifier)
        plotted_count += 1

    if plotted_count == 0: logger.warning("No lines plotted for combined chart."); plt.close(fig); return

    ax.set_title(f"Combined Correlation vs. Lag" + title_suffix, fontsize=12)
    ax.set_xlabel("Lag (Periods)"); ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, max_lag + 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1); _set_axis_intersection_at_zero(ax)

    # Adjust legend visibility based on number of lines
    if plotted_count <= 30:
        ax.legend(loc='best', fontsize='xx-small', ncol=2 if plotted_count > 15 else 1)
    else:
        logger.info(f"Hiding legend for combined chart ({plotted_count} lines > 30).")

    fig.tight_layout()
    # Use a consistent filename structure
    filename_suffix = f"Combined_Top{plotted_count}" if len(plot_data) > max_lines else f"Combined_All_{plotted_count}"
    filepath = _prepare_filenames(output_dir, file_prefix, filename_suffix, "chart")
    try: fig.savefig(filepath); logger.info(f"Saved combined chart: {filepath.name}")
    except Exception as e: logger.error(f"Failed save combined chart {filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig)


def generate_enhanced_heatmap(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str, is_tweak_path: bool # is_tweak_path currently unused
) -> None:
    """Generates heatmap of correlation values, filtered and sorted."""
    logger.info("Generating enhanced correlation heatmap...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data/invalid lag for heatmap."); return

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    heatmap_data = {}
    plot_dpi = config.DEFAULTS.get("plot_dpi", 150)
    heatmap_max_configs = config.DEFAULTS.get("heatmap_max_configs", 50)

    for cfg_id, corrs_full in correlations_by_config_id.items():
        if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag:
             logger.debug(f"Skipping ID {cfg_id} in heatmap prep: insufficient data."); continue

        corrs = corrs_full[:max_lag]
        if all(pd.isna(c) for c in corrs): logger.debug(f"Skipping ID {cfg_id} in heatmap prep: all NaNs."); continue

        info = configs_dict.get(cfg_id)
        if info:
            name = info.get('indicator_name', 'Unk'); params = info.get('params', {})
            params_short = "-".join(f"{k}{v}" for k,v in sorted(params.items())); safe_params = re.sub(r'[\\/*?:"<>|\s]+', '_', params_short)[:50]
            identifier = f"{name}_{cfg_id}_{safe_params}" # Column identifier for heatmap
            heatmap_data[identifier] = corrs
        else: logger.warning(f"Config info missing for ID {cfg_id} in heatmap prep.")

    if not heatmap_data: logger.warning("No valid data columns for heatmap."); return

    # Create DataFrame, ensure numeric, handle potential Infs
    corr_df = pd.DataFrame(heatmap_data, index=range(1, max_lag + 1)).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    # Drop columns/rows that are *entirely* NaN AFTER potential Inf replacement
    corr_df.dropna(axis=1, how='all', inplace=True); corr_df.dropna(axis=0, how='all', inplace=True)
    if corr_df.empty: logger.warning("Heatmap DF empty after initial NaN drop."); return

    # --- Filtering ---
    filtered_df = corr_df.copy(); filter_applied = False
    num_cols_before_filter = len(corr_df.columns)
    if num_cols_before_filter > heatmap_max_configs:
        logger.info(f"Filtering heatmap from {num_cols_before_filter} to top {heatmap_max_configs} by max abs corr.")
        try:
            # Calculate max absolute correlation for each column (config)
            scores = filtered_df.abs().max(skipna=True).dropna()
            if not scores.empty:
                top_cols = scores.nlargest(heatmap_max_configs).index # Use nlargest for efficiency
                filtered_df = filtered_df[top_cols]
                filter_applied = True
                logger.info(f"Filtered heatmap to {len(filtered_df.columns)} columns.")
            else: logger.warning("Cannot calculate scores for heatmap filter.")
        except Exception as filter_e: logger.error(f"Error filtering heatmap: {filter_e}. Proceeding.", exc_info=True)
        # Drop again in case filtering created all-NaN rows/cols
        filtered_df.dropna(axis=1, how='all', inplace=True); filtered_df.dropna(axis=0, how='all', inplace=True)
    if filtered_df.empty: logger.warning("Heatmap DF empty after filter/drop."); return

    # --- Sorting ---
    sorted_df = filtered_df.copy(); sort_desc = "Filtered Order" if filter_applied else "Original Order"
    try:
        # Sort columns (indicators) by mean absolute correlation
        metric = sorted_df.abs().mean(skipna=True).dropna()
        if not metric.empty:
            sorted_cols = metric.sort_values(ascending=False).index
            sorted_df = sorted_df[sorted_cols]
            sort_desc = "Sorted by Mean Abs Corr"
            logger.info("Sorted heatmap cols by mean abs corr.")
        else: logger.warning("Cannot calculate means for heatmap sort.")
    except Exception as sort_e: logger.warning(f"Cannot sort heatmap: {sort_e}.", exc_info=True)
    if sorted_df.empty: logger.warning("Cannot plot heatmap: DF empty after sort."); return

    # --- Plotting ---
    num_configs = len(sorted_df.columns); num_lags = len(sorted_df.index)
    # Dynamic figure size calculation
    fig_w = max(15, min(60, num_lags * 0.2 + 5))
    fig_h = max(10, min(80, num_configs * 0.3 + 2))
    logger.debug(f"Heatmap Fig Size: ({fig_w:.1f}, {fig_h:.1f}) for {num_configs} configs, {num_lags} lags.")

    plt.figure(figsize=(fig_w, fig_h), dpi=plot_dpi)

    # Use annot=False for potentially large heatmaps to avoid clutter/slowness
    sns.heatmap(sorted_df.T, annot=False, cmap='coolwarm', center=0,
                linewidths=0.1, linecolor='lightgrey', cbar=True,
                vmin=-1.0, vmax=1.0, cbar_kws={'shrink': 0.6})

    plt.title(f"Correlation vs. Lag ({num_configs} Configs, {sort_desc})", fontsize=14)
    plt.xlabel("Lag (Periods)", fontsize=12); plt.ylabel("Indicator Configuration", fontsize=12)

    # Dynamic Ticks based on figure size and number of labels
    x_labels = sorted_df.index
    num_x_target = max(10, int(fig_w * 3)) # Target number based on width
    x_step = max(1, math.ceil(len(x_labels) / num_x_target)) if len(x_labels) > 0 else 1
    x_pos = np.arange(len(x_labels))[::x_step]
    x_labs = x_labels[::x_step]

    y_labels = sorted_df.columns
    num_y_target = max(10, int(fig_h * 2)) # Target number based on height
    y_step = max(1, math.ceil(len(y_labels) / num_y_target)) if len(y_labels) > 0 else 1
    y_pos = np.arange(len(y_labels))[::y_step]
    y_labs = y_labels[::y_step]

    # Dynamic font size
    xtick_fs = max(5, min(10, int(100 / math.sqrt(len(x_labs))))) if len(x_labs) > 0 else 8
    ytick_fs = max(5, min(10, int(100 / math.sqrt(len(y_labs))))) if len(y_labs) > 0 else 8

    plt.xticks(ticks=x_pos + 0.5, labels=x_labs, rotation=90, fontsize=xtick_fs)
    plt.yticks(ticks=y_pos + 0.5, labels=y_labs, rotation=0, fontsize=ytick_fs)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust layout slightly
    filename_suffix = f"Heatmap_{sort_desc.replace('|','').replace(' ','_')}"
    filepath = _prepare_filenames(output_dir, file_prefix, filename_suffix, "heatmap")
    try: plt.savefig(filepath); logger.info(f"Saved heatmap: {filepath.name}")
    except Exception as e: logger.error(f"Failed save heatmap {filepath.name}: {e}", exc_info=True)
    finally: plt.close()


def generate_correlation_envelope_chart(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str, is_tweak_path: bool
) -> None:
    """Generates area chart showing max positive and min negative correlation envelopes."""
    logger.info("Generating correlation envelope area chart...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data/invalid lag for envelope."); return

    valid_data = {}
    plot_dpi = config.DEFAULTS.get("plot_dpi", 150)

    for cfg_id, corrs_full in correlations_by_config_id.items():
        if corrs_full and isinstance(corrs_full, list) and len(corrs_full) >= max_lag:
            corrs = corrs_full[:max_lag]
            if not all(pd.isna(c) for c in corrs): valid_data[cfg_id] = corrs
        # else: logger.debug(f"Excluding Cfg {cfg_id} from envelope: insufficient data.") # Reduce noise
    if not valid_data: logger.warning("No valid configs for envelope chart."); return

    corr_df = pd.DataFrame(valid_data, index=range(1, max_lag + 1)).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if corr_df.empty or corr_df.shape[1] == 0: logger.warning("Envelope chart DF empty."); return

    # Calculate envelopes, ensuring results are Series
    max_pos = corr_df.max(axis=1, skipna=True)
    min_neg = corr_df.min(axis=1, skipna=True)
    if max_pos.isna().all() and min_neg.isna().all(): logger.warning("Envelopes all NaN."); return

    # Default Config Handling for Tweak Path
    default_corr_series = None; default_id = None; tweak_ind_name = None; default_params = None
    if is_tweak_path and indicator_configs_processed:
        try:
            # Assume first config is representative if multiple optimized, or use single optimized name
            tweak_ind_name = indicator_configs_processed[0]['indicator_name']
            indicator_factory._load_indicator_definitions() # Ensure definitions are loaded
            ind_def = indicator_factory._get_indicator_definition(tweak_ind_name)
            if ind_def:
                # Find the default parameters for this indicator
                default_params = {k: v.get('default') for k, v in ind_def.get('parameters', {}).items() if 'default' in v}
                if default_params:
                    # Search ALL processed configs for one matching the default params
                    for cfg in indicator_configs_processed:
                        if cfg.get('indicator_name') == tweak_ind_name and utils.compare_param_dicts(cfg.get('params',{}), default_params):
                            found_id = cfg.get('config_id')
                            if found_id is not None and found_id in correlations_by_config_id:
                                default_corr_list = correlations_by_config_id[found_id][:max_lag]
                                if default_corr_list and not all(pd.isna(c) for c in default_corr_list):
                                    default_id = found_id
                                    default_corr_series = pd.Series(default_corr_list, index=range(1, max_lag+1)).astype(float)
                                    logger.info(f"Found corr data for default ID {default_id} ('{tweak_ind_name}').")
                                    break # Stop searching once found
                    if default_corr_series is None: logger.warning(f"Could not find valid correlation data for default config of '{tweak_ind_name}'.")
                else: logger.warning(f"Indicator '{tweak_ind_name}' has no default parameters defined.")
            else: logger.warning(f"Cannot find definition for '{tweak_ind_name}'.")
        except Exception as e: logger.error(f"Error getting default config data: {e}", exc_info=True)
        # Ensure default_id is None if series is None
        default_id = None if default_corr_series is None else default_id

    lags = corr_df.index
    fig, ax = plt.subplots(figsize=(15, 8), dpi=plot_dpi)

    # Plot envelopes (positive part for max, negative part for min)
    # Use where to handle NaNs gracefully in fill_between
    ax.fill_between(lags, 0, max_pos.where(max_pos >= 0).fillna(0), facecolor='mediumseagreen', alpha=0.4, interpolate=True, label='Max Pos Range', zorder=2)
    ax.fill_between(lags, 0, min_neg.where(min_neg <= 0).fillna(0), facecolor='lightcoral', alpha=0.4, interpolate=True, label='Min Neg Range', zorder=2)
    # Plot envelope lines only where they are positive/negative respectively
    ax.plot(lags, max_pos.where(max_pos >= 0), color='darkgreen', lw=0.8, alpha=0.7, label='Max Pos Env', zorder=3)
    ax.plot(lags, min_neg.where(min_neg <= 0), color='darkred', lw=0.8, alpha=0.7, label='Min Neg Env', zorder=3)

    # Plot default line if found
    if default_corr_series is not None and default_id is not None:
        default_clean = default_corr_series.dropna()
        if len(default_clean) >= 2:
            ax.plot(default_clean.index, default_clean.values, color='darkblue', ls='--', lw=1.0, alpha=0.8, label=f"Default '{tweak_ind_name}' (ID: {default_id})", zorder=4)
        else: logger.warning(f"Not enough valid points ({len(default_clean)}) for default line '{tweak_ind_name}'.")

    # Setup plot
    title = f'Correlation Envelope vs. Lag ({corr_df.shape[1]} Configs)'
    if is_tweak_path and tweak_ind_name: title += f" - Optimized: {tweak_ind_name}"
    ax.set_title(title, fontsize=14); ax.set_xlabel('Lag (Periods)', fontsize=12); ax.set_ylabel('Correlation', fontsize=12)
    ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, max_lag + 1); _set_axis_intersection_at_zero(ax)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
    ax.legend(loc='lower right', fontsize='small'); fig.tight_layout()
    filepath = _prepare_filenames(output_dir, file_prefix, "Correlation_Envelope_Area", "chart")
    try: fig.savefig(filepath); logger.info(f"Saved envelope chart: {filepath.name}")
    except Exception as e: logger.error(f"Failed save envelope chart {filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig)


def generate_peak_correlation_report(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str
) -> None:
    """Generates summary report (CSV and console) of peak correlations."""
    logger.info("Generating peak correlation summary report...")
    if not correlations_by_config_id or max_lag <= 0: logger.warning("No corr data/invalid lag for report."); return

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    report_data = []
    for cfg_id, corrs_full in correlations_by_config_id.items():
        info = configs_dict.get(cfg_id)
        if not info: logger.warning(f"Config info missing for ID {cfg_id} in report."); continue
        name = info.get('indicator_name', 'Unk'); params = info.get('params', {}); params_str = json.dumps(params, separators=(',',':'))

        if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag:
            logger.debug(f"Short corr data for {name} (ID {cfg_id}). Skipping report entry."); continue

        corrs = corrs_full[:max_lag]
        # Use helper function to get stats
        stats = _get_stats(corrs, lag_offset=1) # Pass lag_offset=1 since list index 0 is lag 1

        # Extract peak positive and negative separately
        corr_arr = np.array(corrs, dtype=float)
        peak_pos_val = np.nanmax(corr_arr) if not np.isnan(corr_arr).all() else np.nan
        pos_lag_val = (np.nanargmax(corr_arr) + 1) if pd.notna(peak_pos_val) else np.nan

        peak_neg_val = np.nanmin(corr_arr) if not np.isnan(corr_arr).all() else np.nan
        neg_lag_val = (np.nanargmin(corr_arr) + 1) if pd.notna(peak_neg_val) else np.nan

        entry = {
            'Config ID': cfg_id,
            'Indicator': name,
            'Parameters': params_str,
            'Peak Positive Corr': peak_pos_val if pd.notna(peak_pos_val) else np.nan,
            'Peak Positive Lag': pos_lag_val if pd.notna(pos_lag_val) else np.nan,
            'Peak Negative Corr': peak_neg_val if pd.notna(peak_neg_val) else np.nan,
            'Peak Negative Lag': neg_lag_val if pd.notna(neg_lag_val) else np.nan,
            'Peak Absolute Corr': stats.get('max_abs', np.nan),
            'Peak Absolute Lag': stats.get('peak_lag', np.nan)
        }
        report_data.append(entry)

    if not report_data: logger.warning("No data for peak report."); return
    report_df = pd.DataFrame(report_data).sort_values('Peak Absolute Corr', ascending=False, na_position='last')

    # --- Print to console ---
    # print("\n\n--- Peak Correlation Summary ---")
    df_print = report_df.copy()
    # Format numeric columns, handle NaNs
    for col in ['Peak Positive Corr', 'Peak Negative Corr', 'Peak Absolute Corr']:
        df_print[col] = df_print[col].apply(lambda x: f"{float(x):.4f}" if pd.notna(x) else 'N/A')
    for col in ['Peak Positive Lag', 'Peak Negative Lag', 'Peak Absolute Lag']:
        df_print[col] = df_print[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else 'N/A')
    # Truncate long parameter strings
    max_p_len = 50
    df_print['Parameters'] = df_print['Parameters'].apply(lambda x: x if not isinstance(x, str) or len(x) <= max_p_len else x[:max_p_len-3] + '...')
    # Use context manager for printing large dataframes
    # try:
    #     with pd.option_context('display.max_rows', 100, 'display.width', 1000, 'display.max_colwidth', 60):
    #         print(df_print.to_string(index=False, na_rep='N/A'))
    # except Exception as print_e:
    #     logger.error(f"Console print error: {print_e}")
    #     # Fallback print
    #     print(df_print.head().to_string(index=False, na_rep='N/A'))
    # -----------------------

    # --- Save to CSV ---
    csv_filepath = output_dir / f"{file_prefix}_peak_correlation_report.csv"
    try:
        output_dir.mkdir(parents=True, exist_ok=True) # Ensure dir exists
        report_df.to_csv(csv_filepath, index=False, float_format='%.6f', na_rep='NaN') # Save with NaN representation
        logger.info(f"Saved peak report: {csv_filepath}")
        # print(f"\nPeak report saved: {csv_filepath}") # Avoid print during interim updates
    except Exception as e:
        logger.error(f"Failed save peak report CSV {csv_filepath}: {e}", exc_info=True)
        # print("Error saving report CSV.") # Avoid print
    # -----------------------