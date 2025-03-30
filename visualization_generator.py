# visualization_generator.py
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
    safe_identifier = re.sub(r'[\\/*?:"<>|\s]+', '_', config_identifier)
    max_total_len = 200
    base_len = len(output_dir.name) + len(file_prefix) + len(chart_type) + 5
    max_safe_id_len = max(10, max_total_len - base_len)
    if len(safe_identifier) > max_safe_id_len:
        safe_identifier = safe_identifier[:max_safe_id_len] + "_etc"
    filename = f"{file_prefix}_{safe_identifier}_{chart_type}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / filename

def _set_axis_intersection_at_zero(ax):
    """Moves the x-axis spine to y=0."""
    try:
        ax.spines['left'].set_position('zero'); ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom'); ax.yaxis.set_ticks_position('left')
    except Exception as e: logger.warning(f"Could not set axis intersection at zero: {e}")

# --- Plotting Functions ---
def plot_correlation_lines(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str
) -> None:
    """Generates separate line charts with CI bands for each indicator config."""
    logger.info(f"Generating {len(correlations_by_config_id)} separate correlation line charts with CI bands...")
    if max_lag <= 0: logger.error("Max lag <= 0, cannot plot lines."); return

    lags = list(range(1, max_lag + 1))
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    rolling_window_ci = max(3, min(15, max_lag // 4)); ci_multiplier = 1.96 # 95% CI
    plotted_count = 0
    plot_dpi = config.DEFAULTS.get("plot_dpi", 150) # Get DPI from defaults, fallback 150

    for config_id, corr_values_full in correlations_by_config_id.items():
        config_info = configs_dict.get(config_id)
        if not config_info: logger.warning(f"Config info missing for ID {config_id}. Skipping plot."); continue
        name = config_info.get('indicator_name', 'Unk'); params = config_info.get('params',{})
        try:
             params_short = "-".join(f"{k}{v}" for k,v in sorted(params.items()))
             params_title = json.dumps(params, separators=(',', ':'))
        except: params_short = str(params); params_title = params_short
        safe_params = re.sub(r'[\\/*?:"<>|\s]+', '_', params_short)
        config_id_base = f"{name}_{config_id}_{safe_params}"

        if corr_values_full is None or not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag:
            logger.warning(f"Invalid/short corr data for {config_id_base}. Skipping plot."); continue

        corr_values = corr_values_full[:max_lag]; corr_series = pd.Series(corr_values, index=lags, dtype=float)
        if corr_series.isnull().all(): logger.info(f"Corr data all NaN for {config_id_base}. Skipping plot."); continue
        corr_series_clean = corr_series.dropna()
        if len(corr_series_clean) < 2: logger.info(f"Not enough valid points for {config_id_base}. Skipping plot."); continue

        plot_lags = corr_series_clean.index; plot_corrs = corr_series_clean.values
        logger.debug(f"INDIV PLOT: ID={config_id}, ID_Base={config_id_base}, Lags={list(plot_lags[:5])}, Corrs={list(plot_corrs[:5])}")

        # Calculate CI
        rolling_std = corr_series.rolling(window=rolling_window_ci, min_periods=2, center=True).std()
        half_width = ci_multiplier * rolling_std; upper_band = (corr_series + half_width).clip(upper=1.0); lower_band = (corr_series - half_width).clip(lower=-1.0)

        # ---> Corrected DPI Access <---
        fig, ax = plt.subplots(figsize=(12, 6), dpi=plot_dpi)
        # ---> End Correction <---

        ax.plot(plot_lags, plot_corrs, marker='.', linestyle='-', label=f'Corr (ID {config_id})', zorder=3)
        valid_band_lags = rolling_std.dropna().index
        if not valid_band_lags.empty:
            ax.fill_between(valid_band_lags, lower_band.loc[valid_band_lags], upper_band.loc[valid_band_lags],
                            color='skyblue', alpha=0.3, interpolate=True, label=f'~95% CI (Win={rolling_window_ci})', zorder=2)
        else: logger.debug(f"No valid CI band for {config_id_base}.")

        ax.set_title(f"Correlation vs. Lag: {name}\nParams: {params_title} (ID: {config_id})", fontsize=10)
        ax.set_xlabel("Lag (Periods)"); ax.set_ylabel("Pearson Correlation")
        ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, max_lag + 1)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1)
        _set_axis_intersection_at_zero(ax)
        ax.legend(fontsize='small'); fig.tight_layout()
        filepath = _prepare_filenames(output_dir, file_prefix, config_id_base, "line_CI")
        try: fig.savefig(filepath); plotted_count += 1; logger.debug(f"Saved line chart: {filepath.name}")
        except Exception as e: logger.error(f"Failed save line chart {filepath.name}: {e}", exc_info=True)
        finally: plt.close(fig)

    logger.info(f"Generated {plotted_count} separate line charts with CI bands.")

def _get_stats(corr_list: List[Optional[float]]) -> Dict[str, Optional[Any]]:
    """Calculate stats (mean_abs, std_abs, max_abs, peak_lag) from correlation list."""
    corr_array = np.array(corr_list, dtype=float)
    if np.isnan(corr_array).all(): return {'mean_abs': np.nan, 'std_abs': np.nan, 'max_abs': np.nan, 'peak_lag': None}
    abs_corrs = np.abs(corr_array); max_abs_val = np.nanmax(abs_corrs); peak_lag_val = None
    if pd.notna(max_abs_val):
         try:
             peak_lag_index = np.nanargmax(abs_corrs); peak_lag_val = peak_lag_index + 1
             if not np.isclose(abs(corr_array[peak_lag_index]), max_abs_val): # Verify peak index
                  logger.warning(f"Peak lag mismatch: idx {peak_lag_index} val {corr_array[peak_lag_index]}, max abs {max_abs_val}. Re-checking.")
                  indices = np.where(np.isclose(abs_corrs, max_abs_val))[0]
                  peak_lag_val = indices[0] + 1 if len(indices) > 0 else None
         except (ValueError, IndexError) as e: logger.warning(f"Cannot determine peak lag: {e}"); peak_lag_val = None
    return {'mean_abs': np.nanmean(abs_corrs), 'std_abs': np.nanstd(abs_corrs), 'max_abs': max_abs_val, 'peak_lag': peak_lag_val}

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
    plot_dpi = config.DEFAULTS.get("plot_dpi", 150) # Get DPI from defaults
    heatmap_max_configs = config.DEFAULTS.get("heatmap_max_configs", 50) # Use same limit as heatmap

    for cfg_id, corrs_full in correlations_by_config_id.items():
         if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag:
             logger.warning(f"Skipping ID {cfg_id} in combined chart prep: invalid/short data."); continue
         corrs = corrs_full[:max_lag]; stats = _get_stats(corrs)
         if stats['max_abs'] is not None and pd.notna(stats['max_abs']) and stats['max_abs'] > 1e-6:
             info = configs_dict.get(cfg_id)
             if info:
                 name = info.get('indicator_name', 'Unk'); params = info.get('params', {})
                 params_short = "-".join(f"{k}{v}" for k,v in sorted(params.items())); safe_params = re.sub(r'[\\/*?:"<>|\s]+', '_', params_short)
                 identifier = f"{name}_{cfg_id}_{safe_params}"
                 plot_data.append({'config_id': cfg_id, 'identifier': identifier, 'correlations': corrs, 'max_abs': stats['max_abs'], 'peak_lag': stats['peak_lag'] if stats['peak_lag'] else max_lag + 1})
         else: logger.debug(f"Skipping ID {cfg_id} from combined plot: max abs corr near zero/NaN.")

    if not plot_data: logger.warning("No valid data for combined chart after filtering."); return

    plot_data.sort(key=lambda x: x.get('max_abs', 0), reverse=True)
    logger.info(f"Sorted {len(plot_data)} configs by max abs corr for combined plot.")

    # ---> Corrected DPI Access <---
    fig, ax = plt.subplots(figsize=(15, 10), dpi=plot_dpi)
    # ---> End Correction <---

    plotted_count = 0
    max_lines = heatmap_max_configs # Use limit from config
    plot_subset = plot_data
    title_suffix = f" ({len(plot_data)} Configs)"
    if len(plot_data) > max_lines:
        plot_subset = plot_data[:max_lines]; title_suffix = f" (Top {max_lines} of {len(plot_data)} by Max Abs Corr)"
        logger.warning(f"Plotting only top {max_lines} configs on combined chart.")

    for item in plot_subset:
        identifier = item['identifier']; corr_series = pd.Series(item['correlations'], index=lags, dtype=float); corr_clean = corr_series.dropna()
        if len(corr_clean) < 2: logger.warning(f"Not enough points for {identifier} in combined. Skipping."); continue
        plot_lags = corr_clean.index; plot_corrs = corr_clean.values
        logger.debug(f"COMBINED PLOT (Limit {max_lines}): ID={item['config_id']}, ID={identifier}, Lags={list(plot_lags[:5])}, Corrs={list(plot_corrs[:5])}")
        ax.plot(plot_lags, plot_corrs, marker='.', markersize=1, linestyle='-', linewidth=0.7, alpha=0.6, label=identifier)
        plotted_count += 1

    if plotted_count == 0: logger.warning("No lines plotted for combined chart."); plt.close(fig); return

    ax.set_title(f"Combined Correlation vs. Lag" + title_suffix, fontsize=12)
    ax.set_xlabel("Lag (Periods)"); ax.set_ylabel("Pearson Correlation")
    ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, max_lag + 1)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1); _set_axis_intersection_at_zero(ax)
    if plotted_count <= 30: ax.legend(loc='best', fontsize='xx-small', ncol=2 if plotted_count > 15 else 1)
    else: logger.warning(f"Hiding legend for combined chart ({plotted_count} lines).")
    fig.tight_layout()
    filename_suffix = f"Combined_{plotted_count}Configs" if len(plot_data) > max_lines else f"Combined_All_{plotted_count}"
    filepath = _prepare_filenames(output_dir, file_prefix, filename_suffix, "chart")
    try: fig.savefig(filepath); logger.info(f"Saved combined chart: {filepath.name}")
    except Exception as e: logger.error(f"Failed save combined chart {filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig)

def generate_enhanced_heatmap(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str, is_tweak_path: bool
) -> None:
    """Generates heatmap of correlation values, filtered and sorted."""
    logger.info("Generating enhanced correlation heatmap...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data/invalid lag for heatmap."); return

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    heatmap_data = {}
    plot_dpi = config.DEFAULTS.get("plot_dpi", 150) # Get DPI from defaults
    heatmap_max_configs = config.DEFAULTS.get("heatmap_max_configs", 50) # Get limit

    for cfg_id, corrs_full in correlations_by_config_id.items():
        if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag: logger.warning(f"Skipping ID {cfg_id} in heatmap prep: insufficient data."); continue
        corrs = corrs_full[:max_lag]
        if all(pd.isna(c) for c in corrs): logger.info(f"Skipping ID {cfg_id} in heatmap prep: all NaNs."); continue
        info = configs_dict.get(cfg_id)
        if info:
            name = info.get('indicator_name', 'Unk'); params = info.get('params', {})
            params_short = "-".join(f"{k}{v}" for k,v in sorted(params.items())); safe_params = re.sub(r'[\\/*?:"<>|\s]+', '_', params_short)
            identifier = f"{name}_{cfg_id}_{safe_params}"
            heatmap_data[identifier] = corrs
        else: logger.warning(f"Config info missing for ID {cfg_id} in heatmap prep.")

    if not heatmap_data: logger.warning("No valid data columns for heatmap."); return
    corr_df = pd.DataFrame(heatmap_data, index=range(1, max_lag + 1)).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    corr_df.dropna(axis=1, how='all', inplace=True); corr_df.dropna(axis=0, how='all', inplace=True)
    if corr_df.empty: logger.warning("Heatmap DF empty after initial NaN drop."); return

    # --- Filtering ---
    filtered_df = corr_df.copy(); filter_applied = False
    if len(corr_df.columns) > heatmap_max_configs:
        logger.info(f"Filtering heatmap from {len(corr_df.columns)} to top {heatmap_max_configs} by max abs corr.")
        try:
            scores = filtered_df.abs().max(skipna=True).dropna()
            if not scores.empty: top_cols = scores.sort_values(ascending=False).head(heatmap_max_configs).index; filtered_df = filtered_df[top_cols]; filter_applied = True; logger.info(f"Filtered heatmap to {len(filtered_df.columns)} columns.")
            else: logger.warning("Cannot calc scores for heatmap filter.")
        except Exception as filter_e: logger.error(f"Error filtering heatmap: {filter_e}. Proceeding.", exc_info=True)
        filtered_df.dropna(axis=1, how='all', inplace=True); filtered_df.dropna(axis=0, how='all', inplace=True)
    if filtered_df.empty: logger.warning("Heatmap DF empty after filter/drop."); return

    # --- Sorting ---
    sorted_df = filtered_df.copy(); sort_desc = "Filtered Order" if filter_applied else "Original Order"
    try:
        metric = sorted_df.abs().mean(skipna=True).dropna()
        if not metric.empty: sorted_cols = metric.sort_values(ascending=False).index; sorted_df = sorted_df[sorted_cols]; sort_desc = "Sorted by Mean Abs Corr"; logger.info("Sorted heatmap cols by mean abs corr.")
        else: logger.warning("Cannot calc means for heatmap sort.")
    except Exception as sort_e: logger.warning(f"Cannot sort heatmap: {sort_e}.", exc_info=True)
    if sorted_df.empty: logger.warning("Cannot plot heatmap: DF empty after sort."); return

    # --- Plotting ---
    num_configs = len(sorted_df.columns); num_lags = len(sorted_df.index)
    fig_w = max(15, min(60, num_lags * 0.2 + 5)); fig_h = max(10, min(80, num_configs * 0.3 + 2))

    # ---> Corrected DPI Access <---
    plt.figure(figsize=(fig_w, fig_h), dpi=plot_dpi)
    # ---> End Correction <---

    sns.heatmap(sorted_df.T, annot=False, cmap='coolwarm', center=0, linewidths=0.1, linecolor='lightgrey', cbar=True, vmin=-1.0, vmax=1.0, cbar_kws={'shrink': 0.6})
    plt.title(f"Correlation vs. Lag ({num_configs} Configs, {sort_desc})", fontsize=14)
    plt.xlabel("Lag (Periods)", fontsize=12); plt.ylabel("Indicator Configuration", fontsize=12)

    # Dynamic Ticks
    x_labels = sorted_df.index; num_x = min(50, len(x_labels)); x_step = max(1, len(x_labels) // num_x)
    x_pos = np.arange(len(x_labels))[::x_step]; x_labs = x_labels[::x_step]
    y_labels = sorted_df.columns; num_y = min(60, len(y_labels)); y_step = max(1, len(y_labels) // num_y)
    y_pos = np.arange(len(y_labels))[::y_step]; y_labs = y_labels[::y_step]
    xtick_fs = max(5, min(8, 700 / len(x_labs))) if len(x_labs) > 0 else 8
    ytick_fs = max(5, min(8, 700 / len(y_labs))) if len(y_labs) > 0 else 8
    plt.xticks(ticks=x_pos + 0.5, labels=x_labs, rotation=90, fontsize=xtick_fs)
    plt.yticks(ticks=y_pos + 0.5, labels=y_labs, rotation=0, fontsize=ytick_fs)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
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
    plot_dpi = config.DEFAULTS.get("plot_dpi", 150) # Get DPI from defaults

    for cfg_id, corrs_full in correlations_by_config_id.items():
        if corrs_full and isinstance(corrs_full, list) and len(corrs_full) >= max_lag:
            corrs = corrs_full[:max_lag]
            if not all(pd.isna(c) for c in corrs): valid_data[cfg_id] = corrs
        else: logger.debug(f"Excluding Cfg {cfg_id} from envelope: insufficient data.")
    if not valid_data: logger.warning("No valid configs for envelope chart."); return

    corr_df = pd.DataFrame(valid_data, index=range(1, max_lag + 1)).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if corr_df.empty or corr_df.shape[1] == 0: logger.warning("Envelope chart DF empty."); return
    max_pos = corr_df.max(axis=1, skipna=True); min_neg = corr_df.min(axis=1, skipna=True)
    if max_pos.isna().all() and min_neg.isna().all(): logger.warning("Envelopes all NaN."); return

    # Default Config Handling for Tweak Path
    default_corr = None; default_id = None; tweak_ind_name = None; default_params = None
    if is_tweak_path and indicator_configs_processed:
        try:
            tweak_ind_name = indicator_configs_processed[0]['indicator_name'] # Assume first is representative
            indicator_factory._load_indicator_definitions()
            ind_def = indicator_factory._get_indicator_definition(tweak_ind_name)
            if ind_def:
                default_params = {k: v.get('default') for k, v in ind_def.get('parameters', {}).items() if 'default' in v}
                if default_params:
                    for cfg in indicator_configs_processed: # Search full list for match
                        if cfg.get('indicator_name') == tweak_ind_name and utils.compare_param_dicts(cfg.get('params',{}), default_params):
                            found_id = cfg.get('config_id'); logger.debug(f"Found default match ID={found_id}")
                            if found_id and found_id in correlations_by_config_id:
                                default_corr_list = correlations_by_config_id[found_id][:max_lag]
                                if not all(pd.isna(c) for c in default_corr_list): default_id = found_id; default_corr = pd.Series(default_corr_list, index=range(1, max_lag+1)); logger.info(f"Found corr data for default ID {default_id}.")
                                else: logger.warning(f"Default ID {found_id} has NaN corrs.")
                            else: logger.warning(f"Default ID {found_id} not in corr data.")
                            break # Stop searching once found
                else: logger.warning(f"Cannot find definition for '{tweak_ind_name}'.")
        except Exception as e: logger.error(f"Error getting default config data: {e}", exc_info=True)
        default_id = None if default_corr is None else default_id # Ensure ID is None if series not found

    lags = corr_df.index
    # ---> Corrected DPI Access <---
    fig, ax = plt.subplots(figsize=(15, 8), dpi=plot_dpi)
    # ---> End Correction <---

    # Plot envelopes (positive part for max, negative part for min)
    ax.fill_between(lags, 0, max_pos.where(max_pos >= 0, 0).fillna(0), facecolor='mediumseagreen', alpha=0.4, interpolate=True, label='Max Pos Range', zorder=2)
    ax.fill_between(lags, 0, min_neg.where(min_neg <= 0, 0).fillna(0), facecolor='lightcoral', alpha=0.4, interpolate=True, label='Min Neg Range', zorder=2)
    ax.plot(lags, max_pos.where(max_pos >= 0), color='darkgreen', lw=0.8, alpha=0.7, label='Max Pos Env', zorder=3)
    ax.plot(lags, min_neg.where(min_neg <= 0), color='darkred', lw=0.8, alpha=0.7, label='Min Neg Env', zorder=3)
    # Plot default line
    if default_corr is not None and default_id is not None:
        default_clean = default_corr.dropna()
        if len(default_clean) >= 2: ax.plot(default_clean.index, default_clean.values, color='darkblue', ls='--', lw=1.0, alpha=0.8, label=f"Default '{tweak_ind_name}' (ID: {default_id})", zorder=4)
        else: logger.warning(f"Not enough valid points for default line {tweak_ind_name}.")

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
            logger.warning(f"Short corr data for {name} (ID {cfg_id}). Skipping report entry."); continue
        corrs = corrs_full[:max_lag]; corr_arr = np.array(corrs, dtype=float)
        if np.isnan(corr_arr).all():
            logger.info(f"Skipping {name} (ID {cfg_id}): all NaNs."); entry = {'Config ID': cfg_id, 'Indicator': name, 'Parameters': params_str, 'Peak Positive Corr': np.nan, 'Peak Positive Lag': np.nan, 'Peak Negative Corr': np.nan, 'Peak Negative Lag': np.nan, 'Peak Absolute Corr': np.nan, 'Peak Absolute Lag': np.nan}
        else:
            peak_pos = np.nanmax(corr_arr) if not np.isnan(corr_arr).all() else np.nan; pos_idx = np.nanargmax(corr_arr) if pd.notna(peak_pos) else -1; pos_lag = pos_idx + 1 if pos_idx != -1 else np.nan
            peak_neg = np.nanmin(corr_arr) if not np.isnan(corr_arr).all() else np.nan; neg_idx = np.nanargmin(corr_arr) if pd.notna(peak_neg) else -1; neg_lag = neg_idx + 1 if neg_idx != -1 else np.nan
            abs_arr = np.abs(corr_arr); peak_abs = np.nanmax(abs_arr) if not np.isnan(abs_arr).all() else np.nan; abs_idx = np.nanargmax(abs_arr) if pd.notna(peak_abs) else -1; abs_lag = abs_idx + 1 if abs_idx != -1 else np.nan
            entry = {'Config ID': cfg_id, 'Indicator': name, 'Parameters': params_str, 'Peak Positive Corr': peak_pos, 'Peak Positive Lag': pos_lag, 'Peak Negative Corr': peak_neg, 'Peak Negative Lag': neg_lag, 'Peak Absolute Corr': peak_abs, 'Peak Absolute Lag': abs_lag}
        report_data.append(entry)

    if not report_data: logger.warning("No data for peak report."); return
    report_df = pd.DataFrame(report_data).sort_values('Peak Absolute Corr', ascending=False, na_position='last')

    # Print to console
    print("\n\n--- Peak Correlation Summary ---")
    df_print = report_df.copy()
    for col in ['Peak Positive Corr', 'Peak Negative Corr', 'Peak Absolute Corr']: df_print[col] = df_print[col].map('{:.4f}'.format).replace('nan', 'N/A')
    for col in ['Peak Positive Lag', 'Peak Negative Lag', 'Peak Absolute Lag']: df_print[col] = df_print[col].map('{:.0f}'.format).replace('nan', 'N/A')
    max_p_len = 50; df_print['Parameters'] = df_print['Parameters'].apply(lambda x: x if not isinstance(x, str) or len(x) <= max_p_len else x[:max_p_len-3] + '...')
    try:
        with pd.option_context('display.max_rows', 100, 'display.width', 1000): print(df_print.to_string(index=False))
    except Exception as print_e: logger.error(f"Console print error: {print_e}"); print(df_print.head())

    # Save to CSV
    csv_filepath = output_dir / f"{file_prefix}_peak_correlation_report.csv"
    try: report_df.to_csv(csv_filepath, index=False, float_format='%.6f'); logger.info(f"Saved peak report: {csv_filepath}"); print(f"\nPeak report saved: {csv_filepath}")
    except Exception as e: logger.error(f"Failed save peak report CSV {csv_filepath}: {e}", exc_info=True); print("Error saving report CSV.")
