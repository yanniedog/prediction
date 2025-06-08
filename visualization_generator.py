# visualization_generator.py
# Updated: Removed is_tweak_path/is_bayesian_path parameter from heatmap/envelope functions.
# Fixed: Indentation error in generate_enhanced_heatmap filter try block.
# Fixed: Indentation error in generate_enhanced_heatmap sort try block.
# Fixed: Indentation error in generate_peak_correlation_report try block (this error).

import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend BEFORE importing pyplot
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import re
import math

import config # Import the config module
import utils
import indicator_factory # For loading defs if needed (e.g., for default line)

logger = logging.getLogger(__name__)

# --- Helper Functions ---
def _prepare_filenames(output_dir: Path, file_prefix: str, config_identifier: str, chart_type: str) -> Path:
    """Creates a sanitized filename for the plot."""
    safe_identifier = re.sub(r'[\\/*?:"<>|\s]+', '_', str(config_identifier))
    max_total_len = 200; base_len = len(str(output_dir)) + len(file_prefix) + len(chart_type) + 5
    max_safe_id_len = max(10, max_total_len - base_len)
    if len(safe_identifier) > max_safe_id_len: safe_identifier = safe_identifier[:max_safe_id_len] + "_etc"
    filename = f"{file_prefix}_{safe_identifier}_{chart_type}.png"
    return output_dir / filename

def _set_axis_intersection_at_zero(ax):
    """Moves the x-axis spine to y=0."""
    try:
        ax.spines['left'].set_position('zero'); ax.spines['bottom'].set_position('zero')
        ax.spines['right'].set_color('none'); ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom'); ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_tick_params(pad=5); ax.yaxis.set_tick_params(pad=5)
    except Exception as e: logger.warning(f"Could not set axis intersect at zero: {e}")

# --- Plotting Functions ---
def plot_correlation_lines(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str
) -> Optional[Path]:
    """Generates separate line charts with CI bands for each indicator config."""
    num_configs = len(correlations_by_config_id)
    logger.info(f"Generating {num_configs} separate correlation line charts...")
    
    # Validate output directory strictly (do not allow auto-creation)
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Path exists but is not a directory: {output_dir}")
    # Test write access
    test_file = output_dir / ".write_test"
    try:
        test_file.touch()
        test_file.unlink()
    except OSError as e:
        raise PermissionError(f"No write access to directory: {output_dir}") from e
    
    if max_lag <= 0:
        logger.error("Max lag <= 0, cannot plot lines.")
        return None
    if num_configs == 0:
        logger.warning("No correlation data for line plots.")
        return None

    lags = list(range(1, max_lag + 1))
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    rolling_window_ci = max(3, min(15, max_lag // 5 + 1))
    ci_multiplier = 1.96  # 95% CI
    plotted_count = 0
    plot_dpi = config.DEFAULTS.get("plot_dpi", 150)
    output_files = []

    for config_id, corr_values_full in correlations_by_config_id.items():
        config_info = configs_dict.get(config_id)
        if not config_info:
            logger.warning(f"Config info missing ID {config_id}. Skip plot.")
            continue

        name = config_info.get('indicator_name', 'Unknown')
        params = config_info.get('params', {})
        
        try:
            params_short = "-".join(f"{k}{v}" for k, v in sorted(params.items()))
            params_title = json.dumps(params, separators=(',', ':'))
        except TypeError:
            params_short = str(params)
            params_title = params_short

        safe_params = re.sub(r'[\\/*?:"<>|\s]+', '_', params_short)[:50]
        config_id_base = f"{name}_{config_id}_{safe_params}"

        if not isinstance(corr_values_full, list) or len(corr_values_full) < max_lag:
            continue

        corr_values = corr_values_full[:max_lag]
        corr_series = pd.Series(corr_values, index=lags).astype(float)
        
        if corr_series.isnull().all():
            continue
            
        corr_series_clean = corr_series.dropna()
        if len(corr_series_clean) < 2:
            continue

        try:
            # Create the plot
            plt.figure(figsize=(10, 6), dpi=plot_dpi)
            
            # Plot main line
            plt.plot(corr_series_clean.index, corr_series_clean.values, 'b-', label='Correlation')
            
            # Calculate and plot confidence intervals
            rolling_mean = corr_series_clean.rolling(window=rolling_window_ci, min_periods=1).mean()
            rolling_std = corr_series_clean.rolling(window=rolling_window_ci, min_periods=1).std()
            ci_upper = rolling_mean + (ci_multiplier * rolling_std)
            ci_lower = rolling_mean - (ci_multiplier * rolling_std)
            
            plt.fill_between(corr_series_clean.index, ci_lower, ci_upper, alpha=0.2, color='gray', label='95% CI')
            
            # Customize plot
            plt.title(f"{name} Correlation with Price (Config {config_id})\nParams: {params_title}")
            plt.xlabel('Lag')
            plt.ylabel('Correlation')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Save plot
            output_file = output_dir / f"{file_prefix}_{config_id_base}_correlation.png"
            plt.savefig(output_file, bbox_inches='tight', dpi=plot_dpi)
            plt.close()
            
            output_files.append(output_file)
            plotted_count += 1
            
        except Exception as e:
            logger.error(f"Error plotting correlation lines for config {config_id}: {e}")
            continue

    if plotted_count > 0:
        logger.info(f"Successfully plotted {plotted_count} correlation line charts")
        return output_files[0] if output_files else None
    else:
        logger.warning("No correlation line charts were generated")
        return None


def _get_stats(corr_list: List[Optional[float]], lag_offset: int = 1) -> Dict[str, Optional[Any]]:
    """Calculate stats (mean_abs, std_abs, max_abs, peak_lag) from correlation list."""
    if not isinstance(corr_list, list): return {'mean_abs': np.nan, 'std_abs': np.nan, 'max_abs': np.nan, 'peak_lag': None}
    corr_array = np.array(corr_list, dtype=float)
    if np.isnan(corr_array).all(): return {'mean_abs': np.nan, 'std_abs': np.nan, 'max_abs': np.nan, 'peak_lag': None}
    abs_corrs = np.abs(corr_array)
    # Use axis=0 explicitly to avoid FutureWarning
    max_abs_val = np.nanmax(abs_corrs, axis=0) if abs_corrs.size > 0 else np.nan
    peak_lag_val = None
    if pd.notna(max_abs_val):
         try: 
             indices = np.where(np.isclose(abs_corrs, max_abs_val))[0]
             peak_lag_val = indices[0] + lag_offset if len(indices) > 0 else None
         except (ValueError, IndexError) as e: 
             logger.warning(f"Cannot determine peak lag: {e}")
             peak_lag_val = None
    return {
        'mean_abs': np.nanmean(abs_corrs, axis=0) if abs_corrs.size > 0 else np.nan,
        'std_corr': np.nanstd(corr_array, axis=0) if corr_array.size > 0 else np.nan,
        'max_abs': max_abs_val,
        'peak_lag': peak_lag_val
    }


def generate_combined_correlation_chart(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str
) -> Optional[Path]:
    """Generates a combined line chart for multiple indicator configs and returns the output file path."""
    logger.info("Generating combined correlation chart...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data/invalid lag for combined chart."); return None

    lags = list(range(1, max_lag + 1))
    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    plot_data = []; plot_dpi = config.DEFAULTS.get("plot_dpi", 150)
    max_lines = config.DEFAULTS.get("linechart_max_configs", 50) # Use specific limit

    for cfg_id, corrs_full in correlations_by_config_id.items():
         if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag: continue
         corrs = corrs_full[:max_lag]; stats = _get_stats(corrs)
         if stats['max_abs'] is not None and pd.notna(stats['max_abs']) and stats['max_abs'] > 1e-6:
             info = configs_dict.get(cfg_id)
             if info:
                 name = info.get('indicator_name', 'Unk'); params = info.get('params', {})
                 params_short = "-".join(f"{k}{v}" for k,v in sorted(params.items())); safe_params = re.sub(r'[\\/*?:"<>|\s]+', '_', params_short)[:50]
                 identifier = f"{name}_{cfg_id}_{safe_params}"
                 plot_data.append({'config_id': cfg_id, 'identifier': identifier, 'correlations': corrs, 'max_abs': stats['max_abs']})
         else: pass # logger.debug(f"Skip ID {cfg_id} combined: max abs low/NaN.") # Reduce noise
    if not plot_data: logger.warning("No valid data for combined chart after filtering."); return None
    plot_data.sort(key=lambda x: x.get('max_abs', 0), reverse=True); logger.info(f"Prepared {len(plot_data)} configs for combined plot.")
    fig, ax = plt.subplots(figsize=(15, 10), dpi=plot_dpi); plotted_count = 0
    plot_subset = plot_data; title_suffix = f" ({len(plot_data)} Configs)"
    if len(plot_data) > max_lines: plot_subset = plot_data[:max_lines]; title_suffix = f" (Top {max_lines} of {len(plot_data)} by Max Abs Corr)"; logger.info(f"Plotting top {max_lines} on combined chart.")
    for item in plot_subset:
        identifier = item['identifier']; corr_series = pd.Series(item['correlations'], index=lags).astype(float); corr_clean = corr_series.dropna()
        if len(corr_clean) < 2: continue
        plot_lags = corr_clean.index; plot_corrs = corr_clean.values
        ax.plot(plot_lags, plot_corrs, marker='.', markersize=1, linestyle='-', linewidth=0.7, alpha=0.6, label=identifier); plotted_count += 1
    if plotted_count == 0: logger.warning("No lines plotted combined chart."); plt.close(fig); return None
    ax.set_title(f"Combined Correlation vs. Lag" + title_suffix, fontsize=12); ax.set_xlabel("Lag (Periods)"); ax.set_ylabel("Pearson Correlation"); ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, max_lag + 1); ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1); _set_axis_intersection_at_zero(ax)
    if plotted_count <= 30: ax.legend(loc='best', fontsize='xx-small', ncol=2 if plotted_count > 15 else 1)
    else: logger.info(f"Hiding legend combined chart ({plotted_count} lines > 30).")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    filename_suffix = f"Combined_Top{plotted_count}" if len(plot_data) > max_lines else f"Combined_All_{plotted_count}"; filepath = _prepare_filenames(output_dir, file_prefix, filename_suffix, "chart")
    try: fig.savefig(filepath); logger.info(f"Saved combined chart: {filepath.name}")
    except Exception as e: logger.error(f"Failed save combined chart {filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig)
    return filepath


def generate_enhanced_heatmap(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str,
) -> Optional[Path]:
    """Generates heatmap of correlation values, filtered and sorted, and returns the output file path."""
    logger.info("Generating enhanced correlation heatmap...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data/invalid lag for heatmap."); return None

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    heatmap_data = {}; plot_dpi = config.DEFAULTS.get("plot_dpi", 150); heatmap_max_configs = config.DEFAULTS.get("heatmap_max_configs", 50)

    for cfg_id, corrs_full in correlations_by_config_id.items():
        if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag: continue
        corrs = corrs_full[:max_lag];
        if all(pd.isna(c) for c in corrs): continue
        info = configs_dict.get(cfg_id)
        if info:
            name = info.get('indicator_name', 'Unk'); params = info.get('params', {})
            params_short = "-".join(f"{k}{v}" for k,v in sorted(params.items())); safe_params = re.sub(r'[\\/*?:"<>|\s]+', '_', params_short)[:50]
            identifier = f"{name}_{cfg_id}_{safe_params}"
            heatmap_data[identifier] = corrs
        else: logger.warning(f"Config info missing ID {cfg_id} heatmap.")
    if not heatmap_data: logger.warning("No valid data columns for heatmap."); return None

    corr_df = pd.DataFrame(heatmap_data, index=range(1, max_lag + 1)).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan); corr_df.dropna(axis=1, how='all', inplace=True); corr_df.dropna(axis=0, how='all', inplace=True)
    if corr_df.empty: logger.warning("Heatmap DF empty after NaN drop."); return None

    filtered_df = corr_df.copy(); filter_applied = False; num_cols_before = len(corr_df.columns)
    if num_cols_before > heatmap_max_configs:
        logger.info(f"Filtering heatmap from {num_cols_before} to top {heatmap_max_configs} by max abs corr.")
        try:
             scores = filtered_df.abs().max(skipna=True).dropna()
             if not scores.empty:
                 top_cols = scores.nlargest(heatmap_max_configs).index
                 filtered_df = filtered_df[top_cols]
                 filter_applied = True
                 logger.info(f"Filtered heatmap to {len(filtered_df.columns)} columns.")
             else:
                 logger.warning("Cannot calculate scores for heatmap filtering (scores were empty).")
        except Exception as filter_e:
            logger.error(f"Error filtering heatmap: {filter_e}. Proceeding with unfiltered data.", exc_info=True)
            filtered_df = corr_df.copy()
            filter_applied = False

        filtered_df.dropna(axis=1, how='all', inplace=True)
        filtered_df.dropna(axis=0, how='all', inplace=True)

    if filtered_df.empty: logger.warning("Heatmap DF empty after filter/drop stage."); return None

    # Proceed with sorting using the potentially filtered dataframe
    sorted_df = filtered_df.copy(); sort_desc = "Filtered Order" if filter_applied else "Original Order"
    try:
         # --- Start of CORRECTED/INDENTED Block for Sorting ---
         metric = sorted_df.abs().mean(skipna=True).dropna()
         # V V V --- Corrected Line 208 area --- V V V
         if not metric.empty:
             sorted_cols = metric.sort_values(ascending=False).index
             sorted_df = sorted_df[sorted_cols]
             sort_desc = "Sorted by Mean Abs Corr"
             logger.info("Sorted heatmap columns by mean absolute correlation.")
         else:
             logger.warning("Cannot calculate means for heatmap sorting (metric empty). Using current order.")
         # --- End of CORRECTED/INDENTED Block for Sorting ---
    except Exception as sort_e:
        logger.warning(f"Cannot sort heatmap due to error: {sort_e}. Using current order.", exc_info=True)
        # sorted_df remains as filtered_df, sort_desc remains as previously set

    # Proceed with plotting using sorted_df
    num_configs = len(sorted_df.columns); num_lags = len(sorted_df.index); fig_w = max(15, min(60, num_lags * 0.2 + 5)); fig_h = max(10, min(80, num_configs * 0.3 + 2)); logger.debug(f"Heatmap Fig Size: ({fig_w:.1f}, {fig_h:.1f}) for {num_configs} configs, {num_lags} lags.")
    plt.figure(figsize=(fig_w, fig_h), dpi=plot_dpi)
    sns.heatmap(sorted_df.T, annot=False, cmap='coolwarm', center=0, linewidths=0.1, linecolor='lightgrey', cbar=True, vmin=-1.0, vmax=1.0, cbar_kws={'shrink': 0.6})
    plt.title(f"Correlation vs. Lag ({num_configs} Configs, {sort_desc})", fontsize=14); plt.xlabel("Lag (Periods)", fontsize=12); plt.ylabel("Indicator Configuration", fontsize=12)
    x_labels = sorted_df.index; num_x_target = max(1, int(fig_w * 3)); x_step = max(1, math.ceil(len(x_labels) / num_x_target)) if len(x_labels) > 0 else 1; x_pos = np.arange(len(x_labels))[::x_step]; x_labs = x_labels[::x_step]
    y_labels = sorted_df.columns; num_y_target = max(1, int(fig_h * 2)); y_step = max(1, math.ceil(len(y_labels) / num_y_target)) if len(y_labels) > 0 else 1; y_pos = np.arange(len(y_labels))[::y_step]; y_labs = y_labels[::y_step]
    xtick_fs = max(5, min(10, int(100 / math.sqrt(max(1, len(x_labs)))))) if len(x_labs) > 0 else 8; ytick_fs = max(5, min(10, int(100 / math.sqrt(max(1, len(y_labs)))))) if len(y_labs) > 0 else 8
    plt.xticks(ticks=x_pos + 0.5, labels=list(x_labs), rotation=90, fontsize=xtick_fs)
    plt.yticks(ticks=y_pos + 0.5, labels=list(y_labs), rotation=0, fontsize=ytick_fs)
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))
    filename_suffix = f"Heatmap_{sort_desc.replace('|','').replace(' ','_')}"
    filepath = _prepare_filenames(output_dir, file_prefix, filename_suffix, "heatmap")
    try: plt.savefig(filepath); logger.info(f"Saved heatmap: {filepath.name}")
    except Exception as e: logger.error(f"Failed save heatmap {filepath.name}: {e}", exc_info=True)
    finally: plt.close()
    return filepath


def generate_correlation_envelope_chart(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str,
) -> Optional[Path]:
    """Generates area chart showing max positive and min negative correlation envelopes and returns the output file path."""
    logger.info("Generating correlation envelope area chart...")
    if max_lag <= 0 or not correlations_by_config_id: logger.warning("No data/invalid lag for envelope."); return None

    valid_data = {}; plot_dpi = config.DEFAULTS.get("plot_dpi", 150)
    for cfg_id, corrs_full in correlations_by_config_id.items():
        if corrs_full and isinstance(corrs_full, list) and len(corrs_full) >= max_lag:
            corrs = corrs_full[:max_lag]
            if not all(pd.isna(c) for c in corrs): valid_data[cfg_id] = corrs
    if not valid_data: logger.warning("No valid configs for envelope chart."); return None
    corr_df = pd.DataFrame(valid_data, index=range(1, max_lag + 1)).apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    if corr_df.empty or corr_df.shape[1] == 0: logger.warning("Envelope chart DF empty."); return None
    max_pos = corr_df.max(axis=1, skipna=True); min_neg = corr_df.min(axis=1, skipna=True)
    if max_pos.isna().all() and min_neg.isna().all(): logger.warning("Envelopes all NaN."); return None

    lags = corr_df.index
    fig, ax = plt.subplots(figsize=(15, 8), dpi=plot_dpi)
    ax.fill_between(lags, 0, max_pos.where(max_pos >= 0).fillna(0), facecolor='mediumseagreen', alpha=0.4, interpolate=True, label='Max Pos Range', zorder=2)
    ax.fill_between(lags, 0, min_neg.where(min_neg <= 0).fillna(0), facecolor='lightcoral', alpha=0.4, interpolate=True, label='Min Neg Range', zorder=2)
    ax.plot(lags, max_pos.where(max_pos >= 0), color='darkgreen', lw=0.8, alpha=0.7, label='Max Pos Env', zorder=3)
    ax.plot(lags, min_neg.where(min_neg <= 0), color='darkred', lw=0.8, alpha=0.7, label='Min Neg Env', zorder=3)

    title = f'Correlation Envelope vs. Lag ({corr_df.shape[1]} Configs)'; ax.set_title(title, fontsize=14); ax.set_xlabel('Lag (Periods)', fontsize=12); ax.set_ylabel('Correlation', fontsize=12); ax.set_ylim(-1.05, 1.05); ax.set_xlim(0, max_lag + 1); _set_axis_intersection_at_zero(ax); ax.grid(True, which='both', linestyle='--', linewidth=0.5, zorder=1); ax.legend(loc='lower right', fontsize='small'); fig.tight_layout()
    filepath = _prepare_filenames(output_dir, file_prefix, "Correlation_Envelope_Area", "chart")
    try: fig.savefig(filepath); logger.info(f"Saved envelope chart: {filepath.name}")
    except Exception as e: logger.error(f"Failed save envelope chart {filepath.name}: {e}", exc_info=True)
    finally: plt.close(fig)
    return filepath


def generate_peak_correlation_report(
    correlations_by_config_id: Dict[int, List[Optional[float]]],
    indicator_configs_processed: List[Dict[str, Any]],
    max_lag: int, output_dir: Path, file_prefix: str
) -> Optional[Path]:
    """Generates summary report (CSV) of peak correlations and returns the output file path."""
    logger.info("Generating peak correlation summary report...")
    if not correlations_by_config_id or max_lag <= 0: logger.warning("No corr data/invalid lag for peak report."); return None

    configs_dict = {cfg['config_id']: cfg for cfg in indicator_configs_processed if 'config_id' in cfg}
    report_data = []
    for cfg_id, corrs_full in correlations_by_config_id.items():
        info = configs_dict.get(cfg_id)
        if not info: logger.warning(f"Config info missing ID {cfg_id} peak report."); continue
        name = info.get('indicator_name', 'Unk'); params = info.get('params', {}); params_str = json.dumps(params, separators=(',',':'))
        if corrs_full is None or not isinstance(corrs_full, list) or len(corrs_full) < max_lag: continue
        corrs = corrs_full[:max_lag]; stats = _get_stats(corrs, lag_offset=1)
        corr_arr = np.array(corrs, dtype=float)
        peak_pos_val = np.nanmax(corr_arr) if not np.isnan(corr_arr).all() else np.nan
        try: pos_lag_val = (np.nanargmax(corr_arr) + 1) if pd.notna(peak_pos_val) and not np.isnan(corr_arr).all() else np.nan
        except ValueError: pos_lag_val = np.nan
        peak_neg_val = np.nanmin(corr_arr) if not np.isnan(corr_arr).all() else np.nan
        try: neg_lag_val = (np.nanargmin(corr_arr) + 1) if pd.notna(peak_neg_val) and not np.isnan(corr_arr).all() else np.nan
        except ValueError: neg_lag_val = np.nan
        entry = {'Config ID': cfg_id, 'Indicator': name, 'Parameters': params_str, 'Peak Positive Corr': peak_pos_val, 'Peak Positive Lag': pos_lag_val, 'Peak Negative Corr': peak_neg_val, 'Peak Negative Lag': neg_lag_val, 'Peak Absolute Corr': stats.get('max_abs', np.nan), 'Peak Absolute Lag': stats.get('peak_lag', np.nan)}
        report_data.append(entry)
    if not report_data: logger.warning("No data for peak report."); return None
    report_df = pd.DataFrame(report_data).sort_values('Peak Absolute Corr', ascending=False, na_position='last')
    csv_filepath = output_dir / f"{file_prefix}_peak_correlation_report.csv"
    try:
        output_dir.mkdir(parents=True, exist_ok=True);
        report_df.to_csv(csv_filepath, index=False, float_format='%.6f', na_rep='NaN')
        # --- Corrected Indentation START ---
        logger.info(f"Saved peak report: {csv_filepath}")
        # --- Corrected Indentation END ---
    except Exception as e:
        logger.error(f"Failed save peak report CSV {csv_filepath}: {e}", exc_info=True)
    return csv_filepath

def plot_indicator_performance(data, indicator_name, output_path):
    if indicator_name not in data.columns:
        raise ValueError(f"Indicator {indicator_name} not found in data")
    plt.figure(figsize=(10, 4))
    plt.plot(data["timestamp"] if "timestamp" in data.columns else data.index, data[indicator_name], label=indicator_name)
    plt.title(f"Performance of {indicator_name}")
    plt.xlabel("Time")
    plt.ylabel(indicator_name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def generate_charts(
    output_dir: Path,
    price_data: pd.DataFrame = None,
    indicator_data: pd.DataFrame = None,
    correlation_data: pd.DataFrame = None,
    optimization_data: dict = None,
    actual_predictions: pd.Series = None,
    predicted_predictions: pd.Series = None
) -> dict:
    """Generate all main chart types and return a dictionary of chart paths."""
    chart_paths = {}
    if price_data is not None and indicator_data is not None:
        chart_path = output_dir / "indicator_performance.png"
        plot_indicator_performance(price_data, indicator_data, chart_path)
        chart_paths["indicator_performance"] = chart_path
    if correlation_data is not None:
        chart_path = output_dir / "correlation_matrix.png"
        plot_correlation_matrix(correlation_data, chart_path)
        chart_paths["correlation_matrix"] = chart_path
    if optimization_data is not None:
        chart_path = output_dir / "optimization_results.png"
        plot_optimization_results(optimization_data, chart_path)
        chart_paths["optimization_results"] = chart_path
    if actual_predictions is not None and predicted_predictions is not None:
        chart_path = output_dir / "prediction_accuracy.png"
        plot_prediction_accuracy(actual_predictions, predicted_predictions, chart_path)
        chart_paths["prediction_accuracy"] = chart_path
    if not chart_paths:
        raise ValueError("No data provided for chart generation.")
    return chart_paths

def plot_correlation_matrix(
    correlation_data: pd.DataFrame,
    output_dir: Path,
    file_prefix: str = "correlation_matrix"
) -> Optional[Path]:
    """Generate a correlation matrix heatmap from the correlation data.
    
    Args:
        correlation_data: DataFrame containing correlation values
        output_dir: Directory to save the output file
        file_prefix: Prefix for the output filename
        
    Returns:
        Optional[Path]: Path to the generated file if successful, None otherwise
    """
    try:
        if correlation_data.empty:
            logger.warning("Empty correlation data provided for matrix plot")
            return None
            
        # Create figure and axis
        plt.figure(figsize=(12, 10), dpi=config.DEFAULTS.get("plot_dpi", 150))
        
        # Generate correlation matrix
        corr_matrix = correlation_data.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            square=True,
            linewidths=.5,
            cbar_kws={'shrink': .8}
        )
        
        # Customize plot
        plt.title('Correlation Matrix', pad=20)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout and save
        plt.tight_layout()
        output_file = _prepare_filenames(output_dir, file_prefix, "matrix", "heatmap")
        plt.savefig(output_file, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Generated correlation matrix plot: {output_file}")
        return output_file
        
    except Exception as e:
        logger.error(f"Error generating correlation matrix: {e}", exc_info=True)
        plt.close()
        return None

def plot_optimization_results(optimization_data: dict, output_path: Path) -> None:
    """Plot optimization results (supports 1D and 2D parameter sweeps)."""
    import matplotlib.pyplot as plt
    import numpy as np
    if not optimization_data or not isinstance(optimization_data, dict):
        raise ValueError("No optimization data provided.")
    # Only support 1D or 2D for simplicity
    for indicator, results in optimization_data.items():
        if not isinstance(results, dict) or 'params' not in results or 'score' not in results:
            # Try to handle grid format: {param_name: [...], 'score': [...]}
            param_keys = [k for k in results.keys() if k != 'score']
            if len(param_keys) == 1:
                # 1D
                x = results[param_keys[0]]
                y = results['score']
                plt.figure(figsize=(8, 5))
                plt.plot(x, y, marker='o')
                plt.xlabel(param_keys[0])
                plt.ylabel('Score')
                plt.title(f'Optimization Results: {indicator}')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                return
            elif len(param_keys) == 2:
                # 2D
                x = results[param_keys[0]]
                y = results[param_keys[1]]
                z = results['score']
                X, Y = np.meshgrid(np.unique(x), np.unique(y))
                Z = np.full_like(X, np.nan, dtype=float)
                for xi, yi, zi in zip(x, y, z):
                    x_idx = np.where(X[0] == xi)[0][0]
                    y_idx = np.where(Y[:,0] == yi)[0][0]
                    Z[y_idx, x_idx] = zi
                plt.figure(figsize=(8, 6))
                cp = plt.contourf(X, Y, Z, cmap='viridis')
                plt.colorbar(cp)
                plt.xlabel(param_keys[0])
                plt.ylabel(param_keys[1])
                plt.title(f'Optimization Results: {indicator}')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                return
            else:
                raise ValueError("Unsupported optimization data format.")
        else:
            # Single best result
            params = results['params']
            score = results['score']
            plt.figure(figsize=(6, 4))
            plt.bar(list(params.keys()), list(params.values()))
            plt.title(f'Best Params (Score: {score:.4f})')
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
            return
    raise ValueError("No valid optimization results to plot.")

def plot_prediction_accuracy(actual: pd.Series, predicted: pd.Series, output_path: Path) -> None:
    """Plot actual vs predicted values and save to output_path."""
    import matplotlib.pyplot as plt
    if actual is None or predicted is None or len(actual) != len(predicted):
        raise ValueError("Actual and predicted series must be the same length and not None.")
    plt.figure(figsize=(8, 5))
    plt.plot(actual.index, actual.values, label='Actual', marker='o')
    plt.plot(predicted.index, predicted.values, label='Predicted', marker='x')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Prediction Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def _format_chart_data(data):
    """Format chart data for plotting. Accepts DataFrame or dict."""
    if isinstance(data, pd.DataFrame):
        return data.values, data.columns.tolist(), data.index.tolist()
    elif isinstance(data, dict):
        # Assume dict of lists or dict of dicts
        keys = list(data.keys())
        if all(isinstance(v, list) for v in data.values()):
            # Dict of lists
            max_len = max(len(v) for v in data.values())
            values = [v + [None]*(max_len-len(v)) for v in data.values()]
            return values, keys, list(range(max_len))
        elif all(isinstance(v, dict) for v in data.values()):
            # Dict of dicts
            inner_keys = sorted({k for d in data.values() for k in d.keys()})
            values = [[d.get(k, None) for k in inner_keys] for d in data.values()]
            return values, keys, inner_keys
        else:
            raise ValueError("Unsupported dict format for chart data.")
    else:
        raise ValueError("Unsupported data type for chart formatting.")

def _save_chart(fig, output_path: Path) -> None:
    """Save a matplotlib figure to the specified output path."""
    fig.tight_layout()
    fig.savefig(output_path)
    fig.clf()
    import matplotlib.pyplot as plt
    plt.close(fig)