import sys, itertools, pandas as pd, numpy as np, sqlite3, talib as ta, pandas_ta as pta
from correlation_database import CorrelationDatabase
from indicators import compute_all_indicators
from config import DB_PATH
import matplotlib.pyplot as plt, seaborn as sns

# Define default values for each indicator's tweakable parameters
default_parameters = {
    'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9},
    'EMA': {'timeperiod': 30},
    'SMA': {'timeperiod': 30},
    'RSI': {'timeperiod': 14},
    'ATR': {'timeperiod': 14},
    'ADX': {'timeperiod': 14},
    'CCI': {'timeperiod': 14},
    'MOM': {'timeperiod': 10},
    'ROC': {'timeperiod': 10},
    'TRIX': {'timeperiod': 30},
    'ULTOSC': {'timeperiod1':7, 'timeperiod2':14, 'timeperiod3':28},
    'EyeX MFV Volume': {'range1':50, 'range2':75, 'range3':100, 'range4':200},
    'EyeX MFV S/R': {'range1':50, 'range2':75, 'range3':100, 'range4':200, 'pivot_lookback':5},
    'OBV Price Divergence': {'obv_period':14},
    # Add other indicators and their tweakable parameters as needed
}

def list_indicators():
    db = CorrelationDatabase(DB_PATH)
    indicators = db.conn.execute("SELECT name FROM indicators").fetchall()
    # Exclude indicators without tweakable parameters
    indicator_names = sorted([ind[0] for ind in indicators if ind[0] in default_parameters])
    db.close()
    return indicator_names

def get_configurations(params, defaults):
    config_ranges = {}
    for param in params:
        default = defaults.get(param, 1)
        lower = max(1, default - 5)
        upper = min(100, default + 5)
        config_ranges[param] = range(lower, upper + 1)
    return itertools.product(*config_ranges.values())

def compute_indicator(data, indicator, config, params):
    if indicator == 'MACD':
        return ta.MACD(data['Close'], fastperiod=config['fastperiod'], slowperiod=config['slowperiod'], signalperiod=config['signalperiod'])[0]
    elif indicator == 'EMA':
        return ta.EMA(data['Close'], timeperiod=config['timeperiod'])
    elif indicator == 'SMA':
        return ta.SMA(data['Close'], timeperiod=config['timeperiod'])
    elif indicator == 'RSI':
        return ta.RSI(data['Close'], timeperiod=config['timeperiod'])
    elif indicator == 'ATR':
        return ta.ATR(data['high'], data['low'], data['Close'], timeperiod=config['timeperiod'])
    elif indicator == 'ADX':
        return ta.ADX(data['high'], data['low'], data['Close'], timeperiod=config['timeperiod'])
    elif indicator == 'CCI':
        return ta.CCI(data['high'], data['low'], data['Close'], timeperiod=config['timeperiod'])
    elif indicator == 'MOM':
        return ta.MOM(data['Close'], timeperiod=config['timeperiod'])
    elif indicator == 'ROC':
        return ta.ROC(data['Close'], timeperiod=config['timeperiod'])
    elif indicator == 'TRIX':
        return ta.TRIX(data['Close'], timeperiod=config['timeperiod'])
    elif indicator == 'ULTOSC':
        return ta.ULTOSC(data['high'], data['low'], data['Close'], timeperiod1=config['timeperiod1'], timeperiod2=config['timeperiod2'], timeperiod3=config['timeperiod3'])
    elif indicator == 'EyeX MFV Volume':
        mf_multiplier = ((data['Close'] - data['low']) - (data['high'] - data['Close'])) / (data['high'] - data['low'])
        mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
        mf_volume = mf_multiplier * data['volume']
        combined_mfv = sum([
            (mf_volume.rolling(window=config[f'range{i}'], min_periods=1).sum() - mf_volume.shift(config[f'range{i}']).fillna(0))
            .rolling(window=config[f'range{i}'], min_periods=1)
            .apply(lambda x: (x[-1] - np.mean(x)) / np.std(x) * 10 if np.std(x) > 0 else 0, raw=True)
            for i in range(1,5)
        ]).clip(-400, 400)
        return combined_mfv
    elif indicator == 'EyeX MFV S/R':
        mfv = sum([
            ((data['Close'] - data['low']) - (data['high'] - data['Close'])) / (data['high'] - data['low']).replace([np.inf, -np.inf], 0).fillna(0) * data['volume']
            .rolling(window=config[f'range{i}'], min_periods=1).sum().rolling(window=config[f'range{i}'], min_periods=1).apply(lambda x: (x[-1] - np.mean(x)) / np.std(x) * 10 if np.std(x) > 0 else 0)
            for i in range(1,5)
        ])
        pivot_high = data['high'][(data['high'] == data['high'].rolling(window=config['pivot_lookback']*2+1, center=True).max())]
        pivot_low = data['low'][(data['low'] == data['low'].rolling(window=config['pivot_lookback']*2+1, center=True).min())]
        resistance_levels, support_levels = [], []
        bull_attack, bear_attack = [], []
        max_levels = 10
        for i in range(len(data)):
            if i in pivot_high.index:
                resistance_levels.insert(0, data['high'].iloc[i])
                resistance_levels = resistance_levels[:max_levels]
            if i in pivot_low.index:
                support_levels.insert(0, data['low'].iloc[i])
                support_levels = support_levels[:max_levels]
            close = data['Close'].iloc[i]
            near_res = any(abs(close - res)/res <= 0.00001 for res in resistance_levels)
            near_sup = any(abs(close - sup)/sup <= 0.00001 for sup in support_levels)
            bull_attack.append(near_res and mfv.iloc[i] > 0)
            bear_attack.append(near_sup and mfv.iloc[i] < 0)
        data['EyeX MFV S/R Bull'] = bull_attack
        data['EyeX MFV S/R Bear'] = bear_attack
        return data['EyeX MFV S/R Bull']
    elif indicator == 'OBV Price Divergence':
        obv = ta.OBV(data['Close'], data['volume'])
        obv_ma = ta.SMA(obv, timeperiod=config['obv_period'])
        price_ma = ta.SMA(data['Close'], timeperiod=config['obv_period'])
        return (obv_ma - price_ma).fillna(0)
    else:
        return None

def main():
    if len(sys.argv) != 3:
        print("Usage: python tweak-indicator.py SYMBOL TIMEFRAME")
        sys.exit(1)
    symbol, timeframe = sys.argv[1], sys.argv[2]
    db = CorrelationDatabase(DB_PATH)
    indicators = list_indicators()
    print("Do you want to experiment with tweaking the settings for a specific indicator? (y/n) [n]: ", end='')
    choice = input().strip().lower()
    if choice != 'y':
        sys.exit(0)
    print("Select an indicator to tweak:")
    for idx, ind in enumerate(indicators, 1):
        print(f"{idx}. {ind}")
    selection = input(f"Enter number (Default: 1): ").strip()
    try:
        selected_indicator = indicators[int(selection)-1] if selection else indicators[0]
    except:
        selected_indicator = indicators[0]
    params = default_parameters.get(selected_indicator, {})
    if not params:
        print("Selected indicator has no configurable parameters.")
        sys.exit(0)
    print(f"Configurable parameters for {selected_indicator}: {list(params.keys())}")
    config_combinations = get_configurations(params, default_parameters[selected_indicator])
    # Calculate total number of configurations
    total_configs = 1
    for param in params:
        total_configs *= len(range(max(1, default_parameters[selected_indicator][param] -5), min(100, default_parameters[selected_indicator][param] +5) +1))
    print(f"Total configurations to process: {total_configs}")
    data = pd.read_sql_query("""
        SELECT klines.*, symbols.symbol, timeframes.timeframe FROM klines
        JOIN symbols ON klines.symbol_id = symbols.id
        JOIN timeframes ON klines.timeframe_id = timeframes.id
        WHERE symbols.symbol = ? AND timeframes.timeframe = ?
        ORDER BY open_time ASC
    """, db.conn, params=(symbol, timeframe))
    if data.empty:
        print("No data found for the given symbol and timeframe.")
        sys.exit(1)
    data = compute_all_indicators(data)
    correlations = []
    param_names = list(params.keys())
    processed = 0
    for config in config_combinations:
        config_dict = dict(zip(param_names, config))
        indicator_name = f"{selected_indicator}_" + "_".join([f"{k}{v}" for k,v in config_dict.items()])
        existing = db.conn.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,)).fetchone()
        if not existing:
            db.conn.execute("INSERT INTO indicators (name) VALUES (?)", (indicator_name,))
            db.conn.commit()
        indicator_id = db.conn.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,)).fetchone()[0]
        indicator_series = compute_indicator(data, selected_indicator, config_dict, param_names)
        if indicator_series is None:
            continue
        data[indicator_name] = indicator_series
        for lag in range(1, 101):
            shifted = indicator_series.shift(lag)
            valid = pd.concat([shifted, data['Close']], axis=1).dropna()
            if not valid.empty:
                corr = valid.iloc[:,0].corr(valid.iloc[:,1])
                db.insert_correlation(symbol, timeframe, indicator_name, lag, corr)
                correlations.append({'indicator': indicator_name, 'config': config_dict, 'lag': lag, 'correlation': corr})
        processed +=1
        if processed % 100 ==0:
            print(f"Processed {processed}/{total_configs} configurations...")
    df_corr = pd.DataFrame(correlations)
    df_corr.to_csv(f"csv/{selected_indicator}_correlations.csv", index=False)
    # Visualization
    plt.figure(figsize=(15, 10))
    for ind in df_corr['indicator'].unique():
        subset = df_corr[df_corr['indicator'] == ind]
        plt.plot(subset['lag'], subset['correlation'], label=ind)
    plt.title(f'Correlations for {selected_indicator} Configurations')
    plt.xlabel('Lag Period')
    plt.ylabel('Correlation')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"reports/{selected_indicator}_correlations.png", bbox_inches='tight')
    plt.close()
    print("All configurations processed and visualizations generated.")
    db.close()

if __name__ == "__main__":
    main()
