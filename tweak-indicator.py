# tweak-indicator.py
import sys, itertools, pandas as pd, numpy as np, sqlite3, talib as ta
from correlation_database import CorrelationDatabase
from indicators import compute_all_indicators
from config import DB_PATH
import matplotlib.pyplot as plt

default_parameters = {'MACD': {'fastperiod': 12, 'slowperiod': 26, 'signalperiod': 9}, 'EMA': {'timeperiod': 30}, 'SMA': {'timeperiod': 30}, 'RSI': {'timeperiod': 14}, 'ATR': {'timeperiod': 14}, 'ADX': {'timeperiod': 14}, 'CCI': {'timeperiod': 14}, 'MOM': {'timeperiod': 10}, 'ROC': {'timeperiod': 10}, 'TRIX': {'timeperiod': 30}, 'ULTOSC': {'timeperiod1': 7, 'timeperiod2': 14, 'timeperiod3': 28}, 'EyeX MFV Volume': {'range1': 50, 'range2': 75, 'range3': 100, 'range4': 200}, 'EyeX MFV S/R': {'range1': 50, 'range2': 75, 'range3': 100, 'range4': 200, 'pivot_lookback': 5}, 'OBV Price Divergence': {'obv_period': 14}}

def list_indicators():
    db = CorrelationDatabase(DB_PATH)
    indicators = sorted([ind[0] for ind in db.conn.execute("SELECT name FROM indicators").fetchall() if ind[0] in default_parameters])
    db.close()
    return indicators

def get_configurations(params, defaults):
    return itertools.product(*[range(max(1, defaults.get(p, 1) - 5), min(100, defaults.get(p, 1) + 5) + 1) for p in params])

def compute_indicator(data, indicator, config, params):
    if indicator == 'MACD':
        return ta.MACD(data['Close'], **config)[0]
    if indicator in {'EMA', 'SMA', 'RSI', 'ATR', 'ADX', 'CCI', 'MOM', 'ROC', 'TRIX'}:
        return getattr(ta, indicator)(data['Close'], **config)
    if indicator == 'ULTOSC':
        return ta.ULTOSC(data['high'], data['low'], data['Close'], **config)
    if indicator == 'EyeX MFV Volume':
        mfv = ((data['Close'] - data['low']) - (data['high'] - data['Close'])) / (data['high'] - data['low']).replace([np.inf, -np.inf], 0).fillna(0) * data['volume']
        return sum([(mfv.rolling(window=config[f'range{i}']).sum() - mfv.shift(config[f'range{i}']).fillna(0)).rolling(window=config[f'range{i}']).apply(lambda x: (x[-1] - np.mean(x)) / np.std(x) * 10 if np.std(x) > 0 else 0, raw=True) for i in range(1, 5)]).clip(-400, 400)
    if indicator == 'EyeX MFV S/R':
        mfv = sum([((data['Close'] - data['low']) - (data['high'] - data['Close'])) / (data['high'] - data['low']).replace([np.inf, -np.inf], 0).fillna(0) * data['volume'].rolling(window=config[f'range{i}']).sum().rolling(window=config[f'range{i}']).apply(lambda x: (x[-1] - np.mean(x)) / np.std(x) * 10 if np.std(x) > 0 else 0) for i in range(1, 5)])
        pivot_high = data['high'][(data['high'] == data['high'].rolling(window=config['pivot_lookback'] * 2 + 1, center=True).max())]
        pivot_low = data['low'][(data['low'] == data['low'].rolling(window=config['pivot_lookback'] * 2 + 1, center=True).min())]
        res, sup, bull, bear = [], [], [], []
        for i in range(len(data)):
            if i in pivot_high.index: res.insert(0, data['high'].iloc[i]); res = res[:10]
            if i in pivot_low.index: sup.insert(0, data['low'].iloc[i]); sup = sup[:10]
            close = data['Close'].iloc[i]
            bull.append(any(abs(close - r) / r <= 0.00001 for r in res) and mfv.iloc[i] > 0)
            bear.append(any(abs(close - s) / s <= 0.00001 for s in sup) and mfv.iloc[i] < 0)
        data['EyeX MFV S/R Bull'], data['EyeX MFV S/R Bear'] = bull, bear
        return data['EyeX MFV S/R Bull']
    if indicator == 'OBV Price Divergence':
        obv = ta.OBV(data['Close'], data['volume'])
        return (ta.SMA(obv, timeperiod=config['obv_period']) - ta.SMA(data['Close'], timeperiod=config['obv_period'])).fillna(0)

def main():
    if len(sys.argv) != 3: print("Usage: python tweak-indicator.py SYMBOL TIMEFRAME"); sys.exit(1)
    symbol, timeframe = sys.argv[1:]
    db, indicators = CorrelationDatabase(DB_PATH), list_indicators()
    if input("Do you want to experiment with tweaking the settings for a specific indicator? (y/n) [n]: ").strip().lower() != 'y': sys.exit(0)
    print("Select an indicator to tweak:", *[f"{i + 1}. {ind}" for i, ind in enumerate(indicators)], sep='\n')
    selected_indicator = indicators[int(input(f"Enter number (Default: 1): ") or 1) - 1]
    params, configs = default_parameters.get(selected_indicator, {}), list(get_configurations(default_parameters[selected_indicator], default_parameters[selected_indicator]))
    data = pd.read_sql_query("""
        SELECT klines.*, symbols.symbol, timeframes.timeframe FROM klines
        JOIN symbols ON klines.symbol_id = symbols.id
        JOIN timeframes ON klines.timeframe_id = timeframes.id
        WHERE symbols.symbol = ? AND timeframes.timeframe = ?
        ORDER BY open_time ASC
    """, db.conn, params=(symbol, timeframe))
    if data.empty: print("No data found."); sys.exit(1)
    data, correlations, processed = compute_all_indicators(data), [], 0
    for config in configs:
        config_dict, indicator_name = dict(zip(params.keys(), config)), f"{selected_indicator}_" + "_".join([f"{k}{v}" for k, v in zip(params.keys(), config)])
        if not db.conn.execute("SELECT id FROM indicators WHERE name = ?", (indicator_name,)).fetchone():
            db.conn.execute("INSERT INTO indicators (name) VALUES (?)", (indicator_name,))
            db.conn.commit()
        indicator_series = compute_indicator(data, selected_indicator, config_dict, params)
        if indicator_series is None: continue
        data[indicator_name] = indicator_series
        for lag in range(1, 101):
            shifted, valid = indicator_series.shift(lag), pd.concat([indicator_series.shift(lag), data['Close']], axis=1).dropna()
            if not valid.empty:
                corr = valid.iloc[:, 0].corr(valid.iloc[:, 1])
                db.insert_correlation(symbol, timeframe, indicator_name, lag, corr)
                correlations.append({'indicator': indicator_name, 'config': config_dict, 'lag': lag, 'correlation': corr})
        processed += 1
        if processed % 100 == 0: print(f"Processed {processed}/{len(configs)} configurations...")
    df_corr = pd.DataFrame(correlations)
    df_corr.to_csv(f"csv/{selected_indicator}_correlations.csv", index=False)
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
