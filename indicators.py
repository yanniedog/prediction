# indicators.py
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta

def compute_obv_price_divergence(
    data, method="Difference", obv_method="SMA", obv_period=14,
    price_input_type="OHLC/4", price_method="SMA", price_period=14,
    bearish_threshold=-0.8, bullish_threshold=0.8, smoothing=0.01
):
    price_map = {
        "close": data['close'],
        "open": data['open'],
        "high": data['high'],
        "low": data['low'],
        "hl/2": (data['high'] + data['low']) / 2,
        "ohlc/4": (data['open'] + data['high'] + data['low'] + data['close']) / 4
    }
    selected_price = price_map.get(price_input_type.lower())
    if selected_price is None:
        raise ValueError(f"Unsupported price input type: {price_input_type}")

    obv = ta.OBV(data['close'], data['volume'])
    obv_ma = ta.SMA(obv, timeperiod=obv_period) if obv_method == "SMA" else ta.EMA(obv, timeperiod=obv_period) if obv_method == "EMA" else obv

    price_ma = ta.SMA(selected_price, timeperiod=price_period) if price_method == "SMA" else ta.EMA(selected_price, timeperiod=price_period) if price_method == "EMA" else selected_price

    obv_change = obv_ma.pct_change() * 100
    price_change = price_ma.pct_change() * 100

    metric = {
        "Difference": obv_change - price_change,
        "Ratio": obv_change / np.maximum(smoothing, np.abs(price_change)),
        "Log Ratio": np.log(np.maximum(smoothing, np.abs(obv_change)) / np.maximum(smoothing, np.abs(price_change)))
    }.get(method)

    if metric is None:
        raise ValueError(f"Unsupported method: {method}")

    data['obv_price_divergence'] = metric
    return data

def compute_all_indicators(data):
    indicators = {
        'bbands_upper': ta.BBANDS(data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[0],
        'bbands_middle': ta.BBANDS(data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[1],
        'bbands_lower': ta.BBANDS(data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)[2],
        'dema': ta.DEMA(data['close'], timeperiod=30),
        'ema': ta.EMA(data['close'], timeperiod=30),
        'ht_trendline': ta.HT_TRENDLINE(data['close']),
        'kama': ta.KAMA(data['close'], timeperiod=30),
        'ma': ta.MA(data['close'], timeperiod=30, matype=0),
        'mama': ta.MAMA(data['close'], fastlimit=0.5, slowlimit=0.05)[0],
        'fama': ta.MAMA(data['close'], fastlimit=0.5, slowlimit=0.05)[1],
        'midpoint': ta.MIDPOINT(data['close'], timeperiod=14),
        'midprice': ta.MIDPRICE(data['high'], data['low'], timeperiod=14),
        'sar': ta.SAR(data['high'], data['low'], acceleration=0.02, maximum=0.2),
        'sma': ta.SMA(data['close'], timeperiod=30),
        't3': ta.T3(data['close'], timeperiod=5, vfactor=0.7),
        'tema': ta.TEMA(data['close'], timeperiod=30),
        'trima': ta.TRIMA(data['close'], timeperiod=30),
        'wma': ta.WMA(data['close'], timeperiod=30),
        'adx': ta.ADX(data['high'], data['low'], data['close'], timeperiod=14),
        'adxr': ta.ADXR(data['high'], data['low'], data['close'], timeperiod=14),
        'apo': ta.APO(data['close'], fastperiod=12, slowperiod=26, matype=0),
        'aroon_down': ta.AROON(data['high'], data['low'], timeperiod=14)[0],
        'aroon_up': ta.AROON(data['high'], data['low'], timeperiod=14)[1],
        'aroonosc': ta.AROONOSC(data['high'], data['low'], timeperiod=14),
        'bop': ta.BOP(data['open'], data['high'], data['low'], data['close']),
        'cci': ta.CCI(data['high'], data['low'], data['close'], timeperiod=14),
        'cmo': ta.CMO(data['close'], timeperiod=14),
        'dx': ta.DX(data['high'], data['low'], data['close'], timeperiod=14),
        'macd': ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)[0],
        'macd_signal': ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)[1],
        'macd_hist': ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)[2],
        'macdext': ta.MACDEXT(data['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)[0],
        'macdext_signal': ta.MACDEXT(data['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)[1],
        'macdext_hist': ta.MACDEXT(data['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)[2],
        'macdfix': ta.MACDFIX(data['close'], signalperiod=9)[0],
        'macdfix_signal': ta.MACDFIX(data['close'], signalperiod=9)[1],
        'macdfix_hist': ta.MACDFIX(data['close'], signalperiod=9)[2],
        'minus_di': ta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14),
        'minus_dm': ta.MINUS_DM(data['high'], data['low'], timeperiod=14),
        'mom': ta.MOM(data['close'], timeperiod=10),
        'plus_di': ta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14),
        'plus_dm': ta.PLUS_DM(data['high'], data['low'], timeperiod=14),
        'ppo': ta.PPO(data['close'], fastperiod=12, slowperiod=26, matype=0),
        'roc': ta.ROC(data['close'], timeperiod=10),
        'rocp': ta.ROCP(data['close'], timeperiod=10),
        'rocr': ta.ROCR(data['close'], timeperiod=10),
        'rocr100': ta.ROCR100(data['close'], timeperiod=10),
        'rsi': ta.RSI(data['close'], timeperiod=14),
        'stoch_slowk': ta.STOCH(data['high'], data['low'], data['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[0],
        'stoch_slowd': ta.STOCH(data['high'], data['low'], data['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)[1],
        'stochf_fastk': ta.STOCHF(data['high'], data['low'], data['close'], fastk_period=5, fastd_period=3, fastd_matype=0)[0],
        'stochf_fastd': ta.STOCHF(data['high'], data['low'], data['close'], fastk_period=5, fastd_period=3, fastd_matype=0)[1],
        'stochrsi_fastk': ta.STOCHRSI(data['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[0],
        'stochrsi_fastd': ta.STOCHRSI(data['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[1],
        'trix': ta.TRIX(data['close'], timeperiod=30),
        'ultosc': ta.ULTOSC(data['high'], data['low'], data['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28),
        'willr': ta.WILLR(data['high'], data['low'], data['close'], timeperiod=14),
        'ad': ta.AD(data['high'], data['low'], data['close'], data['volume']),
        'adosc': ta.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=3, slowperiod=10),
        'obv': ta.OBV(data['close'], data['volume']),
        'volume_osc': (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).mean(),
        'vwap': (data['close'] * data['volume']).cumsum() / data['volume'].cumsum(),
        'pvi': (data['volume'].diff().gt(0).astype(int).cumsum() * data['close']),
        'nvi': (data['volume'].diff().lt(0).astype(int).cumsum() * data['close']),
        'atr': ta.ATR(data['high'], data['low'], data['close'], timeperiod=14),
        'natr': ta.NATR(data['high'], data['low'], data['close'], timeperiod=14),
        'trange': ta.TRANGE(data['high'], data['low'], data['close']),
        'ht_dcperiod': ta.HT_DCPERIOD(data['close']),
        'ht_dcpphase': ta.HT_DCPHASE(data['close']),
        'ht_phasor_inphase': ta.HT_PHASOR(data['close'])[0],
        'ht_phasor_quadrature': ta.HT_PHASOR(data['close'])[1],
        'ht_sine_sine': ta.HT_SINE(data['close'])[0],
        'ht_sine_leadsine': ta.HT_SINE(data['close'])[1],
        'ht_trendmode': ta.HT_TRENDMODE(data['close']),
        'beta': ta.BETA(data['high'], data['low'], timeperiod=5),
        'correl': ta.CORREL(data['high'], data['low'], timeperiod=30),
        'linearreg': ta.LINEARREG(data['close'], timeperiod=14),
        'linearreg_angle': ta.LINEARREG_ANGLE(data['close'], timeperiod=14),
        'linearreg_intercept': ta.LINEARREG_INTERCEPT(data['close'], timeperiod=14),
        'linearreg_slope': ta.LINEARREG_SLOPE(data['close'], timeperiod=14),
        'stddev': ta.STDDEV(data['close'], timeperiod=5, nbdev=1),
        'tsf': ta.TSF(data['close'], timeperiod=14),
        'var': ta.VAR(data['close'], timeperiod=5, nbdev=1)
    }

    # Attempt to add indicators from pandas_ta
    try:
        indicators.update({
            'ao': pta.ao(data['high'], data['low']),
            'fi': pta.fi(data['close'], data['volume']),
            'ichimoku_conversion': pta.ichimoku(data)['ISA_9'],
            'ichimoku_base': pta.ichimoku(data)['ISB_26'],
            'ichimoku_span_a': pta.ichimoku(data)['ITS_9'],
            'ichimoku_span_b': pta.ichimoku(data)['IKS_26'],
            'kc_upper': pta.kc(data)['KCU_20_2.0'],
            'kc_middle': pta.kc(data)['KCM_20_2.0'],
            'kc_lower': pta.kc(data)['KCL_20_2.0'],
            'mfi': pta.mfi(data['high'], data['low'], data['close'], data['volume']),
            'rvi': pta.rvi(data['close']),
            'stochrsi_slowk': ta.STOCHRSI(data['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[0],
            'stochrsi_slowd': ta.STOCHRSI(data['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)[1],
            'tsi': pta.tsi(data['close'])
        })
    except:
        pass

    data = compute_obv_price_divergence(data)

    for key, value in indicators.items():
        data[key] = value

    # --- Start of EyeX MFV Indicators ---
    # Parameters
    bar_ranges = [50, 75, 100, 200]
    pivot_lookback = 5
    price_proximity = 0.00001
    max_levels = 10

    # Money Flow Volume Calculation
    mf_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
    mf_volume = mf_multiplier * data['volume']

    # Cumulative MFV with rolling sums
    cum_mfv = pd.concat([mf_volume.rolling(window=br, min_periods=1).sum() for br in bar_ranges], axis=1)
    cum_mfv.columns = [f'cumMFV_{br}' for br in bar_ranges]
    normalized_mfv = (cum_mfv - cum_mfv.mean()) / cum_mfv.std()
    mfv_lines = normalized_mfv.clip(-10, 10) * 10
    combined_mfv = mfv_lines.sum(axis=1).clip(-100, 100)
    data['eyex_mfv_volume'] = combined_mfv

    # Pivot Highs and Lows
    pivot_high = data['high'][(data['high'] == data['high'].rolling(window=2 * pivot_lookback + 1, center=True).max())]
    pivot_low = data['low'][(data['low'] == data['low'].rolling(window=2 * pivot_lookback + 1, center=True).min())]

    # Initialize support and resistance levels
    resistance_levels, support_levels = [], []
    eyex_sup_res = np.zeros(len(data))

    for idx in range(len(data)):
        price = data['close'].iloc[idx]
        if pivot_high.iloc[idx] == price:
            resistance_levels.insert(0, price)
            if len(resistance_levels) > max_levels:
                resistance_levels.pop()
        if pivot_low.iloc[idx] == price:
            support_levels.insert(0, price)
            if len(support_levels) > max_levels:
                support_levels.pop()

        near_res = any(abs(price - res) / res <= price_proximity for res in resistance_levels)
        near_sup = any(abs(price - sup) / sup <= price_proximity for sup in support_levels)

        if near_res and combined_mfv.iloc[idx] > 0:
            eyex_sup_res[idx] = 1
        elif near_sup and combined_mfv.iloc[idx] < 0:
            eyex_sup_res[idx] = -1

    data['eyex_mfv_sup_res'] = eyex_sup_res

    # --- End of EyeX MFV Indicators ---

    data.dropna(inplace=True)
    return data
