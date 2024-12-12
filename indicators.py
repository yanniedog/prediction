import pandas as pd, numpy as np, talib as ta, pandas_ta as pta

def compute_obv_price_divergence(data, method="Difference", obv_method="SMA", obv_period=14, price_input_type="OHLC/4", price_method="SMA", price_period=14, bearish_threshold=-0.8, bullish_threshold=0.8, smoothing=0.01):
    price_map = {
        "close": data['close'], "open": data['open'], "high": data['high'], "low": data['low'],
        "hl/2": (data['high'] + data['low']) / 2, "ohlc/4": (data[['open','high','low','close']].sum(axis=1) / 4)
    }
    selected_price = price_map.get(price_input_type.lower(), (data['open'] + data['high'] + data['low'] + data['close']) / 4)
    obv = ta.OBV(data['close'], data['volume'])
    obv_ma = ta.SMA(obv, timeperiod=obv_period) if obv_method == "SMA" else ta.EMA(obv, timeperiod=obv_period)
    price_ma = ta.SMA(selected_price, timeperiod=price_period) if price_method == "SMA" else ta.EMA(selected_price, timeperiod=price_period)
    obv_change = (obv_ma - obv_ma.shift(1)) / obv_ma.shift(1) * 100
    price_change = (price_ma - price_ma.shift(1)) / price_ma.shift(1) * 100
    metric = obv_change - price_change if method == "Difference" else obv_change / np.maximum(smoothing, np.abs(price_change)) if method == "Ratio" else np.log(np.maximum(smoothing, np.abs(obv_change)) / np.maximum(smoothing, np.abs(price_change)))
    data['obv_price_divergence'] = metric
    return data

def compute_all_indicators(data):
    indicators = {
        'bbands_upper': ta.BBANDS(data['close'],5,2,2,0)[0],
        'bbands_middle': ta.BBANDS(data['close'],5,2,2,0)[1],
        'bbands_lower': ta.BBANDS(data['close'],5,2,2,0)[2],
        'dema': ta.DEMA(data['close'],30),
        'ema': ta.EMA(data['close'],30),
        'ht_trendline': ta.HT_TRENDLINE(data['close']),
        'kama': ta.KAMA(data['close'],30),
        'ma': ta.MA(data['close'],30,0),
        'mama': ta.MAMA(data['close'],0.5,0.05)[0],
        'fama': ta.MAMA(data['close'],0.5,0.05)[1],
        'midpoint': ta.MIDPOINT(data['close'],14),
        'midprice': ta.MIDPRICE(data['high'], data['low'],14),
        'sar': ta.SAR(data['high'], data['low'],0.02,0.2),
        'sma': ta.SMA(data['close'],30),
        't3': ta.T3(data['close'],5,0.7),
        'tema': ta.TEMA(data['close'],30),
        'trima': ta.TRIMA(data['close'],30),
        'wma': ta.WMA(data['close'],30),
        'adx': ta.ADX(data['high'], data['low'], data['close'],14),
        'adxr': ta.ADXR(data['high'], data['low'], data['close'],14),
        'apo': ta.APO(data['close'],12,26,0),
        'aroon_down': ta.AROON(data['high'], data['low'],14)[0],
        'aroon_up': ta.AROON(data['high'], data['low'],14)[1],
        'aroonosc': ta.AROONOSC(data['high'], data['low'],14),
        'bop': ta.BOP(data['open'], data['high'], data['low'], data['close']),
        'cci': ta.CCI(data['high'], data['low'], data['close'],14),
        'cmo': ta.CMO(data['close'],14),
        'dx': ta.DX(data['high'], data['low'], data['close'],14),
        'macd': ta.MACD(data['close'],12,26,9)[0],
        'macd_signal': ta.MACD(data['close'],12,26,9)[1],
        'macd_hist': ta.MACD(data['close'],12,26,9)[2],
        'macdext': ta.MACDEXT(data['close'],12,0,26,0,9,0)[0],
        'macdext_signal': ta.MACDEXT(data['close'],12,0,26,0,9,0)[1],
        'macdext_hist': ta.MACDEXT(data['close'],12,0,26,0,9,0)[2],
        'macdfix': ta.MACDFIX(data['close'],9)[0],
        'macdfix_signal': ta.MACDFIX(data['close'],9)[1],
        'macdfix_hist': ta.MACDFIX(data['close'],9)[2],
        'minus_di': ta.MINUS_DI(data['high'], data['low'], data['close'],14),
        'minus_dm': ta.MINUS_DM(data['high'], data['low'],14),
        'mom': ta.MOM(data['close'],10),
        'plus_di': ta.PLUS_DI(data['high'], data['low'], data['close'],14),
        'plus_dm': ta.PLUS_DM(data['high'], data['low'],14),
        'ppo': ta.PPO(data['close'],12,26,0),
        'roc': ta.ROC(data['close'],10),
        'rocp': ta.ROCP(data['close'],10),
        'rocr': ta.ROCR(data['close'],10),
        'rocr100': ta.ROCR100(data['close'],10),
        'rsi': ta.RSI(data['close'],14),
        'stoch_slowk': ta.STOCH(data['high'], data['low'], data['close'],5,3,0)[0],
        'stoch_slowd': ta.STOCH(data['high'], data['low'], data['close'],5,3,0)[1],
        'stochf_fastk': ta.STOCHF(data['high'], data['low'], data['close'],5,3,0)[0],
        'stochf_fastd': ta.STOCHF(data['high'], data['low'], data['close'],5,3,0)[1],
        'stochrsi_fastk': ta.STOCHRSI(data['close'],14,5,3,0)[0],
        'stochrsi_fastd': ta.STOCHRSI(data['close'],14,5,3,0)[1],
        'trix': ta.TRIX(data['close'],30),
        'ultosc': ta.ULTOSC(data['high'], data['low'], data['close'],7,14,28),
        'willr': ta.WILLR(data['high'], data['low'], data['close'],14),
        'ad': ta.AD(data['high'], data['low'], data['close'], data['volume']),
        'adosc': ta.ADOSC(data['high'], data['low'], data['close'], data['volume'],3,10),
        'obv': ta.OBV(data['close'], data['volume']),
        'volume_osc': (data['volume'] - data['volume'].rolling(20).mean()) / data['volume'].rolling(20).mean(),
        'vwap': (data['close'] * data['volume']).cumsum() / data['volume'].cumsum(),
        'pvi': data['volume'].diff().apply(lambda x:1 if x>0 else 0).cumsum() * data['close'],
        'nvi': data['volume'].diff().apply(lambda x:1 if x<0 else 0).cumsum() * data['close'],
        'atr': ta.ATR(data['high'], data['low'], data['close'],14),
        'natr': ta.NATR(data['high'], data['low'], data['close'],14),
        'trange': ta.TRANGE(data['high'], data['low'], data['close']),
        'ht_dcperiod': ta.HT_DCPERIOD(data['close']),
        'ht_dcpphase': ta.HT_DCPHASE(data['close']),
        'ht_phasor_inphase': ta.HT_PHASOR(data['close'])[0],
        'ht_phasor_quadrature': ta.HT_PHASOR(data['close'])[1],
        'ht_sine_sine': ta.HT_SINE(data['close'])[0],
        'ht_sine_leadsine': ta.HT_SINE(data['close'])[1],
        'ht_trendmode': ta.HT_TRENDMODE(data['close']),
        'beta': ta.BETA(data['high'], data['low'],5),
        'correl': ta.CORREL(data['high'], data['low'],30),
        'linearreg': ta.LINEARREG(data['close'],14),
        'linearreg_angle': ta.LINEARREG_ANGLE(data['close'],14),
        'linearreg_intercept': ta.LINEARREG_INTERCEPT(data['close'],14),
        'linearreg_slope': ta.LINEARREG_SLOPE(data['close'],14),
        'stddev': ta.STDDEV(data['close'],5,1),
        'tsf': ta.TSF(data['close'],14),
        'var': ta.VAR(data['close'],5,1),
        'ao': pta.ao(data['high'], data['low']) if 'ao' in pta.available_indicators() else None,
        'fi': pta.fi(data['close'], data['volume']) if 'fi' in pta.available_indicators() else (data['close'].diff() * data['volume']),
        'ichimoku_conversion': data.ta.ichimoku(append=False)['isa_9'] if 'ichimoku' in data.ta.list_indicators() else None,
        'ichimoku_base': data.ta.ichimoku(append=False)['isb_26'] if 'ichimoku' in data.ta.list_indicators() else None,
        'ichimoku_span_a': data.ta.ichimoku(append=False)['its_9'] if 'ichimoku' in data.ta.list_indicators() else None,
        'ichimoku_span_b': data.ta.ichimoku(append=False)['iks_26'] if 'ichimoku' in data.ta.list_indicators() else None,
        'kc_upper': data.ta.kc(append=False)['kcu_20_2.0'] if 'kc' in data.ta.list_indicators() else None,
        'kc_middle': data.ta.kc(append=False)['kcm_20_2.0'] if 'kc' in data.ta.list_indicators() else None,
        'kc_lower': data.ta.kc(append=False)['kcl_20_2.0'] if 'kc' in data.ta.list_indicators() else None,
        'mfi': pta.mfi(data['high'], data['low'], data['close'], data['volume']) if 'mfi' in pta.available_indicators() else None,
        'rvi': pta.rvi(data['close']) if 'rvi' in pta.available_indicators() else None,
        'stochrsi_slowk': data.ta.stochrsi(append=False)['stochrsi_14_5_3_slowk'] if 'stochrsi' in data.ta.list_indicators() else None,
        'stochrsi_slowd': data.ta.stochrsi(append=False)['stochrsi_14_5_3_slowd'] if 'stochrsi' in data.ta.list_indicators() else None,
        'tsi': pta.tsi(data['close']) if 'tsi' in pta.available_indicators() else None,
        'vi_plus': data.ta.vortex(append=False)['vi+_14'] if 'vortex' in data.ta.list_indicators() else None,
        'vi_minus': data.ta.vortex(append=False)['vi-_14'] if 'vortex' in data.ta.list_indicators() else None,
    }
    data = compute_obv_price_divergence(data)
    for k, v in indicators.items():
        if v is not None:
            data[k] = v
    data.dropna(inplace=True)
    return data
