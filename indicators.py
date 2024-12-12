# filename: indicators.py
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta

def compute_obv_price_divergence(data, method="Difference", obv_method="SMA", obv_period=14, price_input_type="OHLC/4", price_method="SMA", price_period=14, bearish_threshold=-0.8, bullish_threshold=0.8, smoothing=0.01):
    price_map = {"close": data['close'], "open": data['open'], "high": data['high'], "low": data['low'], "hl/2": (data['high'] + data['low']) / 2, "ohlc/4": (data[['open','high','low','close']].sum(axis=1) / 4)}
    selected_price = price_map.get(price_input_type.lower(), (data['open'] + data['high'] + data['low'] + data['close']) / 4)
    obv = ta.OBV(data['close'], data['volume'])
    obv_ma = ta.SMA(obv, timeperiod=obv_period) if obv_method.upper() == "SMA" else ta.EMA(obv, timeperiod=obv_period)
    price_ma = ta.SMA(selected_price, timeperiod=price_period) if price_method.upper() == "SMA" else ta.EMA(selected_price, timeperiod=price_period)
    obv_change = (obv_ma - obv_ma.shift(1)) / obv_ma.shift(1) * 100
    price_change = (price_ma - price_ma.shift(1)) / price_ma.shift(1) * 100
    metric = obv_change - price_change if method == "Difference" else obv_change / np.maximum(smoothing, np.abs(price_change)) if method == "Ratio" else np.log(np.maximum(smoothing, np.abs(obv_change)) / np.maximum(smoothing, np.abs(price_change)))
    data['obv_price_divergence'] = metric
    return data

def compute_eyeX_MFV_volume(data, ranges=[50,75,100,200]):
    mf_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
    mf_volume = mf_multiplier * data['volume']
    combined_mfv = sum([
        (mf_volume.rolling(window=br, min_periods=1).sum() - mf_volume.shift(br).fillna(0))
        .rolling(window=br, min_periods=1)
        .apply(lambda x: (x - np.mean(x)) / (np.std(x) if np.std(x) != 0 else 1), raw=True) * 10
        for br in ranges
    ]).clip(-400, 400)
    data['EyeX MFV Volume'] = combined_mfv
    return data

def compute_eyeX_MFV_support_resistance(data, ranges=[50,75,100,200], pivot_lookback=5, price_proximity=0.00001):
    mf_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    mf_multiplier = mf_multiplier.replace([np.inf, -np.inf], 0).fillna(0)
    mf_volume = mf_multiplier * data['volume']
    combined_mfv = sum([
        (mf_volume.rolling(window=br, min_periods=1).sum() - mf_volume.shift(br).fillna(0))
        .rolling(window=br, min_periods=1)
        .apply(lambda x: (x - np.mean(x)) / (np.std(x) if np.std(x) != 0 else 1), raw=True) * 10
        for br in ranges
    ])
    pivot_high = data['high'][(data['high'] == data['high'].rolling(window=pivot_lookback*2+1, center=True).max())]
    pivot_low = data['low'][(data['low'] == data['low'].rolling(window=pivot_lookback*2+1, center=True).min())]
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
        close = data['close'].iloc[i]
        near_res = any(abs(close - res)/res <= price_proximity for res in resistance_levels)
        near_sup = any(abs(close - sup)/sup <= price_proximity for sup in support_levels)
        bull_attack.append(near_res and combined_mfv.iloc[i] > 0)
        bear_attack.append(near_sup and combined_mfv.iloc[i] < 0)
    data['EyeX MFV S/R Bull'] = bull_attack
    data['EyeX MFV S/R Bear'] = bear_attack
    return data

def compute_all_indicators(data):
    indicators = {}
    indicators['bbands_upper'], indicators['bbands_middle'], indicators['bbands_lower'] = ta.BBANDS(data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
    indicators['dema'] = ta.DEMA(data['close'], timeperiod=30)
    indicators['ema'] = ta.EMA(data['close'], timeperiod=30)
    indicators['ht_trendline'] = ta.HT_TRENDLINE(data['close'])
    indicators['kama'] = ta.KAMA(data['close'], timeperiod=30)
    indicators['ma'] = ta.MA(data['close'], timeperiod=30, matype=0)
    indicators['mama'], indicators['fama'] = ta.MAMA(data['close'], fastlimit=0.5, slowlimit=0.05)
    indicators['midpoint'] = ta.MIDPOINT(data['close'], timeperiod=14)
    indicators['midprice'] = ta.MIDPRICE(data['high'], data['low'], timeperiod=14)
    indicators['sar'] = ta.SAR(data['high'], data['low'], acceleration=0.02, maximum=0.2)
    indicators['sma'] = ta.SMA(data['close'], timeperiod=30)
    indicators['t3'] = ta.T3(data['close'], timeperiod=5, vfactor=0.7)
    indicators['tema'] = ta.TEMA(data['close'], timeperiod=30)
    indicators['trima'] = ta.TRIMA(data['close'], timeperiod=30)
    indicators['wma'] = ta.WMA(data['close'], timeperiod=30)
    indicators['adx'] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['adxr'] = ta.ADXR(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['apo'] = ta.APO(data['close'], fastperiod=12, slowperiod=26, matype=0)
    indicators['aroon_down'], indicators['aroon_up'] = ta.AROON(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['aroonosc'] = ta.AROONOSC(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['bop'] = ta.BOP(data['open'], data['high'], data['low'], data['close'])
    indicators['cci'] = ta.CCI(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['cmo'] = ta.CMO(data['close'], timeperiod=14)
    indicators['dx'] = ta.DX(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    indicators['macdext'], indicators['macdext_signal'], indicators['macdext_hist'] = ta.MACDEXT(data['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
    indicators['macdfix'], indicators['macdfix_signal'], indicators['macdfix_hist'] = ta.MACDFIX(data['close'], signalperiod=9)
    indicators['minus_di'] = ta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['minus_dm'] = ta.MINUS_DM(data['high'], data['low'], timeperiod=14)
    indicators['mom'] = ta.MOM(data['close'], timeperiod=10)
    indicators['plus_di'] = ta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['plus_dm'] = ta.PLUS_DM(data['high'], data['low'], timeperiod=14)
    indicators['ppo'] = ta.PPO(data['close'], fastperiod=12, slowperiod=26, matype=0)
    indicators['roc'] = ta.ROC(data['close'], timeperiod=10)
    indicators['rocp'] = ta.ROCP(data['close'], timeperiod=10)
    indicators['rocr'] = ta.ROCR(data['close'], timeperiod=10)
    indicators['rocr100'] = ta.ROCR100(data['close'], timeperiod=10)
    indicators['rsi'] = ta.RSI(data['close'], timeperiod=14)
    indicators['stoch_slowk'], indicators['stoch_slowd'] = ta.STOCH(data['high'], data['low'], data['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    indicators['stochf_fastk'], indicators['stochf_fastd'] = ta.STOCHF(data['high'], data['low'], data['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
    indicators['stochrsi_fastk'], indicators['stochrsi_fastd'] = ta.STOCHRSI(data['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
    indicators['trix'] = ta.TRIX(data['close'], timeperiod=30)
    indicators['ultosc'] = ta.ULTOSC(data['high'], data['low'], data['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    indicators['willr'] = ta.WILLR(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['ad'] = ta.AD(data['high'], data['low'], data['close'], data['volume'])
    indicators['adosc'] = ta.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=3, slowperiod=10)
    indicators['obv'] = ta.OBV(data['close'], data['volume'])
    indicators['volume_osc'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).mean()
    indicators['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
    indicators['pvi'] = data['volume'].diff().apply(lambda x:1 if x > 0 else 0).cumsum() * data['close']
    indicators['nvi'] = data['volume'].diff().apply(lambda x:1 if x < 0 else 0).cumsum() * data['close']
    indicators['atr'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['natr'] = ta.NATR(data['high'], data['low'], data['close'], timeperiod=14)
    indicators['trange'] = ta.TRANGE(data['high'], data['low'], data['close'])
    indicators['ht_dcperiod'] = ta.HT_DCPERIOD(data['close'])
    indicators['ht_dcpphase'] = ta.HT_DCPHASE(data['close'])
    indicators['ht_phasor_inphase'], indicators['ht_phasor_quadrature'] = ta.HT_PHASOR(data['close'])
    indicators['ht_sine_sine'], indicators['ht_sine_leadsine'] = ta.HT_SINE(data['close'])
    indicators['ht_trendmode'] = ta.HT_TRENDMODE(data['close'])
    indicators['beta'] = ta.BETA(data['high'], data['low'], timeperiod=5)
    indicators['correl'] = ta.CORREL(data['high'], data['low'], timeperiod=30)
    indicators['linearreg'] = ta.LINEARREG(data['close'], timeperiod=14)
    indicators['linearreg_angle'] = ta.LINEARREG_ANGLE(data['close'], timeperiod=14)
    indicators['linearreg_intercept'] = ta.LINEARREG_INTERCEPT(data['close'], timeperiod=14)
    indicators['linearreg_slope'] = ta.LINEARREG_SLOPE(data['close'], timeperiod=14)
    indicators['stddev'] = ta.STDDEV(data['close'], timeperiod=5, nbdev=1)
    indicators['tsf'] = ta.TSF(data['close'], timeperiod=14)
    indicators['var'] = ta.VAR(data['close'], timeperiod=5, nbdev=1)
    try: indicators['ao'] = pta.ao(data['high'], data['low'])
    except: indicators['ao'] = None
    try: indicators['fi'] = pta.fi(data['close'], data['volume'])
    except: indicators['fi'] = (data['close'] - data['close'].shift(1)) * data['volume']
    try:
        kc = data.ta.kc(append=False)
        indicators['kc_upper'], indicators['kc_middle'], indicators['kc_lower'] = kc.get('kcu_20_2.0', None), kc.get('kcm_20_2.0', None), kc.get('kcl_20_2.0', None)
    except:
        indicators['kc_upper'] = indicators['kc_middle'] = indicators['kc_lower'] = None
    try: indicators['mfi'] = pta.mfi(data['high'], data['low'], data['close'], data['volume'])
    except: indicators['mfi'] = None
    try: indicators['rvi'] = pta.rvi(data['close'])
    except: indicators['rvi'] = None
    try:
        stochrsi = data.ta.stochrsi(append=False)
        indicators['stochrsi_fastk'] = stochrsi.get('stochrsi_14_5_3_slowk', None)
        indicators['stochrsi_fastd'] = stochrsi.get('stochrsi_14_5_3_slowd', None)
    except:
        indicators['stochrsi_fastk'] = indicators['stochrsi_fastd'] = None
    try:
        tsi = pta.tsi(data['close'])
        for col in tsi.columns: indicators[col] = tsi[col]
    except: pass
    try:
        vortex = data.ta.vortex(append=False)
        indicators['vi_plus'], indicators['vi_minus'] = vortex.get('vi+_14', None), vortex.get('vi-_14', None)
    except:
        indicators['vi_plus'] = indicators['vi_minus'] = None
    data = compute_obv_price_divergence(data)
    data = compute_eyeX_MFV_volume(data)
    data = compute_eyeX_MFV_support_resistance(data)
    for k, v in indicators.items():
        if v is not None: data[k] = v
    data.dropna(inplace=True)
    return data