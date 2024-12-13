# indicators.py
import logging
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta

logger = logging.getLogger()

def z_score(x):
    mean = np.mean(x)
    std = np.std(x)
    if std == 0:
        return 0
    return (x[-1] - mean) / std

def compute_obv_price_divergence(data, method="Difference", obv_method="SMA", obv_period=14, price_input_type="OHLC/4", price_method="SMA", price_period=14, bearish_threshold=-0.8, bullish_threshold=0.8, smoothing=0.01):
    price_map = {
        "close": data['close'],
        "open": data['open'],
        "high": data['high'],
        "low": data['low'],
        "hl/2": (data['high'] + data['low']) / 2,
        "ohlc/4": (data[['open','high','low','close']].sum(axis=1) / 4)
    }
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
        .apply(z_score, raw=True) * 10
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
        .apply(z_score, raw=True) * 10
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
    indicators['aroon_down'], indicators['aroon_up'] = ta.AROON(data['high'], data['low'], timeperiod=14)
    indicators['aroonosc'] = ta.AROONOSC(data['high'], data['low'], timeperiod=14)
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

def compute_configured_indicators(data, indicators):
    for indicator_name in indicators:
        if '_' not in indicator_name and 'EyeX MFV S/R' not in indicator_name and indicator_name not in ['obv_price_divergence', 'dema', 't3', 'sma', 'ema', 'tsf']:
            if indicator_name not in data.columns:
                pass
            continue
        parts = indicator_name.split('_')
        base_indicator = parts[0]
        params = {}
        if base_indicator == 'EyeX':
            if 'MFV Volume' in indicator_name:
                base_indicator = 'EyeX MFV Volume'
                params = {'ranges': [50,75,100,200]}
            elif 'MFV S/R Bull' in indicator_name:
                base_indicator = 'EyeX MFV S/R Bull'
                params = {'ranges': [50,75,100,200], 'pivot_lookback':5, 'price_proximity':0.00001}
        elif base_indicator == 'obv_price_divergence':
            params = {
                'method': 'Difference',
                'obv_method': 'SMA',
                'obv_period': 14,
                'price_input_type': 'OHLC/4',
                'price_method': 'SMA',
                'price_period': 14,
                'bearish_threshold': -0.8,
                'bullish_threshold': 0.8,
                'smoothing': 0.01
            }
        elif base_indicator == 'dema':
            params = {
                'timeperiod': 30
            }
        elif base_indicator in ['t3', 'sma', 'ema', 'tsf']:
            params = {
                'timeperiod': 14  # Example default value
            }
        else:
            for part in parts[1:]:
                key = ''.join(filter(str.isalpha, part))
                value = ''.join(filter(lambda c: c.isdigit() or c == '.', part))
                if key and value:
                    try:
                        if '.' in value:
                            params[key] = float(value)
                        else:
                            params[key] = int(value)
                    except:
                        pass
        if base_indicator == 't3':
            timeperiod = params.get('timeperiod', 5)
            vfactor = params.get('vfactor', 0.7)
            column_name = indicator_name
            data[column_name] = ta.T3(data['close'], timeperiod=timeperiod, vfactor=vfactor)
        elif base_indicator == 'sma':
            timeperiod = params.get('timeperiod', 14)
            column_name = indicator_name
            data[column_name] = ta.SMA(data['close'], timeperiod=timeperiod)
        elif base_indicator == 'ema':
            timeperiod = params.get('timeperiod', 14)
            column_name = indicator_name
            data[column_name] = ta.EMA(data['close'], timeperiod=timeperiod)
        elif base_indicator == 'tsf':
            timeperiod = params.get('timeperiod', 14)
            column_name = indicator_name
            data[column_name] = ta.TSF(data['close'], timeperiod=timeperiod)
        elif base_indicator == 'EyeX MFV Volume':
            ranges = params.get('ranges', [50,75,100,200])
            column_name = indicator_name
            data = compute_eyeX_MFV_volume(data, ranges=ranges)
        elif base_indicator == 'EyeX MFV S/R Bull':
            ranges = params.get('ranges', [50,75,100,200])
            pivot_lookback = params.get('pivot_lookback',5)
            price_proximity = params.get('price_proximity',0.00001)
            column_name = indicator_name
            data = compute_eyeX_MFV_support_resistance(data, ranges=ranges, pivot_lookback=pivot_lookback, price_proximity=price_proximity)
        elif base_indicator == 'obv_price_divergence':
            method = params.get('method', 'Difference')
            obv_method = params.get('obv_method', 'SMA')
            obv_period = params.get('obv_period', 14)
            price_input_type = params.get('price_input_type', 'OHLC/4')
            price_method = params.get('price_method', 'SMA')
            price_period = params.get('price_period', 14)
            bearish_threshold = params.get('bearish_threshold', -0.8)
            bullish_threshold = params.get('bullish_threshold', 0.8)
            smoothing = params.get('smoothing', 0.01)
            column_name = indicator_name
            data = compute_obv_price_divergence(data, method=method, obv_method=obv_method, obv_period=obv_period, price_input_type=price_input_type, price_method=price_method, price_period=price_period, bearish_threshold=bearish_threshold, bullish_threshold=bullish_threshold, smoothing=smoothing)
            logger.info(f"Computed configured indicator: {column_name}")
        elif base_indicator == 'dema':
            timeperiod = params.get('timeperiod', 30)
            column_name = indicator_name
            data[column_name] = ta.DEMA(data['close'], timeperiod=timeperiod)
            logger.info(f"Computed configured indicator: {column_name}")
        else:
            logger.error(f"Unknown indicator base: {base_indicator}. Skipping.")
        if base_indicator.startswith('EyeX') or base_indicator in ['obv_price_divergence', 'dema', 't3', 'sma', 'ema', 'tsf']:
            logger.info(f"Computed configured indicator: {column_name}")
    data.dropna(inplace=True)
    return data
