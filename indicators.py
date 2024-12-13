# indicators.py

import logging
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import json

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
    if method == "Difference":
        metric = obv_change - price_change
    elif method == "Ratio":
        metric = obv_change / np.maximum(smoothing, np.abs(price_change))
    else:
        metric = np.log(np.maximum(smoothing, np.abs(obv_change)) / np.maximum(smoothing, np.abs(price_change)))
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
    try:
        indicators['ao'] = pta.ao(data['high'], data['low'])
    except:
        indicators['ao'] = None
    try:
        indicators['fi'] = pta.fi(data['close'], data['volume'])
    except:
        indicators['fi'] = (data['close'] - data['close'].shift(1)) * data['volume']
    try:
        kc = data.ta.kc(append=False)
        indicators['kc_upper'], indicators['kc_middle'], indicators['kc_lower'] = kc.get('kcu_20_1.5', None), kc.get('kcm_20_1.5', None), kc.get('kcl_20_1.5', None)
    except:
        indicators['kc_upper'] = indicators['kc_middle'] = indicators['kc_lower'] = None
    try:
        indicators['mfi'] = pta.mfi(data['high'], data['low'], data['close'], data['volume'], length=14)
    except:
        indicators['mfi'] = None
    try:
        indicators['rvi'] = pta.rvi(data['close'], length=10)
    except:
        indicators['rvi'] = None
    try:
        stochrsi = data.ta.stochrsi(append=False)
        indicators['stochrsi_fastk'] = stochrsi.get('stochrsi_14_5_3_slowk', None)
        indicators['stochrsi_fastd'] = stochrsi.get('stochrsi_14_5_3_slowd', None)
    except:
        indicators['stochrsi_fastk'] = indicators['stochrsi_fastd'] = None
    try:
        tsi = pta.tsi(data['close'], fast=25, slow=13)
        for col in tsi.columns:
            indicators[col] = tsi[col]
    except:
        pass
    try:
        vortex = data.ta.vortex(append=False)
        indicators['vi_plus'], indicators['vi_minus'] = vortex.get('vi+_14', None), vortex.get('vi-_14', None)
    except:
        indicators['vi_plus'] = indicators['vi_minus'] = None
    data = compute_obv_price_divergence(data)
    data = compute_eyeX_MFV_volume(data)
    data = compute_eyeX_MFV_support_resistance(data)
    for k, v in indicators.items():
        if v is not None:
            data[k] = v
    data.dropna(inplace=True)
    return data

def compute_configured_indicators(data, indicators_list):
    """
    Compute configured indicators based on the provided list and parameters from indicator_params.json.
    
    Args:
        data (pd.DataFrame): The input data containing 'open', 'high', 'low', 'close', 'volume'.
        indicators_list (List[str]): List of indicator names to compute.
    
    Returns:
        pd.DataFrame: The data with configured indicators added.
    """
    # Load indicator parameters from JSON
    with open('indicator_params.json', 'r') as f:
        indicator_params = json.load(f)
    
    for indicator_name in indicators_list:
        if indicator_name not in indicator_params:
            logger.warning(f"No parameters found for '{indicator_name}'. Skipping configuration.")
            continue
        params = indicator_params[indicator_name]
        base_indicator = indicator_name
        # Handle special cases
        if indicator_name.startswith("EyeX"):
            if indicator_name == "EyeX MFV Volume":
                ranges = params.get("ranges", [50, 75, 100, 200])
                data = compute_eyeX_MFV_volume(data, ranges=ranges)
                logger.info(f"Computed configured indicator: {indicator_name}")
            elif indicator_name == "EyeX MFV S/R Bull":
                ranges = params.get("ranges", [50, 75, 100, 200])
                pivot_lookback = params.get("pivot_lookback", 5)
                price_proximity = params.get("price_proximity", 0.00001)
                data = compute_eyeX_MFV_support_resistance(data, ranges=ranges, pivot_lookback=pivot_lookback, price_proximity=price_proximity)
                logger.info(f"Computed configured indicator: {indicator_name}")
        elif base_indicator == "obv_price_divergence":
            method = params.get("method", "Difference")
            obv_method = params.get("obv_method", "SMA")
            obv_period = params.get("obv_period", 14)
            price_input_type = params.get("price_input_type", "OHLC/4")
            price_method = params.get("price_method", "SMA")
            price_period = params.get("price_period", 14)
            bearish_threshold = params.get("bearish_threshold", -0.8)
            bullish_threshold = params.get("bullish_threshold", 0.8)
            smoothing = params.get("smoothing", 0.01)
            data = compute_obv_price_divergence(
                data,
                method=method,
                obv_method=obv_method,
                obv_period=obv_period,
                price_input_type=price_input_type,
                price_method=price_method,
                price_period=price_period,
                bearish_threshold=bearish_threshold,
                bullish_threshold=bullish_threshold,
                smoothing=smoothing
            )
            logger.info(f"Computed configured indicator: {indicator_name}")
        else:
            # Handle standard indicators
            try:
                if base_indicator in ['dema', 'sma', 'ema', 'tsf', 'rsi', 'cmo', 'cci', 'adx', 'dx', 'aroon', 'aroonosc', 'trix', 'ultosc', 'willr', 'mfi', 'vortex']:
                    if base_indicator == 'dema':
                        timeperiod = params.get('timeperiod', 30)
                        data[indicator_name] = ta.DEMA(data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'sma':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = ta.SMA(data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'ema':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = ta.EMA(data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'tsf':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = ta.TSF(data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'rsi':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = ta.RSI(data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'cmo':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = ta.CMO(data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'cci':
                        timeperiod = params.get('timeperiod', 20)
                        data[indicator_name] = ta.CCI(data['high'], data['low'], data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'adx':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'dx':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = ta.DX(data['high'], data['low'], data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'aroon':
                        timeperiod = params.get('timeperiod', 14)
                        aroon_down, aroon_up = ta.AROON(data['high'], data['low'], timeperiod=timeperiod)
                        data[f"{indicator_name}_down"] = aroon_down
                        data[f"{indicator_name}_up"] = aroon_up
                    elif base_indicator == 'aroonosc':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = ta.AROONOSC(data['high'], data['low'], timeperiod=timeperiod)
                    elif base_indicator == 'trix':
                        timeperiod = params.get('timeperiod', 30)
                        data[indicator_name] = ta.TRIX(data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'ultosc':
                        timeperiod1 = params.get('timeperiod1', 7)
                        timeperiod2 = params.get('timeperiod2', 14)
                        timeperiod3 = params.get('timeperiod3', 28)
                        data[indicator_name] = ta.ULTOSC(data['high'], data['low'], data['close'], timeperiod1=timeperiod1, timeperiod2=timeperiod2, timeperiod3=timeperiod3)
                    elif base_indicator == 'willr':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = ta.WILLR(data['high'], data['low'], data['close'], timeperiod=timeperiod)
                    elif base_indicator == 'mfi':
                        timeperiod = params.get('timeperiod', 14)
                        data[indicator_name] = pta.mfi(data['high'], data['low'], data['close'], data['volume'], length=timeperiod)
                    elif base_indicator == 'vortex':
                        timeperiod = params.get('timeperiod', 14)
                        vortex = ta.VORTEX(data['high'], data['low'], data['close'], timeperiod=timeperiod)
                        data[f"{indicator_name}_vi_plus"] = vortex['VI+']
                        data[f"{indicator_name}_vi_minus"] = vortex['VI-']
                    logger.info(f"Computed configured indicator: {indicator_name}")
                elif base_indicator in ['macd', 'macdext', 'macdfix', 'ppo']:
                    # Handle MACD variations
                    if base_indicator == 'macd':
                        fastperiod = params.get('fastperiod', 12)
                        slowperiod = params.get('slowperiod', 26)
                        signalperiod = params.get('signalperiod', 9)
                        macd, macd_signal, macd_hist = ta.MACD(data['close'], fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
                        data[f"{indicator_name}_macd"] = macd
                        data[f"{indicator_name}_macd_signal"] = macd_signal
                        data[f"{indicator_name}_macd_hist"] = macd_hist
                    elif base_indicator == 'macdext':
                        fastperiod = params.get('fastperiod', 12)
                        fastmatype = params.get('fastmatype', 0)
                        slowperiod = params.get('slowperiod', 26)
                        slowmatype = params.get('slowmatype', 0)
                        signalperiod = params.get('signalperiod', 9)
                        signalmatype = params.get('signalmatype', 0)
                        macdext, macdext_signal, macdext_hist = ta.MACDEXT(data['close'], fastperiod=fastperiod, fastmatype=fastmatype, slowperiod=slowperiod, slowmatype=slowmatype, signalperiod=signalperiod, signalmatype=signalmatype)
                        data[f"{indicator_name}_macdext"] = macdext
                        data[f"{indicator_name}_macdext_signal"] = macdext_signal
                        data[f"{indicator_name}_macdext_hist"] = macdext_hist
                    elif base_indicator == 'macdfix':
                        signalperiod = params.get('signalperiod', 9)
                        macdfix, macdfix_signal, macdfix_hist = ta.MACDFIX(data['close'], signalperiod=signalperiod)
                        data[f"{indicator_name}_macdfix"] = macdfix
                        data[f"{indicator_name}_macdfix_signal"] = macdfix_signal
                        data[f"{indicator_name}_macdfix_hist"] = macdfix_hist
                    elif base_indicator == 'ppo':
                        fastperiod = params.get('fastperiod', 12)
                        slowperiod = params.get('slowperiod', 26)
                        matype = params.get('matype', 0)
                        ppo = ta.PPO(data['close'], fastperiod=fastperiod, slowperiod=slowperiod, matype=matype)
                        data[indicator_name] = ppo
                    logger.info(f"Computed configured indicator: {indicator_name}")
                elif base_indicator in ['ao', 'fi', 'kc', 'tsi']:
                    if base_indicator == 'ao':
                        timeperiod = params.get('timeperiod', 5)
                        data[indicator_name] = pta.ao(data['high'], data['low'], length=timeperiod)
                    elif base_indicator == 'fi':
                        timeperiod = params.get('timeperiod', 13)
                        data[indicator_name] = pta.fi(data['close'], data['volume'], length=timeperiod)
                    elif base_indicator == 'kc':
                        length = params.get('length', 20)
                        scalar = params.get('scalar', 1.5)
                        kc = ta.KC(data['high'], data['low'], data['close'], timeperiod=length, atrmultiplier=scalar)
                        data['kc_upper'] = kc[0]
                        data['kc_middle'] = kc[1]
                        data['kc_lower'] = kc[2]
                    elif base_indicator == 'tsi':
                        fastperiod = params.get('fastperiod', 25)
                        slowperiod = params.get('slowperiod', 13)
                        tsi = pta.tsi(data['close'], fast=fastperiod, slow=slowperiod)
                        for col in tsi.columns:
                            data[col] = tsi[col]
                    logger.info(f"Computed configured indicator: {indicator_name}")
                else:
                    logger.warning(f"Indicator '{indicator_name}' not recognized for configuration.")
            except Exception as e:
                logger.error(f"Error computing indicator '{indicator_name}': {e}")
        logger.info(f"Completed computation for indicator: {indicator_name}")
    data.dropna(inplace=True)
    return data
