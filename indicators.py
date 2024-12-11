# indicators.py
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta

def compute_obv_price_divergence(data,method="Difference",obv_method="SMA",obv_period=14,price_input_type="OHLC/4",price_method="SMA",price_period=14,bearish_threshold=-0.8,bullish_threshold=0.8,smoothing=0.01):
    if price_input_type.lower()=="close":
        selected_price=data['close']
    elif price_input_type.lower()=="open":
        selected_price=data['open']
    elif price_input_type.lower()=="high":
        selected_price=data['high']
    elif price_input_type.lower()=="low":
        selected_price=data['low']
    elif price_input_type.lower()=="hl/2":
        selected_price=(data['high']+data['low'])/2
    elif price_input_type.lower()=="ohlc/4":
        selected_price=(data['open']+data['high']+data['low']+data['close'])/4
    else:
        raise ValueError(f"Unsupported price input type: {price_input_type}")
    obv=ta.OBV(data['close'],data['volume'])
    if obv_method=="SMA":
        obv_ma=ta.SMA(obv,timeperiod=obv_period)
    elif obv_method=="EMA":
        obv_ma=ta.EMA(obv,timeperiod=obv_period)
    else:
        obv_ma=obv
    if price_method=="SMA":
        price_ma=ta.SMA(selected_price,timeperiod=price_period)
    elif price_method=="EMA":
        price_ma=ta.EMA(selected_price,timeperiod=price_period)
    else:
        price_ma=selected_price
    obv_change_percent=(obv_ma - obv_ma.shift(1))/obv_ma.shift(1)*100
    price_change_percent=(price_ma - price_ma.shift(1))/price_ma.shift(1)*100
    if method=="Difference":
        metric=obv_change_percent - price_change_percent
    elif method=="Ratio":
        metric=obv_change_percent/np.maximum(smoothing,np.abs(price_change_percent))
    elif method=="Log Ratio":
        metric=np.log(np.maximum(smoothing,np.abs(obv_change_percent))/np.maximum(smoothing,np.abs(price_change_percent)))
    else:
        raise ValueError(f"Unsupported method: {method}")
    data['obv_price_divergence']=metric
    return data

def compute_all_indicators(data):
    indicators={}
    indicators['bbands_upper'],indicators['bbands_middle'],indicators['bbands_lower']=ta.BBANDS(data['close'],timeperiod=5,nbdevup=2,nbdevdn=2,matype=0)
    indicators['dema']=ta.DEMA(data['close'],timeperiod=30)
    indicators['ema']=ta.EMA(data['close'],timeperiod=30)
    indicators['ht_trendline']=ta.HT_TRENDLINE(data['close'])
    indicators['kama']=ta.KAMA(data['close'],timeperiod=30)
    indicators['ma']=ta.MA(data['close'],timeperiod=30,matype=0)
    indicators['mama'],indicators['fama']=ta.MAMA(data['close'],fastlimit=0.5,slowlimit=0.05)
    indicators['midpoint']=ta.MIDPOINT(data['close'],timeperiod=14)
    indicators['midprice']=ta.MIDPRICE(data['high'],data['low'],timeperiod=14)
    indicators['sar']=ta.SAR(data['high'],data['low'],acceleration=0.02,maximum=0.2)
    indicators['sma']=ta.SMA(data['close'],timeperiod=30)
    indicators['t3']=ta.T3(data['close'],timeperiod=5,vfactor=0.7)
    indicators['tema']=ta.TEMA(data['close'],timeperiod=30)
    indicators['trima']=ta.TRIMA(data['close'],timeperiod=30)
    indicators['wma']=ta.WMA(data['close'],timeperiod=30)
    indicators['adx']=ta.ADX(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['adxr']=ta.ADXR(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['apo']=ta.APO(data['close'],fastperiod=12,slowperiod=26,matype=0)
    indicators['aroon_down'],indicators['aroon_up']=ta.AROON(data['high'],data['low'],timeperiod=14)
    indicators['aroonosc']=ta.AROONOSC(data['high'],data['low'],timeperiod=14)
    indicators['bop']=ta.BOP(data['open'],data['high'],data['low'],data['close'])
    indicators['cci']=ta.CCI(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['cmo']=ta.CMO(data['close'],timeperiod=14)
    indicators['dx']=ta.DX(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['macd'],indicators['macd_signal'],indicators['macd_hist']=ta.MACD(data['close'],fastperiod=12,slowperiod=26,signalperiod=9)
    indicators['macdext'],indicators['macdext_signal'],indicators['macdext_hist']=ta.MACDEXT(data['close'],fastperiod=12,fastmatype=0,slowperiod=26,slowmatype=0,signalperiod=9,signalmatype=0)
    indicators['macdfix'],indicators['macdfix_signal'],indicators['macdfix_hist']=ta.MACDFIX(data['close'],signalperiod=9)
    indicators['minus_di']=ta.MINUS_DI(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['minus_dm']=ta.MINUS_DM(data['high'],data['low'],timeperiod=14)
    indicators['mom']=ta.MOM(data['close'],timeperiod=10)
    indicators['plus_di']=ta.PLUS_DI(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['plus_dm']=ta.PLUS_DM(data['high'],data['low'],timeperiod=14)
    indicators['ppo']=ta.PPO(data['close'],fastperiod=12,slowperiod=26,matype=0)
    indicators['roc']=ta.ROC(data['close'],timeperiod=10)
    indicators['rocp']=ta.ROCP(data['close'],timeperiod=10)
    indicators['rocr']=ta.ROCR(data['close'],timeperiod=10)
    indicators['rocr100']=ta.ROCR100(data['close'],timeperiod=10)
    indicators['rsi']=ta.RSI(data['close'],timeperiod=14)
    indicators['stoch_slowk'],indicators['stoch_slowd']=ta.STOCH(data['high'],data['low'],data['close'],fastk_period=5,slowk_period=3,slowk_matype=0,slowd_period=3,slowd_matype=0)
    indicators['stochf_fastk'],indicators['stochf_fastd']=ta.STOCHF(data['high'],data['low'],data['close'],fastk_period=5,fastd_period=3,fastd_matype=0)
    indicators['stochrsi_fastk'],indicators['stochrsi_fastd']=ta.STOCHRSI(data['close'],timeperiod=14,fastk_period=5,fastd_period=3,fastd_matype=0)
    indicators['trix']=ta.TRIX(data['close'],timeperiod=30)
    indicators['ultosc']=ta.ULTOSC(data['high'],data['low'],data['close'],timeperiod1=7,timeperiod2=14,timeperiod3=28)
    indicators['willr']=ta.WILLR(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['ad']=ta.AD(data['high'],data['low'],data['close'],data['volume'])
    indicators['adosc']=ta.ADOSC(data['high'],data['low'],data['close'],data['volume'],fastperiod=3,slowperiod=10)
    indicators['obv']=ta.OBV(data['close'],data['volume'])
    indicators['volume_osc']=(data['volume']-data['volume'].rolling(window=20).mean())/data['volume'].rolling(window=20).mean()
    indicators['vwap']=(data['close']*data['volume']).cumsum()/data['volume'].cumsum()
    indicators['pvi']=data['volume'].diff().apply(lambda x:1 if x>0 else 0).cumsum()*data['close']
    indicators['nvi']=data['volume'].diff().apply(lambda x:1 if x<0 else 0).cumsum()*data['close']
    indicators['atr']=ta.ATR(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['natr']=ta.NATR(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['trange']=ta.TRANGE(data['high'],data['low'],data['close'])
    indicators['ht_dcperiod']=ta.HT_DCPERIOD(data['close'])
    indicators['ht_dcpphase']=ta.HT_DCPHASE(data['close'])
    indicators['ht_phasor_inphase'],indicators['ht_phasor_quadrature']=ta.HT_PHASOR(data['close'])
    indicators['ht_sine_sine'],indicators['ht_sine_leadsine']=ta.HT_SINE(data['close'])
    indicators['ht_trendmode']=ta.HT_TRENDMODE(data['close'])
    indicators['beta']=ta.BETA(data['high'],data['low'],timeperiod=5)
    indicators['correl']=ta.CORREL(data['high'],data['low'],timeperiod=30)
    indicators['linearreg']=ta.LINEARREG(data['close'],timeperiod=14)
    indicators['linearreg_angle']=ta.LINEARREG_ANGLE(data['close'],timeperiod=14)
    indicators['linearreg_intercept']=ta.LINEARREG_INTERCEPT(data['close'],timeperiod=14)
    indicators['linearreg_slope']=ta.LINEARREG_SLOPE(data['close'],timeperiod=14)
    indicators['stddev']=ta.STDDEV(data['close'],timeperiod=5,nbdev=1)
    indicators['tsf']=ta.TSF(data['close'],timeperiod=14)
    indicators['var']=ta.VAR(data['close'],timeperiod=5,nbdev=1)
    try:
        ao=pta.ao(data['high'],data['low'])
        indicators['ao']=ao
    except AttributeError:
        pass
    try:
        fi=pta.fi(data['close'],data['volume'])
        indicators['fi']=fi
    except AttributeError:
        data['fi']=(data['close']-data['close'].shift(1))*data['volume']
        indicators['fi']=data['fi']
    try:
        ichimoku=data.ta.ichimoku(append=False)
        expected_columns=['isa_9','isb_26','its_9','iks_26']
        for col in expected_columns:
            if col not in ichimoku.columns:
                raise KeyError(col)
        indicators['ichimoku_conversion']=ichimoku['isa_9']
        indicators['ichimoku_base']=ichimoku['isb_26']
        indicators['ichimoku_span_a']=ichimoku['its_9']
        indicators['ichimoku_span_b']=ichimoku['iks_26']
    except:
        pass
    try:
        kc=data.ta.kc(append=False)
        expected_columns=['kcu_20_2.0','kcm_20_2.0','kcl_20_2.0']
        for col in expected_columns:
            if col not in kc.columns:
                raise KeyError(col)
        indicators['kc_upper']=kc['kcu_20_2.0']
        indicators['kc_middle']=kc['kcm_20_2.0']
        indicators['kc_lower']=kc['kcl_20_2.0']
    except:
        pass
    try:
        mfi=pta.mfi(data['high'],data['low'],data['close'],data['volume'])
        indicators['mfi']=mfi
    except:
        pass
    try:
        rvi=pta.rvi(data['close'])
        indicators['rvi']=rvi
    except:
        pass
    try:
        stochrsi=data.ta.stochrsi(append=False)
        expected_columns=['stochrsi_14_5_3_slowk','stochrsi_14_5_3_slowd']
        for col in expected_columns:
            if col not in stochrsi.columns:
                raise KeyError(col)
        indicators['stochrsi_slowk']=stochrsi['stochrsi_14_5_3_slowk']
        indicators['stochrsi_slowd']=stochrsi['stochrsi_14_5_3_slowd']
    except:
        pass
    try:
        tsi=pta.tsi(data['close'])
        for col in tsi.columns:
            indicators[col]=tsi[col]
    except:
        pass
    try:
        vortex=data.ta.vortex(append=False)
        expected_columns=['vi+_14','vi-_14']
        for col in expected_columns:
            if col not in vortex.columns:
                raise KeyError(col)
        indicators['vi_plus']=vortex['vi+_14']
        indicators['vi_minus']=vortex['vi-_14']
    except:
        pass
    data=compute_obv_price_divergence(data)
    for key,value in indicators.items():
        data[key]=value
    data.dropna(inplace=True)
    return data