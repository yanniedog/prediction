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
    else:
        selected_price=(data['open']+data['high']+data['low']+data['close'])/4
    obv=ta.OBV(data['close'],data['volume'])
    obv_ma=ta.SMA(obv,timeperiod=obv_period) if obv_method=="SMA" else (ta.EMA(obv,timeperiod=obv_period) if obv_method=="EMA" else obv)
    price_ma=ta.SMA(selected_price,timeperiod=price_period) if price_method=="SMA" else (ta.EMA(selected_price,timeperiod=price_period) if price_method=="EMA" else selected_price)
    obv_change_percent=(obv_ma - obv_ma.shift(1))/obv_ma.shift(1)*100
    price_change_percent=(price_ma - price_ma.shift(1))/price_ma.shift(1)*100
    if method=="Difference":
        metric=obv_change_percent - price_change_percent
    elif method=="Ratio":
        metric=obv_change_percent/np.maximum(smoothing,np.abs(price_change_percent))
    else:
        metric=np.log(np.maximum(smoothing,np.abs(obv_change_percent))/np.maximum(smoothing,np.abs(price_change_percent)))
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
    indicators['macd'],indicators['macd_signal'],indicators['macd_hist']=ta.MACD(data['close'],12,26,9)
    indicators['macdext'],indicators['macdext_signal'],indicators['macdext_hist']=ta.MACDEXT(data['close'],12,0,26,0,9,0)
    indicators['macdfix'],indicators['macdfix_signal'],indicators['macdfix_hist']=ta.MACDFIX(data['close'],9)
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
    indicators['stoch_slowk'],indicators['stoch_slowd']=ta.STOCH(data['high'],data['low'],data['close'],5,3,0,3,0)
    indicators['stochf_fastk'],indicators['stochf_fastd']=ta.STOCHF(data['high'],data['low'],data['close'],5,3,0)
    indicators['stochrsi_fastk'],indicators['stochrsi_fastd']=ta.STOCHRSI(data['close'],14,5,3,0)
    indicators['trix']=ta.TRIX(data['close'],timeperiod=30)
    indicators['ultosc']=ta.ULTOSC(data['high'],data['low'],data['close'],7,14,28)
    indicators['willr']=ta.WILLR(data['high'],data['low'],data['close'],timeperiod=14)
    indicators['ad']=ta.AD(data['high'],data['low'],data['close'],data['volume'])
    indicators['adosc']=ta.ADOSC(data['high'],data['low'],data['close'],data['volume'],3,10)
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
    except:
        pass
    try:
        fi=pta.fi(data['close'],data['volume'])
        indicators['fi']=fi
    except:
        data['fi']=(data['close']-data['close'].shift(1))*data['volume']
        indicators['fi']=data['fi']
    try:
        ichimoku=data.ta.ichimoku(append=False)
        for col in ['isa_9','isb_26','its_9','iks_26']:
            if col not in ichimoku.columns: raise KeyError
        indicators['ichimoku_conversion']=ichimoku['isa_9']
        indicators['ichimoku_base']=ichimoku['isb_26']
        indicators['ichimoku_span_a']=ichimoku['its_9']
        indicators['ichimoku_span_b']=ichimoku['iks_26']
    except:
        pass
    try:
        kc=data.ta.kc(append=False)
        for col in ['kcu_20_2.0','kcm_20_2.0','kcl_20_2.0']:
            if col not in kc.columns: raise KeyError
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
        for col in ['stochrsi_14_5_3_slowk','stochrsi_14_5_3_slowd']:
            if col not in stochrsi.columns: raise KeyError
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
        for col in ['vi+_14','vi-_14']:
            if col not in vortex.columns: raise KeyError
        indicators['vi_plus']=vortex['vi+_14']
        indicators['vi_minus']=vortex['vi-_14']
    except:
        pass
    data=compute_obv_price_divergence(data)
    for k,v in indicators.items():
        data[k]=v
    data.dropna(inplace=True)
    return data
