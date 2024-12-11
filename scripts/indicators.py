# indicators.py
import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_obv_price_divergence(data: pd.DataFrame, method="Difference", obv_method="SMA", obv_period=14, price_input_type="OHLC/4", price_method="SMA", price_period=14, bearish_threshold=-0.8, bullish_threshold=0.8, smoothing=0.01) -> pd.DataFrame:
    """
    Compute On-Balance Volume (OBV) and Price Divergence metrics.

    Parameters:
    - data: DataFrame containing 'close', 'volume', 'open', 'high', 'low' columns.
    - method: Method to calculate divergence ('Difference', 'Ratio', 'Log').
    - obv_method: Method to smooth OBV ('SMA', 'EMA', or None).
    - obv_period: Period for OBV moving average.
    - price_input_type: Type of price to use ('Close', 'Open', 'High', 'Low', 'HL/2', 'OHLC/4').
    - price_method: Method to smooth price ('SMA', 'EMA', or None).
    - price_period: Period for price moving average.
    - bearish_threshold: Threshold for bearish divergence.
    - bullish_threshold: Threshold for bullish divergence.
    - smoothing: Smoothing factor to prevent division by zero.

    Returns:
    - DataFrame with 'obv_price_divergence' column added.
    """
    try:
        if price_input_type.lower() == "close":
            selected_price = data['close']
        elif price_input_type.lower() == "open":
            selected_price = data['open']
        elif price_input_type.lower() == "high":
            selected_price = data['high']
        elif price_input_type.lower() == "low":
            selected_price = data['low']
        elif price_input_type.lower() == "hl/2":
            selected_price = (data['high'] + data['low']) / 2
        else:
            selected_price = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        logging.debug(f"Selected price type '{price_input_type}' for divergence computation.")
        
        obv = ta.OBV(data['close'], data['volume'])
        logging.debug("Computed On-Balance Volume (OBV).")
        
        if obv_method == "SMA":
            obv_ma = ta.SMA(obv, timeperiod=obv_period)
            logging.debug(f"Applied SMA smoothing to OBV with period {obv_period}.")
        elif obv_method == "EMA":
            obv_ma = ta.EMA(obv, timeperiod=obv_period)
            logging.debug(f"Applied EMA smoothing to OBV with period {obv_period}.")
        else:
            obv_ma = obv
            logging.debug("No smoothing applied to OBV.")
        
        if price_method == "SMA":
            price_ma = ta.SMA(selected_price, timeperiod=price_period)
            logging.debug(f"Applied SMA smoothing to price with period {price_period}.")
        elif price_method == "EMA":
            price_ma = ta.EMA(selected_price, timeperiod=price_period)
            logging.debug(f"Applied EMA smoothing to price with period {price_period}.")
        else:
            price_ma = selected_price
            logging.debug("No smoothing applied to price.")
        
        obv_change_percent = (obv_ma - obv_ma.shift(1)) / obv_ma.shift(1) * 100
        price_change_percent = (price_ma - price_ma.shift(1)) / price_ma.shift(1) * 100
        logging.debug("Calculated percentage changes for OBV and Price.")
        
        if method == "Difference":
            metric = obv_change_percent - price_change_percent
            logging.debug("Computed divergence using 'Difference' method.")
        elif method == "Ratio":
            metric = obv_change_percent / np.maximum(smoothing, np.abs(price_change_percent))
            logging.debug("Computed divergence using 'Ratio' method.")
        else:
            metric = np.log(np.maximum(smoothing, np.abs(obv_change_percent)) / np.maximum(smoothing, np.abs(price_change_percent)))
            logging.debug("Computed divergence using 'Log' method.")
        
        data['obv_price_divergence'] = metric
        logging.info("Added 'obv_price_divergence' to DataFrame.")
        return data
    except Exception as e:
        logging.error(f"Error in compute_obv_price_divergence: {e}")
        data['obv_price_divergence'] = np.nan
        return data

def compute_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all necessary technical indicators and add them to the DataFrame.

    Parameters:
    - data: Original DataFrame containing the data.

    Returns:
    - DataFrame with added indicator columns.
    """
    indicators = {}
    try:
        indicators['bbands_upper'], indicators['bbands_middle'], indicators['bbands_lower'] = ta.BBANDS(data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        logging.debug("Computed Bollinger Bands.")
        
        indicators['dema'] = ta.DEMA(data['close'], timeperiod=30)
        logging.debug("Computed DEMA.")
        
        indicators['ema'] = ta.EMA(data['close'], timeperiod=30)
        logging.debug("Computed EMA.")
        
        indicators['ht_trendline'] = ta.HT_TRENDLINE(data['close'])
        logging.debug("Computed HT Trendline.")
        
        indicators['kama'] = ta.KAMA(data['close'], timeperiod=30)
        logging.debug("Computed KAMA.")
        
        indicators['ma'] = ta.MA(data['close'], timeperiod=30, matype=0)
        logging.debug("Computed MA.")
        
        indicators['mama'], indicators['fama'] = ta.MAMA(data['close'], fastlimit=0.5, slowlimit=0.05)
        logging.debug("Computed MAMA and FAMA.")
        
        indicators['midpoint'] = ta.MIDPOINT(data['close'], timeperiod=14)
        logging.debug("Computed Midpoint.")
        
        indicators['midprice'] = ta.MIDPRICE(data['high'], data['low'], timeperiod=14)
        logging.debug("Computed Midprice.")
        
        indicators['sar'] = ta.SAR(data['high'], data['low'], acceleration=0.02, maximum=0.2)
        logging.debug("Computed SAR.")
        
        indicators['sma'] = ta.SMA(data['close'], timeperiod=30)
        logging.debug("Computed SMA.")
        
        indicators['t3'] = ta.T3(data['close'], timeperiod=5, vfactor=0.7)
        logging.debug("Computed T3.")
        
        indicators['tema'] = ta.TEMA(data['close'], timeperiod=30)
        logging.debug("Computed TEMA.")
        
        indicators['trima'] = ta.TRIMA(data['close'], timeperiod=30)
        logging.debug("Computed TRIMA.")
        
        indicators['wma'] = ta.WMA(data['close'], timeperiod=30)
        logging.debug("Computed WMA.")
        
        indicators['adx'] = ta.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        logging.debug("Computed ADX.")
        
        indicators['adxr'] = ta.ADXR(data['high'], data['low'], data['close'], timeperiod=14)
        logging.debug("Computed ADXR.")
        
        indicators['apo'] = ta.APO(data['close'], fastperiod=12, slowperiod=26, matype=0)
        logging.debug("Computed APO.")
        
        indicators['aroon_down'], indicators['aroon_up'] = ta.AROON(data['high'], data['low'], timeperiod=14)
        logging.debug("Computed Aroon Down and Aroon Up.")
        
        indicators['aroonosc'] = ta.AROONOSC(data['high'], data['low'], timeperiod=14)
        logging.debug("Computed Aroon Oscillator.")
        
        indicators['bop'] = ta.BOP(data['open'], data['high'], data['low'], data['close'])
        logging.debug("Computed BOP.")
        
        indicators['cci'] = ta.CCI(data['high'], data['low'], data['close'], timeperiod=14)
        logging.debug("Computed CCI.")
        
        indicators['cmo'] = ta.CMO(data['close'], timeperiod=14)
        logging.debug("Computed CMO.")
        
        indicators['dx'] = ta.DX(data['high'], data['low'], data['close'], timeperiod=14)
        logging.debug("Computed DX.")
        
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = ta.MACD(data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        logging.debug("Computed MACD.")
        
        indicators['macdext'], indicators['macdext_signal'], indicators['macdext_hist'] = ta.MACDEXT(data['close'], fastperiod=12, fastmatype=0, slowperiod=26, slowmatype=0, signalperiod=9, signalmatype=0)
        logging.debug("Computed MACDEXT.")
        
        indicators['macdfix'], indicators['macdfix_signal'], indicators['macdfix_hist'] = ta.MACDFIX(data['close'], signalperiod=9)
        logging.debug("Computed MACDFIX.")
        
        indicators['minus_di'] = ta.MINUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        logging.debug("Computed Minus DI.")
        
        indicators['minus_dm'] = ta.MINUS_DM(data['high'], data['low'], timeperiod=14)
        logging.debug("Computed Minus DM.")
        
        indicators['mom'] = ta.MOM(data['close'], timeperiod=10)
        logging.debug("Computed MOM.")
        
        indicators['plus_di'] = ta.PLUS_DI(data['high'], data['low'], data['close'], timeperiod=14)
        logging.debug("Computed Plus DI.")
        
        indicators['plus_dm'] = ta.PLUS_DM(data['high'], data['low'], timeperiod=14)
        logging.debug("Computed Plus DM.")
        
        indicators['ppo'] = ta.PPO(data['close'], fastperiod=12, slowperiod=26, matype=0)
        logging.debug("Computed PPO.")
        
        indicators['roc'] = ta.ROC(data['close'], timeperiod=10)
        logging.debug("Computed ROC.")
        
        indicators['rocp'] = ta.ROCP(data['close'], timeperiod=10)
        logging.debug("Computed ROCP.")
        
        indicators['rocr'] = ta.ROCR(data['close'], timeperiod=10)
        logging.debug("Computed ROCR.")
        
        indicators['rocr100'] = ta.ROCR100(data['close'], timeperiod=10)
        logging.debug("Computed ROCR100.")
        
        indicators['rsi'] = ta.RSI(data['close'], timeperiod=14)
        logging.debug("Computed RSI.")
        
        indicators['stoch_slowk'], indicators['stoch_slowd'] = ta.STOCH(data['high'], data['low'], data['close'], fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        logging.debug("Computed Stochastic Oscillator SlowK and SlowD.")
        
        indicators['stochf_fastk'], indicators['stochf_fastd'] = ta.STOCHF(data['high'], data['low'], data['close'], fastk_period=5, fastd_period=3, fastd_matype=0)
        logging.debug("Computed Stochastic FastK and FastD.")
        
        indicators['stochrsi_fastk'], indicators['stochrsi_fastd'] = ta.STOCHRSI(data['close'], timeperiod=14, fastk_period=5, fastd_period=3, fastd_matype=0)
        logging.debug("Computed Stochastic RSI FastK and FastD.")
        
        indicators['trix'] = ta.TRIX(data['close'], timeperiod=30)
        logging.debug("Computed TRIX.")
        
        indicators['ultosc'] = ta.ULTOSC(data['high'], data['low'], data['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
        logging.debug("Computed Ultimate Oscillator.")
        
        indicators['willr'] = ta.WILLR(data['high'], data['low'], data['close'], timeperiod=14)
        logging.debug("Computed Williams %R.")
        
        indicators['ad'] = ta.AD(data['high'], data['low'], data['close'], data['volume'])
        logging.debug("Computed AD.")
        
        indicators['adosc'] = ta.ADOSC(data['high'], data['low'], data['close'], data['volume'], fastperiod=3, slowperiod=10)
        logging.debug("Computed ADOSC.")
        
        indicators['obv'] = ta.OBV(data['close'], data['volume'])
        logging.debug("Computed OBV.")
        
        indicators['volume_osc'] = (data['volume'] - data['volume'].rolling(window=20).mean()) / data['volume'].rolling(window=20).mean()
        logging.debug("Computed Volume Oscillator.")
        
        indicators['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        logging.debug("Computed VWAP.")
        
        indicators['pvi'] = (data['volume'].diff().apply(lambda x: 1 if x > 0 else 0).cumsum()) * data['close']
        logging.debug("Computed PVI.")
        
        indicators['nvi'] = (data['volume'].diff().apply(lambda x: 1 if x < 0 else 0).cumsum()) * data['close']
        logging.debug("Computed NVI.")
        
        indicators['atr'] = ta.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        logging.debug("Computed ATR.")
        
        indicators['natr'] = ta.NATR(data['high'], data['low'], data['close'], timeperiod=14)
        logging.debug("Computed NATR.")
        
        indicators['trange'] = ta.TRANGE(data['high'], data['low'], data['close'])
        logging.debug("Computed True Range.")
        
        indicators['ht_dcperiod'] = ta.HT_DCPERIOD(data['close'])
        logging.debug("Computed HT_DCPERIOD.")
        
        indicators['ht_dcpphase'] = ta.HT_DCPHASE(data['close'])
        logging.debug("Computed HT_DCPHASE.")
        
        indicators['ht_phasor_inphase'], indicators['ht_phasor_quadrature'] = ta.HT_PHASOR(data['close'])
        logging.debug("Computed HT_PHASOR InPhase and Quadrature.")
        
        indicators['ht_sine_sine'], indicators['ht_sine_leadsine'] = ta.HT_SINE(data['close'])
        logging.debug("Computed HT_SINE Sine and Leadsine.")
        
        indicators['ht_trendmode'] = ta.HT_TRENDMODE(data['close'])
        logging.debug("Computed HT_TRENDMODE.")
        
        indicators['beta'] = ta.BETA(data['high'], data['low'], timeperiod=5)
        logging.debug("Computed BETA.")
        
        indicators['correl'] = ta.CORREL(data['high'], data['low'], timeperiod=30)
        logging.debug("Computed CORREL.")
        
        indicators['linearreg'] = ta.LINEARREG(data['close'], timeperiod=14)
        indicators['linearreg_angle'] = ta.LINEARREG_ANGLE(data['close'], timeperiod=14)
        indicators['linearreg_intercept'] = ta.LINEARREG_INTERCEPT(data['close'], timeperiod=14)
        indicators['linearreg_slope'] = ta.LINEARREG_SLOPE(data['close'], timeperiod=14)
        logging.debug("Computed Linear Regression indicators.")
        
        indicators['stddev'] = ta.STDDEV(data['close'], timeperiod=5, nbdev=1)
        logging.debug("Computed STDDEV.")
        
        indicators['tsf'] = ta.TSF(data['close'], timeperiod=14)
        logging.debug("Computed TSF.")
        
        indicators['var'] = ta.VAR(data['close'], timeperiod=5, nbdev=1)
        logging.debug("Computed VAR.")
        
        try:
            ao = pta.ao(data['high'], data['low'])
            indicators['ao'] = ao
            logging.debug("Computed AO using pandas_ta.")
        except Exception as e:
            logging.warning(f"Failed to compute AO using pandas_ta: {e}")
        
        try:
            fi = pta.fi(data['close'], data['volume'])
            indicators['fi'] = fi
            logging.debug("Computed FI using pandas_ta.")
        except Exception as e:
            logging.warning(f"Failed to compute FI using pandas_ta: {e}")
            try:
                data['fi'] = (data['close'] - data['close'].shift(1)) * data['volume']
                indicators['fi'] = data['fi']
                logging.debug("Computed FI using fallback method.")
            except Exception as ex:
                logging.error(f"Failed to compute FI using fallback method: {ex}")
        
        try:
            ichimoku = pta.ichimoku(data['high'], data['low'], append=False)
            for col in ['ISA_9', 'ISB_26', 'ITS_9', 'IKS_26']:
                if col not in ichimoku.columns:
                    raise KeyError(f"Missing column '{col}' in Ichimoku data.")
            indicators['ichimoku_conversion'] = ichimoku['ISA_9']
            indicators['ichimoku_base'] = ichimoku['ISB_26']
            indicators['ichimoku_span_a'] = ichimoku['ITS_9']
            indicators['ichimoku_span_b'] = ichimoku['IKS_26']
            logging.debug("Computed Ichimoku Cloud indicators.")
        except Exception as e:
            logging.warning(f"Failed to compute Ichimoku Cloud indicators: {e}")
        
        try:
            kc = pta.kc(data['high'], data['low'], data['close'], append=False)
            for col in ['KCU_20_2.0', 'KCM_20_2.0', 'KCL_20_2.0']:
                if col not in kc.columns:
                    raise KeyError(f"Missing column '{col}' in Keltner Channels data.")
            indicators['kc_upper'] = kc['KCU_20_2.0']
            indicators['kc_middle'] = kc['KCM_20_2.0']
            indicators['kc_lower'] = kc['KCL_20_2.0']
            logging.debug("Computed Keltner Channels indicators.")
        except Exception as e:
            logging.warning(f"Failed to compute Keltner Channels indicators: {e}")
        
        try:
            mfi = pta.mfi(data['high'], data['low'], data['close'], data['volume'])
            indicators['mfi'] = mfi
            logging.debug("Computed MFI using pandas_ta.")
        except Exception as e:
            logging.warning(f"Failed to compute MFI using pandas_ta: {e}")
        
        try:
            rvi = pta.rvi(data['close'])
            indicators['rvi'] = rvi
            logging.debug("Computed RVI using pandas_ta.")
        except Exception as e:
            logging.warning(f"Failed to compute RVI using pandas_ta: {e}")
        
        try:
            stochrsi = pta.stochrsi(data['close'], append=False)
            for col in ['STOCHRSI_14_5_3_SLOWK', 'STOCHRSI_14_5_3_SLOWD']:
                if col not in stochrsi.columns:
                    raise KeyError(f"Missing column '{col}' in Stochastic RSI data.")
            indicators['stochrsi_slowk'] = stochrsi['STOCHRSI_14_5_3_SLOWK']
            indicators['stochrsi_slowd'] = stochrsi['STOCHRSI_14_5_3_SLOWD']
            logging.debug("Computed Stochastic RSI indicators.")
        except Exception as e:
            logging.warning(f"Failed to compute Stochastic RSI indicators: {e}")
        
        try:
            tsi = pta.tsi(data['close'])
            for col in tsi.columns:
                indicators[col] = tsi[col]
            logging.debug("Computed TSI using pandas_ta.")
        except Exception as e:
            logging.warning(f"Failed to compute TSI using pandas_ta: {e}")
        
        try:
            vortex = pta.vortex(data['high'], data['low'], data['close'], append=False)
            for col in ['VI+_14', 'VI-_14']:
                if col not in vortex.columns:
                    raise KeyError(f"Missing column '{col}' in Vortex Indicator data.")
            indicators['vi_plus'] = vortex['VI+_14']
            indicators['vi_minus'] = vortex['VI-_14']
            logging.debug("Computed Vortex Indicator.")
        except Exception as e:
            logging.warning(f"Failed to compute Vortex Indicator: {e}")
        
        data = compute_obv_price_divergence(data)
        logging.debug("Computed OBV Price Divergence.")
        
        for k, v in indicators.items():
            data[k] = v
        logging.info("Added all indicators to the DataFrame.")
        
        data.dropna(inplace=True)
        logging.info("Dropped rows with NaN values.")
        
        return data
    except Exception as e:
        logging.error(f"Error in compute_all_indicators: {e}")
        return data

if __name__ == "__main__":
    data = pd.DataFrame({
        'close': np.random.rand(100),
        'volume': np.random.rand(100),
        'open': np.random.rand(100),
        'high': np.random.rand(100),
        'low': np.random.rand(100)
    })
    
    data = compute_all_indicators(data)
    
    print(data.head())