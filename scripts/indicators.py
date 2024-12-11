# indicators.py

import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def compute_obv_price_divergence(
    data: pd.DataFrame, 
    method="Difference", 
    obv_method="SMA", 
    obv_period=14, 
    price_input_type="OHLC/4", 
    price_method="SMA", 
    price_period=14, 
    bearish_threshold=-0.8, 
    bullish_threshold=0.8, 
    smoothing=0.01
) -> pd.DataFrame:
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
        price_input_type = price_input_type.lower()
        if price_input_type == "close":
            selected_price = data['close']
        elif price_input_type == "open":
            selected_price = data['open']
        elif price_input_type == "high":
            selected_price = data['high']
        elif price_input_type == "low":
            selected_price = data['low']
        elif price_input_type == "hl/2":
            selected_price = (data['high'] + data['low']) / 2
        else:
            selected_price = (data['open'] + data['high'] + data['low'] + data['close']) / 4
        logging.debug(f"Selected price type '{price_input_type}' for divergence computation.")
        
        obv = ta.OBV(data['close'], data['volume'])
        logging.debug("Computed On-Balance Volume (OBV).")
        
        if obv_method == "sma":
            obv_ma = ta.SMA(obv, timeperiod=obv_period)
            logging.debug(f"Applied SMA smoothing to OBV with period {obv_period}.")
        elif obv_method == "ema":
            obv_ma = ta.EMA(obv, timeperiod=obv_period)
            logging.debug(f"Applied EMA smoothing to OBV with period {obv_period}.")
        else:
            obv_ma = obv
            logging.debug("No smoothing applied to OBV.")
        
        price_method = price_method.lower()
        if price_method == "sma":
            price_ma = ta.SMA(selected_price, timeperiod=price_period)
            logging.debug(f"Applied SMA smoothing to price with period {price_period}.")
        elif price_method == "ema":
            price_ma = ta.EMA(selected_price, timeperiod=price_period)
            logging.debug(f"Applied EMA smoothing to price with period {price_period}.")
        else:
            price_ma = selected_price
            logging.debug("No smoothing applied to price.")
        
        obv_change_percent = (obv_ma - obv_ma.shift(1)) / obv_ma.shift(1) * 100
        price_change_percent = (price_ma - price_ma.shift(1)) / price_ma.shift(1) * 100
        logging.debug("Calculated percentage changes for OBV and Price.")
        
        method = method.lower()
        if method == "difference":
            metric = obv_change_percent - price_change_percent
            logging.debug("Computed divergence using 'Difference' method.")
        elif method == "ratio":
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
        indicators['sma'] = ta.SMA(data['close'], timeperiod=30)
        logging.debug("Computed SMA.")
        
        indicators['rsi'] = ta.RSI(data['close'], timeperiod=14)
        logging.debug("Computed RSI.")
        
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = ta.MACD(
            data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        logging.debug("Computed MACD.")
        
        indicators['bbands_upper'], indicators['bbands_middle'], indicators['bbands_lower'] = ta.BBANDS(
            data['close'], timeperiod=5, nbdevup=2, nbdevdn=2, matype=0)
        logging.debug("Computed Bollinger Bands.")
        
        indicators['ema'] = ta.EMA(data['close'], timeperiod=30)
        logging.debug("Computed EMA.")
        
        indicators['obv'] = ta.OBV(data['close'], data['volume'])
        logging.debug("Computed OBV.")
        
        try:
            indicators['fi'] = pta.fi(data['close'], data['volume'], append=False)
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
            ichimoku = pta.ichimoku(data['high'], data['low'], data['close'], append=False)
            if isinstance(ichimoku, tuple):
                ichimoku = ichimoku[0]  # Access the DataFrame if a tuple is returned
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
            if isinstance(kc, tuple):
                kc = kc[0]  # Access the DataFrame if a tuple is returned
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
            vortex = pta.vortex(data['high'], data['low'], data['close'], append=False)
            if isinstance(vortex, tuple):
                vortex = vortex[0]  # Access the DataFrame if a tuple is returned
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
    np.random.seed(42)
    sample_size = 100
    data = pd.DataFrame({
        'open': np.random.uniform(100, 200, sample_size),
        'high': np.random.uniform(100, 200, sample_size),
        'low': np.random.uniform(90, 190, sample_size),
        'close': np.random.uniform(100, 200, sample_size),
        'volume': np.random.uniform(1000, 5000, sample_size)
    })
    
    data_with_indicators = compute_all_indicators(data)
    
    print(data_with_indicators.head())
