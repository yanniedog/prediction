import pandas as pd
import numpy as np
import talib as ta
import pandas_ta as pta
import sys

data = pd.DataFrame({
    'open': np.random.uniform(100, 200, 100),
    'high': np.random.uniform(200, 300, 100),
    'low': np.random.uniform(50, 100, 100),
    'close': np.random.uniform(100, 200, 100),
    'volume': np.random.uniform(1000, 5000, 100)
})
try:
    data['rsi_14'] = ta.RSI(data['close'], timeperiod=14)
except AttributeError:
    print("TA-Lib RSI function not found.")
    sys.exit(1)
try:
    data['ao_5_34'] = pta.ao(data['high'], data['low'], length1=5, length2=34)
except Exception as e:
    print(f"Error computing AO: {e}")
    sys.exit(1)
missing = [col for col in ['rsi_14', 'ao_5_34'] if col not in data.columns]
if missing:
    print(f"Missing indicators: {missing}")
    sys.exit(1)
if data[['rsi_14', 'ao_5_34']].isnull().all().any():
    print("Indicators contain only NaN values.")
    sys.exit(1)
print("All indicators computed successfully.")
