"""Patch module to fix pandas_ta NaN import issues."""

import pandas as pd
import numpy as np
import warnings
from typing import Any, Dict, List, Optional, Union

# Suppress deprecation warnings for pkg_resources
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")

# Fix numpy NaN import issue for pandas_ta
if not hasattr(np, 'NaN'):
    np.NaN = np.nan

# Patch pandas_ta NaN handling
def patch_pandas_ta():
    """Apply patches to pandas_ta to fix NaN handling issues."""
    try:
        # Suppress warnings during import
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pandas_ta as ta
        
        # Store original functions
        original_indicators = {}
        
        def safe_indicator(func):
            """Wrapper to safely handle NaN values in indicators."""
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    if isinstance(result, pd.Series):
                        # Replace inf with NaN
                        result = result.replace([np.inf, -np.inf], np.nan)
                        # Forward fill NaN values
                        result = result.fillna(method='ffill')
                        # Backward fill any remaining NaN values
                        result = result.fillna(method='bfill')
                    return result
                except Exception as e:
                    print(f"Warning: Indicator calculation failed: {e}")
                    return pd.Series(np.nan, index=args[0].index)
            return wrapper
            
        # Patch all indicator functions
        for name in dir(ta):
            if name.startswith('_') or name in ['pd', 'np', 'pd_ta']:
                continue
            try:
                func = getattr(ta, name)
                if callable(func):
                    original_indicators[name] = func
                    setattr(ta, name, safe_indicator(func))
            except Exception:
                continue
                
        return original_indicators
    except ImportError:
        print("Warning: pandas_ta not available, skipping patch")
        return {}

# Apply patches
_original_indicators = patch_pandas_ta()

def restore_original_indicators():
    """Restore original pandas_ta indicator functions."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import pandas_ta as ta
        for name, func in _original_indicators.items():
            setattr(ta, name, func)
    except ImportError:
        pass

# Register cleanup
import atexit
atexit.register(restore_original_indicators) 