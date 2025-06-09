"""Patch for pandas_ta package to fix NaN import issue."""
import sys
import numpy as np

# Add NaN to numpy's __all__ and module dict
if not hasattr(np, 'NaN'):
    np.NaN = np.nan
    if hasattr(np, '__all__'):
        np.__all__.append('NaN')
    else:
        np.__all__ = list(getattr(np, '__all__', [])) + ['NaN']

# Monkey patch numpy's __getattr__ to handle NaN
original_getattr = np.__getattr__ if hasattr(np, '__getattr__') else None

def patched_getattr(name):
    if name == 'NaN':
        return np.nan
    if original_getattr is not None:
        return original_getattr(name)
    raise AttributeError(f"module 'numpy' has no attribute '{name}'")

np.__getattr__ = patched_getattr 