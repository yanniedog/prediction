import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, List, Any, Generator
from unittest.mock import patch, MagicMock
import custom_indicators as ci
import utils
import config
import logging
from contextlib import contextmanager
from custom_indicators import (
    compute_obv_price_divergence, compute_volume_oscillator,
    IndicatorError, MissingColumnsError, InvalidParameterError, UnsupportedMethodError,
    _check_required_cols
)

@contextmanager
def suppress_expected_errors():
    """Temporarily suppress expected error messages during tests."""
    logger = logging.getLogger('custom_indicators')
    original_level = logger.level
    logger.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        logger.setLevel(original_level)

@pytest.fixture(scope="function")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Create a sample DataFrame with OHLCV data."""
    dates = pd.date_range(start='2020-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 102,
        'low': np.random.randn(100).cumsum() + 98,
        'close': np.random.randn(100).cumsum() + 101,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)
    data['high'] = data[['open', 'close']].max(axis=1) + 1
    data['low'] = data[['open', 'close']].min(axis=1) - 1
    return data

@pytest.fixture
def sample_data_with_extremes() -> pd.DataFrame:
    """Create sample data with extreme values for edge case testing."""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'open': [100] * 50,
        'high': [102] * 50,
        'low': [98] * 50,
        'close': [101] * 50,
        'volume': [1000] * 50
    }, index=dates)
    
    # Add some extreme values
    data.loc[dates[10], 'volume'] = 0  # Zero volume
    data.loc[dates[20], 'volume'] = np.inf  # Infinite volume
    data.loc[dates[30], 'close'] = 0  # Zero price
    data.loc[dates[40], 'close'] = np.inf  # Infinite price
    
    return data

def test_check_required_cols():
    """Test the _check_required_cols helper function."""
    data = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})

    # Test with all columns present
    _check_required_cols(data, ['col1', 'col2'], 'test_indicator')  # Should not raise

    # Test with missing columns
    with pytest.raises(MissingColumnsError, match="Missing required columns for test_indicator: \\['col3'\\]"):
        _check_required_cols(data, ['col1', 'col3'], 'test_indicator')

    # Test with empty DataFrame
    empty_df = pd.DataFrame()
    with pytest.raises(MissingColumnsError, match="Missing required columns for test_indicator: \\['col1'\\]"):
        _check_required_cols(empty_df, ['col1'], 'test_indicator')

def test_compute_obv_price_divergence(sample_data):
    """Test OBV/Price divergence calculation with different methods and invalid inputs."""
    # Test valid methods
    with suppress_expected_errors():
        result = compute_obv_price_divergence(sample_data, method="Difference")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'obv_price_divergence' in result.columns

        result = compute_obv_price_divergence(sample_data, method="Ratio")
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'obv_price_divergence' in result.columns

        # Test invalid method
        with pytest.raises(UnsupportedMethodError, match="Unsupported divergence method: Log Ratio"):
            compute_obv_price_divergence(sample_data, method="Log Ratio")

        # Test invalid price input type
        with pytest.raises(UnsupportedMethodError, match="Unsupported price input type: invalid"):
            compute_obv_price_divergence(sample_data, price_input_type="invalid")

        # Test invalid OBV method
        with pytest.raises(UnsupportedMethodError, match="Unsupported obv_method: invalid"):
            compute_obv_price_divergence(sample_data, obv_method="invalid")

        # Test invalid price method
        with pytest.raises(UnsupportedMethodError, match="Unsupported price_method: invalid"):
            compute_obv_price_divergence(sample_data, price_method="invalid")

        # Test missing columns
        with pytest.raises(MissingColumnsError, match="Missing required columns for OBV/Price Divergence"):
            compute_obv_price_divergence(sample_data.drop('volume', axis=1))

        # Test invalid periods
        with pytest.raises(InvalidParameterError, match="Invalid obv_period: 0"):
            compute_obv_price_divergence(sample_data, obv_period=0)

        with pytest.raises(InvalidParameterError, match="Invalid price_period: -1"):
            compute_obv_price_divergence(sample_data, price_period=-1)

        # Test invalid smoothing
        with pytest.raises(InvalidParameterError, match="Invalid smoothing: -0.1"):
            compute_obv_price_divergence(sample_data, smoothing=-0.1)

def test_compute_volume_oscillator(sample_data):
    """Test Volume Oscillator calculation with valid inputs, invalid inputs, and edge cases."""
    with suppress_expected_errors():
        # Test valid window
        result = compute_volume_oscillator(sample_data, window=20)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'volume_osc' in result.columns

        # Test invalid window
        with pytest.raises(InvalidParameterError, match="Window \\(1\\) must be an integer >= 2"):
            compute_volume_oscillator(sample_data, window=1)

        # Test missing volume column
        with pytest.raises(MissingColumnsError, match="Missing required columns for Volume Oscillator"):
            compute_volume_oscillator(sample_data.drop('volume', axis=1))

        # Test with zero volumes
        data_zero_vol = sample_data.copy()
        data_zero_vol.loc[data_zero_vol.index[0:10], 'volume'] = 0
        result = compute_volume_oscillator(data_zero_vol)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'volume_osc' in result.columns
        assert result['volume_osc'].isna().any()  # Should have NaN for zero volumes

        # Test with very small window
        result = compute_volume_oscillator(sample_data, window=2)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'volume_osc' in result.columns

        # Test with window larger than data
        result = compute_volume_oscillator(sample_data, window=200)
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        assert 'volume_osc' in result.columns
        assert result['volume_osc'].isna().any()  # Should have NaN at start

def test_compute_vwap_basic(sample_data: pd.DataFrame) -> None:
    """Test basic VWAP calculation."""
    result = ci.compute_vwap(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert ci.VWAP in result.columns
    assert not result[ci.VWAP].isna().all()
    assert len(result) == len(sample_data)
    
    # Verify VWAP is between cumulative min(low) and cumulative max(high) up to each row
    cum_min_low = sample_data['low'].expanding().min()
    cum_max_high = sample_data['high'].expanding().max()
    vwap = result[ci.VWAP]
    assert (vwap >= cum_min_low).all()
    assert (vwap <= cum_max_high).all()

def test_compute_vwap_invalid_inputs(sample_data: pd.DataFrame) -> None:
    """Test VWAP with invalid inputs."""
    # Test with missing columns
    data_missing_close = sample_data.drop('close', axis=1)
    result = ci.compute_vwap(data_missing_close)
    assert result is None
    
    data_missing_volume = sample_data.drop('volume', axis=1)
    result = ci.compute_vwap(data_missing_volume)
    assert result is None

def test_compute_vwap_edge_cases(sample_data_with_extremes: pd.DataFrame) -> None:
    """Test VWAP with edge cases."""
    result = ci.compute_vwap(sample_data_with_extremes)
    
    assert isinstance(result, pd.DataFrame)
    assert ci.VWAP in result.columns
    # Verify that extreme values are handled
    assert not result[ci.VWAP].isin([np.inf, -np.inf]).any()
    # Verify zero volume handling
    zero_volume_idx = sample_data_with_extremes['volume'] == 0
    assert result.loc[zero_volume_idx, ci.VWAP].isna().all()

def test_compute_pvi_basic(sample_data: pd.DataFrame) -> None:
    """Test basic PVI calculation."""
    result = ci.compute_pvi(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert ci.PVI in result.columns
    assert not result[ci.PVI].isna().all()
    assert len(result) == len(sample_data)
    # Verify PVI starts at 1000
    assert result[ci.PVI].iloc[0] == 1000.0

def test_compute_nvi_basic(sample_data: pd.DataFrame) -> None:
    """Test basic NVI calculation."""
    result = ci.compute_nvi(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert ci.NVI in result.columns
    assert not result[ci.NVI].isna().all()
    assert len(result) == len(sample_data)
    # Verify NVI starts at 1000
    assert result[ci.NVI].iloc[0] == 1000.0

def test_compute_volume_indices_invalid_inputs(sample_data: pd.DataFrame) -> None:
    """Test PVI and NVI with invalid inputs."""
    # Test with missing columns
    data_missing_close = sample_data.drop('close', axis=1)
    assert ci.compute_pvi(data_missing_close) is None
    assert ci.compute_nvi(data_missing_close) is None
    
    data_missing_volume = sample_data.drop('volume', axis=1)
    assert ci.compute_pvi(data_missing_volume) is None
    assert ci.compute_nvi(data_missing_volume) is None

def test_compute_volume_indices_edge_cases(sample_data_with_extremes: pd.DataFrame) -> None:
    """Test PVI and NVI with edge cases."""
    # Test PVI
    pvi_result = ci.compute_pvi(sample_data_with_extremes)
    assert isinstance(pvi_result, pd.DataFrame)
    assert ci.PVI in pvi_result.columns
    assert not pvi_result[ci.PVI].isna().all()
    assert not pvi_result[ci.PVI].isin([np.inf, -np.inf]).any()
    
    # Test NVI
    nvi_result = ci.compute_nvi(sample_data_with_extremes)
    assert isinstance(nvi_result, pd.DataFrame)
    assert ci.NVI in nvi_result.columns
    assert not nvi_result[ci.NVI].isna().all()
    assert not nvi_result[ci.NVI].isin([np.inf, -np.inf]).any()

def test_compute_volume_indices_extreme_price_changes() -> None:
    """Test PVI and NVI with extreme price changes."""
    # Create data with extreme price changes
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'close': [100] * 10,
        'volume': [1000] * 10
    }, index=dates)
    
    # Add extreme price changes
    data.loc[dates[5], 'close'] = 1000  # 900% increase
    data.loc[dates[6], 'close'] = 10    # 99% decrease
    
    # Test PVI
    pvi_result = ci.compute_pvi(data)
    assert isinstance(pvi_result, pd.DataFrame)
    assert ci.PVI in pvi_result.columns
    # Verify that extreme changes are handled (previous value used)
    assert pvi_result[ci.PVI].iloc[5] == pvi_result[ci.PVI].iloc[4]
    assert pvi_result[ci.PVI].iloc[6] == pvi_result[ci.PVI].iloc[5]
    
    # Test NVI
    nvi_result = ci.compute_nvi(data)
    assert isinstance(nvi_result, pd.DataFrame)
    assert ci.NVI in nvi_result.columns
    # Verify that extreme changes are handled (previous value used)
    assert nvi_result[ci.NVI].iloc[5] == nvi_result[ci.NVI].iloc[4]
    assert nvi_result[ci.NVI].iloc[6] == nvi_result[ci.NVI].iloc[5]

def test_compute_volume_indices_sequential_updates() -> None:
    """Test that PVI and NVI update correctly based on volume conditions."""
    # Create data with alternating volume changes
    dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
    data = pd.DataFrame({
        'close': [100 + i for i in range(10)],  # Increasing prices
        'volume': [1000 + (100 if i % 2 == 0 else -100) for i in range(10)]  # Alternating volume changes
    }, index=dates)
    
    # Test PVI (should update on positive volume changes)
    pvi_result = ci.compute_pvi(data)
    assert isinstance(pvi_result, pd.DataFrame)
    assert ci.PVI in pvi_result.columns
    # Verify PVI changes only on even indices (positive volume changes)
    for i in range(1, len(data)):
        if i % 2 == 0:  # Even index (positive volume change)
            assert pvi_result[ci.PVI].iloc[i] != pvi_result[ci.PVI].iloc[i-1]
        else:  # Odd index (negative volume change)
            assert pvi_result[ci.PVI].iloc[i] == pvi_result[ci.PVI].iloc[i-1]
    
    # Test NVI (should update on negative volume changes)
    nvi_result = ci.compute_nvi(data)
    assert isinstance(nvi_result, pd.DataFrame)
    assert ci.NVI in nvi_result.columns
    # Verify NVI changes only on odd indices (negative volume changes)
    for i in range(1, len(data)):
        if i % 2 == 1:  # Odd index (negative volume change)
            assert nvi_result[ci.NVI].iloc[i] != nvi_result[ci.NVI].iloc[i-1]
        else:  # Even index (positive volume change)
            assert nvi_result[ci.NVI].iloc[i] == nvi_result[ci.NVI].iloc[i-1]

def test_custom_indicators_integration(sample_data: pd.DataFrame) -> None:
    """Test integration of multiple custom indicators."""
    # Calculate multiple indicators
    obv_div = ci.compute_obv_price_divergence(sample_data)
    vol_osc = ci.compute_volume_oscillator(sample_data)
    vwap = ci.compute_vwap(sample_data)
    pvi = ci.compute_pvi(sample_data)
    nvi = ci.compute_nvi(sample_data)
    # Verify all indicators have same length
    for ind in [obv_div, vol_osc, vwap, pvi, nvi]:
        assert isinstance(ind, pd.DataFrame)
        assert len(ind) == len(sample_data)

def test_custom_indicators_performance(sample_data: pd.DataFrame) -> None:
    """Test performance of custom indicators with larger dataset."""
    import time
    # Create a larger dataset
    large_data = pd.concat([sample_data]*10)
    start_time = time.time()
    obv_div = ci.compute_obv_price_divergence(large_data)
    vol_osc = ci.compute_volume_oscillator(large_data)
    vwap = ci.compute_vwap(large_data)
    pvi = ci.compute_pvi(large_data)
    nvi = ci.compute_nvi(large_data)
    end_time = time.time()
    # All should return DataFrames of correct length
    for ind in [obv_div, vol_osc, vwap, pvi, nvi]:
        assert isinstance(ind, pd.DataFrame)
        assert len(ind) == len(large_data)
    # Should run in reasonable time (<2s for 1000 rows)
    assert (end_time - start_time) < 2 