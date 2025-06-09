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
    _check_required_cols,
    OBV_PRICE_DIVERGENCE,
    VOLUME_OSCILLATOR,
    VWAP,
    PVI,
    NVI
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
    dates = pd.date_range(start='2024-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.uniform(100, 200, 100),
        'high': np.random.uniform(200, 300, 100),
        'low': np.random.uniform(50, 100, 100),
        'close': np.random.uniform(100, 200, 100),
        'volume': np.random.uniform(1000, 5000, 100)
    })
    return data

@pytest.fixture
def sample_data_with_extremes() -> pd.DataFrame:
    """Create sample data with extreme values for edge case testing."""
    dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
    data = pd.DataFrame({
        'open': [100.0] * 50,
        'high': [102.0] * 50,
        'low': [98.0] * 50,
        'close': [101.0] * 50,  # float for close
        'volume': [1000.0] * 50  # float for volume
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
        with pytest.raises(UnsupportedMethodError, match="Unsupported divergence method: InvalidMethod"):
            compute_obv_price_divergence(sample_data, method="InvalidMethod")

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
        # No need to check for NaN, as min_periods=1 ensures all values are filled

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
    with pytest.raises(MissingColumnsError, match="Missing required columns for VWAP: \\['close'\\]"):
        ci.compute_vwap(data_missing_close)

    data_missing_volume = sample_data.drop('volume', axis=1)
    with pytest.raises(MissingColumnsError, match="Missing required columns for VWAP: \\['volume'\\]"):
        ci.compute_vwap(data_missing_volume)

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
    # Test with missing columns for PVI
    data_missing_close = sample_data.drop('close', axis=1)
    with pytest.raises(MissingColumnsError, match="Missing required columns for PVI: \\['close'\\]"):
        ci.compute_pvi(data_missing_close)

    data_missing_volume = sample_data.drop('volume', axis=1)
    with pytest.raises(MissingColumnsError, match="Missing required columns for PVI: \\['volume'\\]"):
        ci.compute_pvi(data_missing_volume)

    # Test with missing columns for NVI
    with pytest.raises(MissingColumnsError, match="Missing required columns for NVI: \\['close'\\]"):
        ci.compute_nvi(data_missing_close)

    with pytest.raises(MissingColumnsError, match="Missing required columns for NVI: \\['volume'\\]"):
        ci.compute_nvi(data_missing_volume)

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

def test_compute_obv_price_divergence_valid(sample_data):
    """Test OBV/Price divergence calculation with valid data."""
    result = compute_obv_price_divergence(
        data=sample_data,
        method="Difference",
        obv_method="SMA",
        obv_period=14,
        price_input_type="close",
        price_method="SMA",
        price_period=14
    )
    
    assert isinstance(result, pd.DataFrame)
    assert OBV_PRICE_DIVERGENCE in result.columns
    assert len(result) == len(sample_data)
    assert not result[OBV_PRICE_DIVERGENCE].isna().all()

def test_compute_obv_price_divergence_missing_columns():
    """Test OBV/Price divergence with missing columns."""
    data = pd.DataFrame({'close': [1, 2, 3]})  # Missing required columns
    with pytest.raises(MissingColumnsError):
        compute_obv_price_divergence(data)

def test_compute_obv_price_divergence_invalid_period(sample_data):
    """Test OBV/Price divergence with invalid period."""
    with pytest.raises(InvalidParameterError):
        compute_obv_price_divergence(
            data=sample_data,
            obv_period=0  # Invalid period
        )

def test_compute_obv_price_divergence_invalid_method(sample_data):
    """Test OBV/Price divergence with invalid method."""
    with pytest.raises(UnsupportedMethodError):
        compute_obv_price_divergence(
            data=sample_data,
            method="InvalidMethod"
        )

def test_compute_obv_price_divergence_all_methods(sample_data):
    """Test OBV/Price divergence with all supported methods."""
    methods = ["Difference", "Ratio", "Log Ratio"]
    obv_methods = ["SMA", "EMA", "NONE"]
    price_methods = ["SMA", "EMA", "NONE"]
    price_inputs = ["close", "open", "high", "low", "hl/2", "ohlc/4"]
    
    for method in methods:
        for obv_method in obv_methods:
            for price_method in price_methods:
                for price_input in price_inputs:
                    result = compute_obv_price_divergence(
                        data=sample_data,
                        method=method,
                        obv_method=obv_method,
                        price_input_type=price_input,
                        price_method=price_method
                    )
                    assert isinstance(result, pd.DataFrame)
                    assert OBV_PRICE_DIVERGENCE in result.columns

def test_compute_volume_oscillator_valid(sample_data):
    """Test Volume Oscillator calculation with valid data."""
    result = compute_volume_oscillator(sample_data, window=20)
    
    assert isinstance(result, pd.DataFrame)
    assert VOLUME_OSCILLATOR in result.columns
    assert len(result) == len(sample_data)
    assert not result[VOLUME_OSCILLATOR].isna().all()

def test_compute_volume_oscillator_missing_volume():
    """Test Volume Oscillator with missing volume column."""
    data = pd.DataFrame({'close': [1, 2, 3]})
    with pytest.raises(MissingColumnsError):
        compute_volume_oscillator(data)

def test_compute_volume_oscillator_invalid_window(sample_data):
    """Test Volume Oscillator with invalid window."""
    with pytest.raises(InvalidParameterError):
        compute_volume_oscillator(sample_data, window=1)

def test_compute_volume_oscillator_zero_volume():
    """Test Volume Oscillator with zero volume."""
    data = pd.DataFrame({
        'volume': [0, 0, 0],
        'close': [1, 2, 3]
    })
    result = compute_volume_oscillator(data, window=2)
    assert result[VOLUME_OSCILLATOR].isna().all()

def test_compute_vwap_valid(sample_data):
    """Test VWAP calculation with valid data."""
    result = ci.compute_vwap(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert VWAP in result.columns
    assert len(result) == len(sample_data)
    assert not result[VWAP].isna().all()

def test_compute_vwap_missing_columns():
    """Test VWAP with missing columns."""
    data = pd.DataFrame({'close': [1, 2, 3]})
    with pytest.raises(MissingColumnsError):
        ci.compute_vwap(data)

def test_compute_vwap_zero_volume():
    """Test VWAP with zero volume."""
    data = pd.DataFrame({
        'open': [1, 2, 3],
        'high': [2, 3, 4],
        'low': [0.5, 1.5, 2.5],
        'close': [1.5, 2.5, 3.5],
        'volume': [0, 0, 0]
    })
    result = ci.compute_vwap(data)
    assert result[VWAP].isna().all()

def test_compute_pvi_valid(sample_data):
    """Test PVI calculation with valid data."""
    result = ci.compute_pvi(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert PVI in result.columns
    assert len(result) == len(sample_data)
    assert not result[PVI].isna().all()

def test_compute_pvi_missing_columns():
    """Test PVI with missing columns."""
    data = pd.DataFrame({'close': [1, 2, 3]})
    with pytest.raises(MissingColumnsError):
        ci.compute_pvi(data)

def test_compute_pvi_zero_volume():
    """Test PVI with zero volume."""
    data = pd.DataFrame({
        'close': [1, 2, 3],
        'volume': [0, 0, 0]
    })
    result = ci.compute_pvi(data)
    assert result[PVI].isna().all()

def test_compute_nvi_valid(sample_data):
    """Test NVI calculation with valid data."""
    result = ci.compute_nvi(sample_data)
    
    assert isinstance(result, pd.DataFrame)
    assert NVI in result.columns
    assert len(result) == len(sample_data)
    assert not result[NVI].isna().all()

def test_compute_nvi_missing_columns():
    """Test NVI with missing columns."""
    data = pd.DataFrame({'close': [1, 2, 3]})
    with pytest.raises(MissingColumnsError):
        ci.compute_nvi(data)

def test_compute_nvi_zero_volume():
    """Test NVI with zero volume."""
    data = pd.DataFrame({
        'close': [1, 2, 3],
        'volume': [0, 0, 0]
    })
    result = ci.compute_nvi(data)
    assert result[NVI].isna().all()

def test_compute_returns_valid(sample_data):
    """Test returns calculation with valid data."""
    result = ci.compute_returns(sample_data, period=1)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert not result.isna().all()

def test_compute_returns_missing_close():
    """Test returns calculation with missing close column."""
    data = pd.DataFrame({'volume': [1, 2, 3]})
    with pytest.raises(MissingColumnsError):
        ci.compute_returns(data)

def test_compute_returns_invalid_period(sample_data):
    """Test returns calculation with invalid period."""
    with pytest.raises(InvalidParameterError):
        ci.compute_returns(sample_data, period=0)

def test_compute_returns_different_periods(sample_data):
    """Test returns calculation with different periods."""
    periods = [1, 2, 5, 10]
    for period in periods:
        result = ci.compute_returns(sample_data, period=period)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

def test_compute_volatility_valid(sample_data):
    """Test volatility calculation with valid data."""
    result = ci.compute_volatility(sample_data, period=20)
    
    assert isinstance(result, pd.Series)
    assert len(result) == len(sample_data)
    assert not result.isna().all()

def test_compute_volatility_missing_close():
    """Test volatility calculation with missing close column."""
    data = pd.DataFrame({'volume': [1, 2, 3]})
    with pytest.raises(MissingColumnsError):
        ci.compute_volatility(data)

def test_compute_volatility_invalid_period(sample_data):
    """Test volatility calculation with invalid period."""
    with pytest.raises(InvalidParameterError):
        ci.compute_volatility(sample_data, period=0)

def test_compute_volatility_different_periods(sample_data):
    """Test volatility calculation with different periods."""
    periods = [5, 10, 20, 50]
    for period in periods:
        result = ci.compute_volatility(sample_data, period=period)
        assert isinstance(result, pd.Series)
        assert len(result) == len(sample_data)

def test_all_indicators_empty_dataframe():
    """Test all indicators with empty DataFrame."""
    empty_df = pd.DataFrame()
    
    with pytest.raises(MissingColumnsError):
        compute_obv_price_divergence(empty_df)
    with pytest.raises(MissingColumnsError):
        compute_volume_oscillator(empty_df)
    with pytest.raises(MissingColumnsError):
        ci.compute_vwap(empty_df)
    with pytest.raises(MissingColumnsError):
        ci.compute_pvi(empty_df)
    with pytest.raises(MissingColumnsError):
        ci.compute_nvi(empty_df)
    with pytest.raises(MissingColumnsError):
        ci.compute_returns(empty_df)
    with pytest.raises(MissingColumnsError):
        ci.compute_volatility(empty_df)

def test_all_indicators_single_row():
    """Test all indicators with single row of data."""
    single_row = pd.DataFrame({
        'open': [100],
        'high': [200],
        'low': [50],
        'close': [150],
        'volume': [1000]
    })
    
    # These should not raise errors but may return NaN values
    obv_result = compute_obv_price_divergence(single_row)
    vol_osc_result = compute_volume_oscillator(single_row)
    vwap_result = ci.compute_vwap(single_row)
    pvi_result = ci.compute_pvi(single_row)
    nvi_result = ci.compute_nvi(single_row)
    try:
        returns_result = ci.compute_returns(single_row)
    except ValueError:
        returns_result = None
    try:
        vol_result = ci.compute_volatility(single_row)
    except ValueError:
        vol_result = None
    assert isinstance(obv_result, pd.DataFrame)
    assert isinstance(vol_osc_result, pd.DataFrame)
    assert isinstance(vwap_result, pd.DataFrame)
    assert isinstance(pvi_result, pd.DataFrame)
    assert isinstance(nvi_result, pd.DataFrame)
    assert isinstance(returns_result, pd.Series)
    assert isinstance(vol_result, pd.Series) 