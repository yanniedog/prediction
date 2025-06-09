import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
from pathlib import Path

import correlation_calculator
from correlation_calculator import CorrelationCalculator, calculate_correlation_indicator_vs_future_price


class TestCorrelationCalculatorExtended:
    """Extended tests for CorrelationCalculator class to improve coverage."""
    
    def setup_method(self):
        """Set up test data and calculator instance."""
        self.calculator = CorrelationCalculator()
        
        # Create test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'volume': np.random.randint(1000, 10000, 100),
            'sma_5': np.random.randn(100).cumsum() + 50,
            'rsi_14': np.random.uniform(0, 100, 100),
            'macd': np.random.randn(100),
            'bb_upper': np.random.randn(100).cumsum() + 110,
            'bb_lower': np.random.randn(100).cumsum() + 90
        }, index=dates)
        
        # Add some NaN values to test edge cases
        self.test_data.loc[10:15, 'sma_5'] = np.nan
        self.test_data.loc[20:25, 'rsi_14'] = np.nan
        
    def test_calculate_correlation_with_different_methods(self):
        """Test correlation calculation with different methods."""
        data1 = pd.Series([1, 2, 3, 4, 5])
        data2 = pd.Series([2, 4, 6, 8, 10])
        
        # Test pearson correlation
        result = self.calculator.calculate_correlation(data1, data2, 'pearson')
        assert abs(result - 1.0) < 1e-10
        
        # Test spearman correlation
        result = self.calculator.calculate_correlation(data1, data2, 'spearman')
        assert abs(result - 1.0) < 1e-10
        
        # Test kendall correlation
        result = self.calculator.calculate_correlation(data1, data2, 'kendall')
        assert abs(result - 1.0) < 1e-10
        
    def test_calculate_correlation_with_nan_values(self):
        """Test correlation calculation with NaN values."""
        data1 = pd.Series([1, 2, np.nan, 4, 5])
        data2 = pd.Series([2, 4, 6, np.nan, 10])
        
        result = self.calculator.calculate_correlation(data1, data2)
        assert not np.isnan(result)
        
    def test_calculate_correlations_with_multiple_lags(self):
        """Test correlation calculation with multiple lags."""
        data = self.test_data.copy()
        indicator = 'sma_5'
        max_lag = 5
        
        result = self.calculator.calculate_correlations(data, indicator, max_lag)
        
        assert isinstance(result, dict)
        assert len(result) == max_lag
        for lag in range(1, max_lag + 1):
            assert lag in result
            assert isinstance(result[lag], (float, type(None)))
            
    def test_calculate_rolling_correlation(self):
        """Test rolling correlation calculation."""
        data1 = pd.Series(np.random.randn(50))
        data2 = pd.Series(np.random.randn(50))
        window = 10
        
        result = self.calculator.calculate_rolling_correlation(data1, data2, window)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(data1)
        assert result.iloc[:window-1].isna().all()  # First window-1 values should be NaN
        
    def test_calculate_correlation_matrix(self):
        """Test correlation matrix calculation."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        result = self.calculator.calculate_correlation_matrix(data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (len(data.columns), len(data.columns))
        assert (result.index == data.columns).all()
        assert (result.columns == data.columns).all()
        
    def test_test_correlation_significance(self):
        """Test correlation significance testing."""
        data1 = pd.Series(np.random.randn(100))
        data2 = pd.Series(np.random.randn(100))
        
        result = self.calculator.test_correlation_significance(data1, data2)
        
        assert isinstance(result, dict)
        assert 'correlation' in result
        assert 'p_value' in result
        assert 'significant' in result
        assert isinstance(result['significant'], bool)
        
    def test_cluster_correlations(self):
        """Test correlation clustering."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14', 'macd']]
        
        result = self.calculator.cluster_correlations(data, n_clusters=2)
        
        assert isinstance(result, dict)
        assert 'cluster_labels' in result
        assert 'cluster_centers' in result
        assert 'silhouette_score' in result
        
    def test_analyze_correlation_network(self):
        """Test correlation network analysis."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        result = self.calculator.analyze_correlation_network(data, threshold=0.5)
        
        assert isinstance(result, dict)
        assert 'network' in result
        assert 'centrality' in result
        assert 'communities' in result
        
    def test_generate_correlation_report(self):
        """Test correlation report generation."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        result = self.calculator.generate_correlation_report(data)
        
        assert isinstance(result, dict)
        assert 'summary' in result
        assert 'strong_correlations' in result
        assert 'weak_correlations' in result
        
    def test_decompose_correlation(self):
        """Test correlation decomposition."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14', 'macd']]
        
        result = self.calculator.decompose_correlation(data, n_components=2)
        
        assert isinstance(result, dict)
        assert 'components' in result
        assert 'explained_variance' in result
        assert 'loadings' in result
        
    def test_analyze_correlation_stability(self):
        """Test correlation stability analysis."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        result = self.calculator.analyze_correlation_stability(data, window_size=20)
        
        assert isinstance(result, dict)
        assert 'stability_metrics' in result
        assert 'rolling_correlations' in result
        
    def test_forecast_correlations(self):
        """Test correlation forecasting."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        result = self.calculator.forecast_correlations(data, forecast_horizon=5)
        
        assert isinstance(result, dict)
        assert 'forecast' in result
        assert 'confidence_intervals' in result
        
    def test_detect_correlation_regimes(self):
        """Test correlation regime detection."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        result = self.calculator.detect_correlation_regimes(data, n_regimes=2)
        
        assert isinstance(result, dict)
        assert 'regime_labels' in result
        assert 'regime_characteristics' in result
        
    def test_detect_correlation_anomalies(self):
        """Test correlation anomaly detection."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        result = self.calculator.detect_correlation_anomalies(data, window_size=20, threshold=2.0)
        
        assert isinstance(result, dict)
        assert 'anomalies' in result
        assert 'anomaly_scores' in result
        
    def test_calculate_lag_correlation(self):
        """Test lag correlation calculation."""
        data1 = pd.Series(np.random.randn(50))
        data2 = pd.Series(np.random.randn(50))
        lags = range(1, 6)
        
        result = self.calculator.calculate_lag_correlation(data1, data2, lags)
        
        assert isinstance(result, pd.Series)
        assert len(result) == len(lags)
        
    def test_plot_correlation_heatmap(self):
        """Test correlation heatmap plotting."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        result = self.calculator.plot_correlation_heatmap(data)
        
        assert result is not None
        assert hasattr(result, 'savefig')
        
    def test_plot_correlation_scatter(self):
        """Test correlation scatter plotting."""
        data1 = pd.Series(np.random.randn(50))
        data2 = pd.Series(np.random.randn(50))
        
        result = self.calculator.plot_correlation_scatter(data1, data2)
        
        assert result is not None
        assert hasattr(result, 'savefig')
        
    def test_visualize_correlation(self):
        """Test correlation visualization."""
        data1 = pd.Series(np.random.randn(50))
        data2 = pd.Series(np.random.randn(50))
        
        result = self.calculator.visualize_correlation(data1, data2)
        
        assert result is not None
        assert hasattr(result, 'savefig')
        
    def test_visualize_correlation_matrix(self):
        """Test correlation matrix visualization."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        result = self.calculator.visualize_correlation_matrix(data)
        
        assert result is not None
        assert hasattr(result, 'savefig')


class TestCorrelationCalculatorFunctions:
    """Tests for standalone functions in correlation_calculator.py."""
    
    def test_calculate_correlation_indicator_vs_future_price_valid(self):
        """Test valid correlation calculation."""
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105],
            'indicator': [1, 2, 3, 4, 5, 6]
        })
        
        result = calculate_correlation_indicator_vs_future_price(data, 'indicator', 2)
        
        assert isinstance(result, (float, type(None)))
        if result is not None:
            assert -1 <= result <= 1
            
    def test_calculate_correlation_indicator_vs_future_price_missing_columns(self):
        """Test correlation calculation with missing columns."""
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105]
        })
        
        result = calculate_correlation_indicator_vs_future_price(data, 'missing_indicator', 2)
        
        assert result is None
        
    def test_calculate_correlation_indicator_vs_future_price_invalid_lag(self):
        """Test correlation calculation with invalid lag."""
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105],
            'indicator': [1, 2, 3, 4, 5, 6]
        })
        
        result = calculate_correlation_indicator_vs_future_price(data, 'indicator', 0)
        
        assert result is None
        
    def test_calculate_correlation_indicator_vs_future_price_all_nan(self):
        """Test correlation calculation with all NaN values."""
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105],
            'indicator': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
        })
        
        result = calculate_correlation_indicator_vs_future_price(data, 'indicator', 2)
        
        assert np.isnan(result)
        
    def test_calculate_correlation_indicator_vs_future_price_constant_value(self):
        """Test correlation calculation with constant indicator values."""
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104, 105],
            'indicator': [5, 5, 5, 5, 5, 5]
        })
        
        result = calculate_correlation_indicator_vs_future_price(data, 'indicator', 2)
        
        assert np.isnan(result)
        
    def test_calculate_correlation_indicator_vs_future_price_insufficient_data(self):
        """Test correlation calculation with insufficient data."""
        data = pd.DataFrame({
            'close': [100, 101],
            'indicator': [1, 2]
        })
        
        result = calculate_correlation_indicator_vs_future_price(data, 'indicator', 2)
        
        assert np.isnan(result)
        
    @patch('correlation_calculator.sqlite_manager')
    @patch('correlation_calculator.leaderboard_manager')
    def test_process_correlations_success(self, mock_leaderboard, mock_sqlite):
        """Test successful correlation processing."""
        # Mock database connection
        mock_conn = Mock()
        mock_sqlite.create_connection.return_value = mock_conn
        
        # Create test data
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'indicator_1': np.random.randn(100),
            'indicator_2': np.random.randn(100)
        })
        
        # Mock indicator configs
        indicator_configs = [
            {'config_id': 1, 'indicator_name': 'indicator_1'},
            {'config_id': 2, 'indicator_name': 'indicator_2'}
        ]
        
        # Mock display and periodic report functions
        display_func = Mock()
        periodic_func = Mock()
        
        result = correlation_calculator.process_correlations(
            data=data,
            db_path='test.db',
            symbol_id=1,
            timeframe_id=1,
            indicator_configs_processed=indicator_configs,
            max_lag=5,
            analysis_start_time_global=time.time(),
            total_analysis_steps_global=10,
            current_step_base=1.0,
            total_steps_in_phase=2.0,
            display_progress_func=display_func,
            periodic_report_func=periodic_func
        )
        
        assert isinstance(result, bool)
        mock_conn.close.assert_called_once()
        
    def test_process_correlations_missing_close_column(self):
        """Test correlation processing with missing close column."""
        data = pd.DataFrame({
            'indicator_1': np.random.randn(100),
            'indicator_2': np.random.randn(100)
        })
        
        indicator_configs = [{'config_id': 1, 'indicator_name': 'indicator_1'}]
        
        display_func = Mock()
        periodic_func = Mock()
        
        result = correlation_calculator.process_correlations(
            data=data,
            db_path='test.db',
            symbol_id=1,
            timeframe_id=1,
            indicator_configs_processed=indicator_configs,
            max_lag=5,
            analysis_start_time_global=time.time(),
            total_analysis_steps_global=10,
            current_step_base=1.0,
            total_steps_in_phase=2.0,
            display_progress_func=display_func,
            periodic_report_func=periodic_func
        )
        
        assert result is False
        
    def test_process_correlations_invalid_max_lag(self):
        """Test correlation processing with invalid max lag."""
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'indicator_1': np.random.randn(100)
        })
        
        indicator_configs = [{'config_id': 1, 'indicator_name': 'indicator_1'}]
        
        display_func = Mock()
        periodic_func = Mock()
        
        result = correlation_calculator.process_correlations(
            data=data,
            db_path='test.db',
            symbol_id=1,
            timeframe_id=1,
            indicator_configs_processed=indicator_configs,
            max_lag=0,  # Invalid
            analysis_start_time_global=time.time(),
            total_analysis_steps_global=10,
            current_step_base=1.0,
            total_steps_in_phase=2.0,
            display_progress_func=display_func,
            periodic_report_func=periodic_func
        )
        
        assert result is False
        
    def test_process_correlations_insufficient_data(self):
        """Test correlation processing with insufficient data."""
        data = pd.DataFrame({
            'close': [100, 101],  # Only 2 rows
            'indicator_1': [1, 2]
        })
        
        indicator_configs = [{'config_id': 1, 'indicator_name': 'indicator_1'}]
        
        display_func = Mock()
        periodic_func = Mock()
        
        result = correlation_calculator.process_correlations(
            data=data,
            db_path='test.db',
            symbol_id=1,
            timeframe_id=1,
            indicator_configs_processed=indicator_configs,
            max_lag=5,
            analysis_start_time_global=time.time(),
            total_analysis_steps_global=10,
            current_step_base=1.0,
            total_steps_in_phase=2.0,
            display_progress_func=display_func,
            periodic_report_func=periodic_func
        )
        
        assert result is False
        
    def test_process_correlations_no_indicator_columns(self):
        """Test correlation processing with no indicator columns."""
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'not_an_indicator': np.random.randn(100)
        })
        
        indicator_configs = [{'config_id': 1, 'indicator_name': 'indicator_1'}]
        
        display_func = Mock()
        periodic_func = Mock()
        
        result = correlation_calculator.process_correlations(
            data=data,
            db_path='test.db',
            symbol_id=1,
            timeframe_id=1,
            indicator_configs_processed=indicator_configs,
            max_lag=5,
            analysis_start_time_global=time.time(),
            total_analysis_steps_global=10,
            current_step_base=1.0,
            total_steps_in_phase=2.0,
            display_progress_func=display_func,
            periodic_report_func=periodic_func
        )
        
        assert result is False
        
    @patch('correlation_calculator.sqlite_manager')
    def test_process_correlations_db_connection_failure(self, mock_sqlite):
        """Test correlation processing with database connection failure."""
        mock_sqlite.create_connection.return_value = None
        
        data = pd.DataFrame({
            'close': np.random.randn(100).cumsum() + 100,
            'indicator_1': np.random.randn(100)
        })
        
        indicator_configs = [{'config_id': 1, 'indicator_name': 'indicator_1'}]
        
        display_func = Mock()
        periodic_func = Mock()
        
        result = correlation_calculator.process_correlations(
            data=data,
            db_path='test.db',
            symbol_id=1,
            timeframe_id=1,
            indicator_configs_processed=indicator_configs,
            max_lag=5,
            analysis_start_time_global=time.time(),
            total_analysis_steps_global=10,
            current_step_base=1.0,
            total_steps_in_phase=2.0,
            display_progress_func=display_func,
            periodic_report_func=periodic_func
        )
        
        assert result is False


class TestCorrelationCalculatorEdgeCases:
    """Tests for edge cases and error handling."""
    
    def setup_method(self):
        """Set up test data."""
        self.calculator = CorrelationCalculator()
        
    def test_calculate_correlation_empty_series(self):
        """Test correlation calculation with empty series."""
        data1 = pd.Series([])
        data2 = pd.Series([])
        
        with pytest.raises(ValueError):
            self.calculator.calculate_correlation(data1, data2)
            
    def test_calculate_correlation_different_lengths(self):
        """Test correlation calculation with different length series."""
        data1 = pd.Series([1, 2, 3])
        data2 = pd.Series([1, 2, 3, 4])
        
        with pytest.raises(ValueError):
            self.calculator.calculate_correlation(data1, data2)
            
    def test_calculate_correlation_invalid_method(self):
        """Test correlation calculation with invalid method."""
        data1 = pd.Series([1, 2, 3, 4, 5])
        data2 = pd.Series([2, 4, 6, 8, 10])
        
        with pytest.raises(ValueError):
            self.calculator.calculate_correlation(data1, data2, 'invalid_method')
            
    def test_calculate_rolling_correlation_invalid_window(self):
        """Test rolling correlation with invalid window."""
        data1 = pd.Series([1, 2, 3, 4, 5])
        data2 = pd.Series([2, 4, 6, 8, 10])
        
        with pytest.raises(ValueError):
            self.calculator.calculate_rolling_correlation(data1, data2, 0)
            
    def test_cluster_correlations_invalid_clusters(self):
        """Test clustering with invalid number of clusters."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(ValueError):
            self.calculator.cluster_correlations(data, n_clusters=0)
            
    def test_analyze_correlation_network_invalid_threshold(self):
        """Test network analysis with invalid threshold."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(ValueError):
            self.calculator.analyze_correlation_network(data, threshold=2.0)
            
    def test_decompose_correlation_invalid_components(self):
        """Test decomposition with invalid number of components."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(ValueError):
            self.calculator.decompose_correlation(data, n_components=0)
            
    def test_forecast_correlations_invalid_horizon(self):
        """Test forecasting with invalid horizon."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(ValueError):
            self.calculator.forecast_correlations(data, forecast_horizon=0)
            
    def test_detect_correlation_regimes_invalid_regimes(self):
        """Test regime detection with invalid number of regimes."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(ValueError):
            self.calculator.detect_correlation_regimes(data, n_regimes=0)
            
    def test_detect_correlation_anomalies_invalid_window(self):
        """Test anomaly detection with invalid window size."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(ValueError):
            self.calculator.detect_correlation_anomalies(data, window_size=0)
            
    def test_detect_correlation_anomalies_invalid_threshold(self):
        """Test anomaly detection with invalid threshold."""
        data = pd.DataFrame({
            'col1': [1, 2, 3, 4, 5],
            'col2': [2, 4, 6, 8, 10]
        })
        
        with pytest.raises(ValueError):
            self.calculator.detect_correlation_anomalies(data, threshold=0)
            
    def test_calculate_lag_correlation_invalid_lags(self):
        """Test lag correlation with invalid lags."""
        data1 = pd.Series([1, 2, 3, 4, 5])
        data2 = pd.Series([2, 4, 6, 8, 10])
        
        with pytest.raises(ValueError):
            self.calculator.calculate_lag_correlation(data1, data2, [])
            
    def test_calculate_lag_correlation_negative_lags(self):
        """Test lag correlation with negative lags."""
        data1 = pd.Series([1, 2, 3, 4, 5])
        data2 = pd.Series([2, 4, 6, 8, 10])
        
        with pytest.raises(ValueError):
            self.calculator.calculate_lag_correlation(data1, data2, [-1, 0, 1])


class TestCorrelationCalculatorIntegration:
    """Integration tests for correlation calculator."""
    
    def setup_method(self):
        """Set up test data."""
        self.calculator = CorrelationCalculator()
        
        # Create realistic test data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=200, freq='D')
        
        # Create correlated data
        base = np.random.randn(200).cumsum()
        self.test_data = pd.DataFrame({
            'close': base + 100,
            'volume': base * 1000 + 5000,
            'sma_5': base.rolling(5).mean() + 50,
            'rsi_14': np.random.uniform(20, 80, 200),
            'macd': base.diff(12) - base.diff(26),
            'bb_upper': base + 110,
            'bb_lower': base + 90
        }, index=dates)
        
        # Fill NaN values
        self.test_data = self.test_data.fillna(method='bfill').fillna(method='ffill')
        
    def test_full_correlation_analysis_workflow(self):
        """Test complete correlation analysis workflow."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        # 1. Calculate correlation matrix
        corr_matrix = self.calculator.calculate_correlation_matrix(data)
        assert not corr_matrix.isna().all().all()
        
        # 2. Test significance
        significance = self.calculator.test_correlation_significance(
            data['close'], data['volume']
        )
        assert 'significant' in significance
        
        # 3. Generate report
        report = self.calculator.generate_correlation_report(data)
        assert 'summary' in report
        
        # 4. Analyze stability
        stability = self.calculator.analyze_correlation_stability(data)
        assert 'stability_metrics' in stability
        
        # 5. Detect regimes
        regimes = self.calculator.detect_correlation_regimes(data)
        assert 'regime_labels' in regimes
        
    def test_correlation_visualization_workflow(self):
        """Test correlation visualization workflow."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14']]
        
        # 1. Create heatmap
        heatmap = self.calculator.plot_correlation_heatmap(data)
        assert heatmap is not None
        
        # 2. Create scatter plot
        scatter = self.calculator.plot_correlation_scatter(
            data['close'], data['volume']
        )
        assert scatter is not None
        
        # 3. Visualize correlation matrix
        matrix_viz = self.calculator.visualize_correlation_matrix(data)
        assert matrix_viz is not None
        
    def test_advanced_correlation_analysis(self):
        """Test advanced correlation analysis features."""
        data = self.test_data[['close', 'volume', 'sma_5', 'rsi_14', 'macd']]
        
        # 1. Clustering
        clusters = self.calculator.cluster_correlations(data, n_clusters=2)
        assert len(clusters['cluster_labels']) == len(data.columns)
        
        # 2. Network analysis
        network = self.calculator.analyze_correlation_network(data)
        assert 'network' in network
        
        # 3. Decomposition
        decomposition = self.calculator.decompose_correlation(data)
        assert 'components' in decomposition
        
        # 4. Forecasting
        forecast = self.calculator.forecast_correlations(data)
        assert 'forecast' in forecast
        
        # 5. Anomaly detection
        anomalies = self.calculator.detect_correlation_anomalies(data)
        assert 'anomalies' in anomalies 