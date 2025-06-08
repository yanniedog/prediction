import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_data_manager_initialization(data_manager):
    """Test DataManager initialization."""
    assert data_manager is not None
    assert hasattr(data_manager, 'config')

def test_load_data(data_manager, test_data, tmp_path):
    """Test loading data from different sources."""
    # Test loading from DataFrame
    loaded_data = data_manager.load_data(test_data)
    assert isinstance(loaded_data, pd.DataFrame)
    assert not loaded_data.empty
    assert all(col in loaded_data.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    # Test loading from CSV
    csv_path = tmp_path / "test_data.csv"
    test_data.to_csv(csv_path)
    loaded_from_csv = data_manager.load_data(str(csv_path))
    assert isinstance(loaded_from_csv, pd.DataFrame)
    assert not loaded_from_csv.empty

def test_data_validation(data_manager, test_data):
    """Test data validation methods."""
    # Test valid data
    assert data_manager.validate_data(test_data)

    # Test invalid data
    invalid_data = test_data.copy()
    invalid_data.loc[invalid_data.index[0], 'close'] = np.nan
    assert not data_manager.validate_data(invalid_data)

def test_data_preprocessing(data_manager, test_data):
    """Test data preprocessing methods."""
    processed_data = data_manager.preprocess_data(test_data)
    assert isinstance(processed_data, pd.DataFrame)
    assert not processed_data.empty
    assert not processed_data.isnull().any().any()

def test_data_splitting(data_manager, test_data):
    """Test train/test data splitting."""
    original_length = len(test_data)
    train_data, test_data = data_manager.split_data(test_data, test_size=0.2)
    assert isinstance(train_data, pd.DataFrame)
    assert isinstance(test_data, pd.DataFrame)
    assert len(train_data) + len(test_data) == original_length  # Total length should equal original
    assert len(test_data) == pytest.approx(original_length * 0.2, rel=0.1)  # Test size should be ~20%

def test_feature_engineering(data_manager, test_data):
    """Test feature engineering methods."""
    features = data_manager.engineer_features(test_data)
    assert isinstance(features, pd.DataFrame)
    assert not features.empty
    assert not features.isnull().any().any()

def test_data_normalization(data_manager, test_data):
    """Test data normalization methods."""
    normalized_data = data_manager.normalize_data(test_data)
    assert isinstance(normalized_data, pd.DataFrame)
    assert not normalized_data.empty
    assert normalized_data.select_dtypes(include=[np.number]).max().max() <= 1
    assert normalized_data.select_dtypes(include=[np.number]).min().min() >= -1

def test_data_aggregation(data_manager, test_data):
    """Test data aggregation methods."""
    # Test daily to weekly aggregation
    weekly_data = data_manager.aggregate_data(test_data, 'W')
    assert isinstance(weekly_data, pd.DataFrame)
    assert not weekly_data.empty
    assert len(weekly_data) < len(test_data)

    # Test daily to monthly aggregation
    monthly_data = data_manager.aggregate_data(test_data, 'M')
    assert isinstance(monthly_data, pd.DataFrame)
    assert not monthly_data.empty
    assert len(monthly_data) < len(weekly_data)

def test_data_cleaning(data_manager, test_data):
    """Test data cleaning methods."""
    # Add some outliers
    dirty_data = test_data.copy()
    dirty_data.loc[dirty_data.index[0], 'close'] = dirty_data['close'].mean() * 10
    
    cleaned_data = data_manager.clean_data(dirty_data)
    assert isinstance(cleaned_data, pd.DataFrame)
    assert not cleaned_data.empty
    assert not cleaned_data.isnull().any().any()
    
    # Verify outliers were handled
    assert cleaned_data['close'].max() < dirty_data['close'].max()

def test_data_sampling(data_manager, test_data):
    """Test data sampling methods."""
    # Test random sampling
    sampled_data = data_manager.sample_data(test_data, n_samples=10)
    assert isinstance(sampled_data, pd.DataFrame)
    assert len(sampled_data) == 10

    # Test systematic sampling
    systematic_sample = data_manager.sample_data(test_data, method='systematic', step=5)
    assert isinstance(systematic_sample, pd.DataFrame)
    assert len(systematic_sample) == len(test_data) // 5 