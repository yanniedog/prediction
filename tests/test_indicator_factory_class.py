import pytest
from indicator_factory_class import IndicatorFactory
import json
from pathlib import Path
import tempfile
import shutil
from indicator_factory_class import IndicatorFactoryClass

def test_indicator_factory_loads_json(tmp_path, monkeypatch):
    params = {"test": {"a": 1}}
    json_path = tmp_path / "indicator_params.json"
    json_path.write_text(json.dumps(params))
    monkeypatch.chdir(tmp_path)
    factory = IndicatorFactory()
    assert factory.indicator_params == params
    assert isinstance(factory.indicators, dict)

def test_indicator_factory_fallback(monkeypatch):
    # Remove indicator_params.json and patch indicator_definitions
    monkeypatch.setattr('indicator_factory_class.indicator_definitions', {"fallback": {"b": 2}})
    monkeypatch.setattr('builtins.open', lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
    factory = IndicatorFactory()
    assert factory.indicator_params == {"fallback": {"b": 2}}

@pytest.fixture(scope="function")
def temp_params_file(tmp_path):
    params = {
        "RSI": {
            "type": "momentum",
            "inputs": ["close"],
            "parameters": {
                "period": {"type": "int", "min": 2, "max": 30, "default": 14}
            }
        },
        "BB": {
            "type": "volatility",
            "inputs": ["close"],
            "parameters": {
                "period": {"type": "int", "min": 2, "max": 30, "default": 20},
                "stddev": {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0}
            }
        }
    }
    file_path = tmp_path / "params.json"
    with open(file_path, "w") as f:
        json.dump(params, f)
    return file_path

def test_factory_class_initialization(temp_params_file):
    factory = IndicatorFactoryClass(str(temp_params_file))
    assert hasattr(factory, "params")
    assert "RSI" in factory.params
    assert "BB" in factory.params
    # Test with invalid file
    with pytest.raises((FileNotFoundError, json.JSONDecodeError, ValueError)):
        IndicatorFactoryClass("/invalid/path.json")

def test_get_indicator_params(temp_params_file):
    factory = IndicatorFactoryClass(str(temp_params_file))
    rsi_params = factory.get_indicator_params("RSI")
    assert isinstance(rsi_params, dict)
    assert rsi_params["type"] == "momentum"
    # Test with non-existent indicator
    with pytest.raises(KeyError):
        factory.get_indicator_params("NON_EXISTENT")

def test_get_all_indicators(temp_params_file):
    factory = IndicatorFactoryClass(str(temp_params_file))
    indicators = factory.get_all_indicators()
    assert set(indicators) == {"RSI", "BB"}

def test_validate_params(temp_params_file):
    factory = IndicatorFactoryClass(str(temp_params_file))
    # Valid params
    valid = {"period": 14}
    assert factory.validate_params("RSI", valid)
    # Invalid param type
    invalid = {"period": "not_an_int"}
    with pytest.raises(ValueError):
        factory.validate_params("RSI", invalid)
    # Out of range
    invalid = {"period": 100}
    with pytest.raises(ValueError):
        factory.validate_params("RSI", invalid)
    # Missing param
    invalid = {}
    with pytest.raises(ValueError):
        factory.validate_params("RSI", invalid)

def test_error_handling(temp_params_file):
    factory = IndicatorFactoryClass(str(temp_params_file))
    # Invalid indicator
    with pytest.raises(KeyError):
        factory.get_indicator_params("INVALID")
    # Invalid params structure
    with pytest.raises(ValueError):
        factory.validate_params("RSI", {"period": None})
    # Invalid file content
    bad_file = temp_params_file.parent / "bad.json"
    bad_file.write_text("not json")
    with pytest.raises(json.JSONDecodeError):
        IndicatorFactoryClass(str(bad_file)) 