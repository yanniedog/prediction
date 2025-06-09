import indicator_params
import json
from pathlib import Path
import pytest
import tempfile
from indicator_params import (
    get_indicator_params,
    get_all_indicators,
    validate_indicator_params
)

def test_indicator_params_loads_json(tmp_path):
    # Create a dummy indicator_params.json
    params = {"test_indicator": {"param1": 1, "param2": 2}}
    json_path = tmp_path / "indicator_params.json"
    json_path.write_text(json.dumps(params))
    loaded = indicator_params.load_indicator_definitions(path=json_path)
    assert loaded == params 

def test_get_indicator_params(tmp_path):
    params = {
        "RSI": {"type": "momentum", "params": {"timeperiod": {"type": "int", "min": 2, "max": 30, "default": 14}}},
        "BB": {"type": "volatility", "params": {"timeperiod": {"type": "int", "min": 2, "max": 30, "default": 20}, "nbdevup": {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0}}}
    }
    file_path = tmp_path / "params.json"
    with open(file_path, "w") as f:
        json.dump(params, f)
    # Patch the module to use this file
    import indicator_params as ip
    ip.PARAMS_FILE = str(file_path)
    rsi = get_indicator_params("RSI")
    assert rsi["type"] == "momentum"
    bb = get_indicator_params("BB")
    assert bb["type"] == "volatility"
    with pytest.raises(KeyError):
        get_indicator_params("NON_EXISTENT")

def test_get_all_indicators(tmp_path):
    params = {
        "RSI": {"type": "momentum", "params": {"timeperiod": {"type": "int", "min": 2, "max": 30, "default": 14}}},
        "BB": {"type": "volatility", "params": {"timeperiod": {"type": "int", "min": 2, "max": 30, "default": 20}, "nbdevup": {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0}}}
    }
    file_path = tmp_path / "params.json"
    with open(file_path, "w") as f:
        json.dump(params, f)
    import indicator_params as ip
    ip.PARAMS_FILE = str(file_path)
    indicators = get_all_indicators()
    assert set(indicators) == {"RSI", "BB"}

def test_validate_indicator_params(tmp_path):
    params = {
        "RSI": {"type": "momentum", "params": {"timeperiod": {"type": "int", "min": 2, "max": 30, "default": 14}}},
        "BB": {"type": "volatility", "params": {"timeperiod": {"type": "int", "min": 2, "max": 30, "default": 20}, "nbdevup": {"type": "float", "min": 1.0, "max": 3.0, "default": 2.0}}}
    }
    file_path = tmp_path / "params.json"
    with open(file_path, "w") as f:
        json.dump(params, f)
    import indicator_params as ip
    ip.PARAMS_FILE = str(file_path)
    # Valid
    assert validate_indicator_params("RSI", {"timeperiod": 14})
    # Invalid type
    with pytest.raises(ValueError):
        validate_indicator_params("RSI", {"timeperiod": "bad"})
    # Out of range
    with pytest.raises(ValueError):
        validate_indicator_params("RSI", {"timeperiod": 100})
    # Missing param
    with pytest.raises(ValueError):
        validate_indicator_params("RSI", {})
    # Non-existent indicator
    with pytest.raises(KeyError):
        validate_indicator_params("NON_EXISTENT", {"timeperiod": 14}) 