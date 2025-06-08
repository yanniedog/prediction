import indicator_params
import json
from pathlib import Path

def test_indicator_params_loads_json(tmp_path):
    # Create a dummy indicator_params.json
    params = {"test_indicator": {"param1": 1, "param2": 2}}
    json_path = tmp_path / "indicator_params.json"
    json_path.write_text(json.dumps(params))
    loaded = indicator_params.load_indicator_definitions(path=json_path)
    assert loaded == params 