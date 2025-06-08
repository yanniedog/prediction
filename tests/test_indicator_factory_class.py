import pytest
from indicator_factory_class import IndicatorFactory
import json
from pathlib import Path

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