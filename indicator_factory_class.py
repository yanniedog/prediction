import json
from typing import Dict, Any
from indicator_params import indicator_definitions

class IndicatorFactory:
    """Class-based implementation for loading and managing indicator parameters."""
    def __init__(self):
        self.indicator_params = self.load_params()
        self.indicators = {}  # Initialize empty dict for indicators

    def load_params(self) -> Dict[str, Dict[str, Any]]:
        """Load indicator parameters from indicator_params.json or indicator_definitions."""
        try:
            with open('indicator_params.json', 'r') as f:
                params = json.load(f)
            return params
        except Exception:
            # Fallback to Python definitions if JSON fails
            return indicator_definitions 