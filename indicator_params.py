"""Module for loading and managing indicator definitions."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Path to indicator parameters JSON file
INDICATOR_PARAMS_PATH = Path("indicator_params.json")

def load_indicator_definitions(path: Path = INDICATOR_PARAMS_PATH) -> Dict[str, Any]:
    """Load indicator definitions from JSON file."""
    try:
        if not path.exists():
            logger.error(f"{path} not found")
            return {}
        with open(path, 'r') as f:
            definitions = json.load(f)
        if not isinstance(definitions, dict):
            logger.error("Invalid indicator definitions format")
            return {}
        return definitions
    except Exception as e:
        logger.error(f"Failed to load indicator definitions: {e}")
        return {}

# Load definitions at module level
indicator_definitions = load_indicator_definitions() 