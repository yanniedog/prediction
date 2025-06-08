"""Module for loading and managing indicator definitions."""
import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

def load_indicator_definitions() -> Dict[str, Any]:
    """Load indicator definitions from JSON file."""
    try:
        json_path = Path("indicator_params.json")
        if not json_path.exists():
            logger.error("indicator_params.json not found")
            return {}
        
        with open(json_path, 'r') as f:
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