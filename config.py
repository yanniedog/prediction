# config.py
import logging
import os
from pathlib import Path

logger = logging.getLogger()

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / 'database' / 'klines.db'
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
