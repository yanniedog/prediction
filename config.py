# filename: config.py
import os
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR=os.path.join(SCRIPT_DIR,'database')
DB_FILENAME='klines.db'
DB_PATH=os.path.join(DATABASE_DIR,DB_FILENAME)
os.makedirs(DATABASE_DIR,exist_ok=True)