# AGENTS.md

## Cursor Cloud specific instructions

This is a Python (3.11) command-line crypto technical-analysis / correlation tool.
The entry point is `python main.py` (interactive menu: Analysis / Custom / Backtest / Quit).
It downloads OHLCV klines from Binance, computes TA-Lib / custom indicators, correlates
them against future price across lags, and produces reports/predictions/backtests.

### Environment / how to run
- Python 3.11.7 is required (`.python-version`). The startup update script provisions it via
  `uv` and creates the virtualenv at `.venv`. System Python is 3.12 and will NOT work
  (the pinned `numpy==1.24.4` has no 3.12 wheels).
- Activate with `source .venv/bin/activate`, then run `python main.py`, `python -m pytest`, etc.
- `requirements.txt` is UTF-8. Install with `pip install -r requirements.txt` or
  `uv pip install -r requirements.txt`.

### Non-obvious dependency notes (already handled by the update script)
- `pandas_ta==0.3.14b0` (pinned in `requirements.txt`) was removed from PyPI and its repo is gone.
  It is installed from a pinned fork commit under the import name `pandas_ta`
  (`git+https://github.com/aarigs/pandas-ta.git@<sha>` with `--no-deps`). No configured indicator
  actually uses pandas_ta (only `talib` and `custom` types in `indicator_params.json`); it only
  needs to be importable.
- TA-Lib: the repo ships only Windows wheels in `drivers/`. On Linux the update script installs
  the `TA-Lib` PyPI wheel (bundles the C library). TA-Lib IS required (11 `talib` indicators).
- `indicator_factory.py` emits `warnings.warn(...)` if `talib` or `pandas_ta` are missing, and
  `pytest.ini` turns UserWarnings into errors, so BOTH must be importable or test collection fails.
- `pytest.ini` passes `--timeout=30`, so `pytest-timeout` must be installed (it is not listed in
  `requirements.txt`); the update script installs it.

### Testing
- Run `python -m pytest` (config in `pytest.ini`: coverage gate `--cov-fail-under=90`,
  warnings-as-errors, `--maxfail=6`).
- Known pre-existing failures unrelated to environment setup (this is a WIP repo, ~72% coverage):
  several tests assume Windows paths (e.g. `C:/Windows/System32/...`, `/invalid/path`) and some
  test setups use integer `.loc` slicing on a `DatetimeIndex` which pandas 2.x rejects. About
  377/419 tests pass with the current dependency set.
- A harmless `lost sys.stderr` / `I/O operation on closed file` message can appear at interpreter
  shutdown (a logging-teardown quirk in `tests/conftest.py`); it does not affect results.
- There is no configured linter (no flake8/ruff/pylint config); `python -m py_compile *.py` is a
  reasonable syntax check.

### Data source / running the app end-to-end
- The app fetches from `https://api.binance.com` (hardcoded in `data_manager.py`). That host returns
  HTTP 451 (geo-blocked) from the cloud VM, so the built-in download will fail here.
- The non-geoblocked market-data mirror `https://data-api.binance.vision/api/v3/klines` serves the
  identical kline format and works from the VM. For a runnable demo you can set
  `data_manager.BINANCE_API_BASE_URL = "https://data-api.binance.vision/api/v3/klines"` at runtime
  (do not edit the source) before calling `data_manager.download_binance_data(...)`.
- Per-symbol databases are written to `database/<SYMBOL>_<TIMEFRAME>.db`; the leaderboard is
  `correlation_leaderboard.db`. `main.py` deletes reports/logs/leaderboard on startup (auto-cleanup).
