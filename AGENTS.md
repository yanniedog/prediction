# Repository instructions

This repository contains a Python command-line technical-analysis tool. The entry point is `main.py`; it downloads market data and writes databases and reports.

## Setup

- Use Python 3.11.7, as specified by `.python-version`.
- Create an isolated environment: `python -m venv .venv`.
- Activate `.venv/bin/activate` on POSIX or `.venv/Scripts/Activate.ps1` on PowerShell.
- Install the UTF-8 requirements with `python -m pip install -r requirements.txt`.
- The checked-in `pandas_ta==0.3.14b0` pin currently prevents a fresh Python 3.11 installation from PyPI. Do not claim the environment is ready until dependency installation succeeds. Do not substitute an unreviewed fork or remove dependencies silently.
- No checked-in startup script provisions Python, installs a pinned fork, or repairs dependencies automatically.
- TA-Lib and pandas_ta imports are used by the indicator modules. Check actual imports and platform support before changing their versions.

## Verification

- Run `python -m compileall -q .` for syntax validation inside an isolated clone without a virtual environment beneath it, or limit compilation to tracked Python files when `.venv` exists.
- Run `python -m pytest` for functional verification once dependencies are installed. Read `pytest.ini` for coverage, warning and timeout settings; the timeout option additionally requires `pytest-timeout`.
- Record the actual completion summary and exit status. Syntax checks do not demonstrate that dependencies, tests, or live data access work.
- Do not repeat historical pass counts or assume a particular host's Python version, network access or geographic restrictions.
- On Windows, never use `os.kill(pid, 0)` to check process liveness. Isolate unverified process-control tests from the shared console.

## Runtime and local data

- Inspect `config.py`, `data_manager.py` and `main.py` before running the interactive application. Startup cleanup can remove generated reports, logs and the leaderboard.
- Use a separate data directory or disposable clone for runtime checks; preserve existing databases and backups.
- Verify the configured market-data endpoint directly. Do not bypass geographic or access restrictions with an alternate endpoint.
- Keep generated databases, reports, backups, environments and credentials out of version control. `requirements.txt` remains tracked despite the broad text-file ignore rule.
