# PowerShell script to set up Python venv, install requirements, TA-lib, and verify

# 1. Set Python version
$python_version = "3.11.7"

# 2. Check if Python is available
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "Python is not installed or not in PATH. Please install Python $python_version."
    exit 1
}

# 3. Create venv if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# 4. Activate venv
$venv_activate = ".\venv\Scripts\Activate.ps1"
if (-not (Test-Path $venv_activate)) {
    Write-Host "Could not find venv activation script. Exiting."
    exit 1
}
Write-Host "Activating virtual environment..."
. $venv_activate

# 5. Upgrade pip
Write-Host "Upgrading pip..."
pip install --upgrade pip

# 6. Install requirements
if (Test-Path "requirements.txt") {
    Write-Host "Installing requirements..."
    pip install -r requirements.txt
} else {
    Write-Host "requirements.txt not found!"
    exit 1
}

# 7. Install TA-lib wheel for Python 3.11
$talib_wheel = "drivers/ta_lib-0.6.3-cp311-cp311-win_amd64.whl"
if (Test-Path $talib_wheel) {
    Write-Host "Installing TA-lib wheel..."
    pip install $talib_wheel
} else {
    Write-Host "TA-lib wheel not found in drivers directory!"
    exit 1
}

# 8. Verify installation
Write-Host "Verifying key package imports..."
$verify = @'
import numpy as np
import pandas as pd
import talib
import matplotlib.pyplot as plt
import seaborn as sns
print("All key packages imported successfully!")
'@
python -c $verify

Write-Host "\nSetup complete! To activate the environment in the future, run:`n.\venv\Scripts\Activate.ps1" 