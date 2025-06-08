# Activate virtual environment if it exists
if (Test-Path "venv") {
    Write-Host "Activating virtual environment..."
    & .\venv\Scripts\Activate.ps1
}

# Install test dependencies if needed
Write-Host "Installing test dependencies..."
pip install -r requirements.txt

# Run tests with coverage
Write-Host "Running tests with coverage..."
pytest tests/ -v --cov=. --cov-report=term-missing --cov-report=html --cov-fail-under=90

# Check if tests passed
if ($LASTEXITCODE -eq 0) {
    Write-Host "`nAll tests passed with 90% coverage!" -ForegroundColor Green
    Write-Host "Coverage report generated in htmlcov/index.html"
} else {
    Write-Host "`nTests failed or coverage below 90%" -ForegroundColor Red
    exit 1
} 