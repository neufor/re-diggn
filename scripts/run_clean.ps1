# Run EURUSD M1 cleaning script
# Usage: .\scripts\run_clean.ps1  (from project root)

Set-Location "C:\Projects\My\Re-Diggn"

# Initialise uv project if pyproject.toml doesn't exist yet
if (-not (Test-Path "pyproject.toml")) {
    Write-Host "Initialising uv project..."
    uv init --no-workspace --python 3.11
}

# Add required packages (uv is idempotent — safe to re-run)
Write-Host "Ensuring dependencies are present..."
uv add pandas pyarrow numpy

# Run the script
Write-Host "Running clean_eurusd_m1.py ..."
uv run python scripts\clean_eurusd_m1.py

Write-Host "Done. Log at scripts\clean_eurusd_m1.log"
