# Auto-activate venv and run Python commands
# Usage: .\run.ps1 script.py arg1 arg2
# Example: .\run.ps1 main.py
# Example: .\run.ps1 run_phase2_consolidation.py -i file.csv -o outputs

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
$VenvPath = Join-Path $ProjectRoot "venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

# Use $args to capture everything (PowerShell won't parse these as parameters)
$PythonArgs = $args

# Check if venv exists
if (-not (Test-Path $VenvPython)) {
    Write-Host "ERROR: Virtual environment not found at $VenvPath" -ForegroundColor Red
    Write-Host "Run: python -m venv venv" -ForegroundColor Yellow
    exit 1
}

# Load .env if present
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim()
            [Environment]::SetEnvironmentVariable($key, $value, "Process")
        }
    }
}

# Run Python with venv
if ($PythonArgs.Count -eq 0) {
    Write-Host "Usage: .\run.ps1 script.py [args...]" -ForegroundColor Yellow
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\run.ps1 main.py" -ForegroundColor Gray
    Write-Host "  .\run.ps1 run_phase2_consolidation.py -i data.csv -o outputs -t 0.30" -ForegroundColor Gray
    exit 1
}

# Pass all arguments directly to Python
& $VenvPython $PythonArgs
exit $LASTEXITCODE
