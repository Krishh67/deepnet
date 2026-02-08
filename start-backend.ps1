# Start Backend Server
Write-Host "🌊 Starting OceanGuard Backend API..." -ForegroundColor Cyan
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "✗ Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# Check if API key is set
if (-not (Test-Path .env)) {
    Write-Host "✗ .env file not found" -ForegroundColor Yellow
    Write-Host "  Please create .env file with GEMINI_API_KEY=your_key_here" -ForegroundColor Yellow
    exit 1
}

Write-Host "✓ Environment file found" -ForegroundColor Green
Write-Host ""
Write-Host "Starting FastAPI server on http://localhost:8000..." -ForegroundColor Cyan
Write-Host "API docs will be available at http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Start the server
python -m uvicorn api1:app --reload --port 8000
