param(
    [string]$Corpus = "French"
)

$VENV_PYTHON = ".\venv_gpu\Scripts\python.exe"
if (-not (Test-Path $VENV_PYTHON)) {
    $VENV_PYTHON = "python"
}

Write-Host "Starting MATCHER against corpus: $Corpus" -ForegroundColor Cyan

& $VENV_PYTHON -m src.textreuse.cli run `
    --pt_dir "input" `
    --out_dir "output_results" `
    --include_corpora $Corpus `
    --viz

Write-Host "Matching complete. Check 'output_results' for reports." -ForegroundColor Green
