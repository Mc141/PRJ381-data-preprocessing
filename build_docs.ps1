# Build Sphinx Documentation
Write-Host "Building Sphinx documentation..." -ForegroundColor Green

Set-Location docs
sphinx-build -b html . _build/html

Write-Host ""
Write-Host "Documentation built successfully!" -ForegroundColor Green
Write-Host "Open: file:///c:/Users/MC/Desktop/Code/Python/PRJ381-data-preprocessing/docs/_build/html/index.html" -ForegroundColor Cyan

# Optionally open in browser
$choice = Read-Host "Open documentation in browser? (y/n)"
if ($choice -eq "y" -or $choice -eq "Y") {
    Start-Process "file:///c:/Users/MC/Desktop/Code/Python/PRJ381-data-preprocessing/docs/_build/html/index.html"
}
