@echo off
echo Building Sphinx documentation...
cd docs
sphinx-build -b html . _build/html
echo.
echo Documentation built successfully!
echo Open: file:///c:/Users/MC/Desktop/Code/Python/PRJ381-data-preprocessing/docs/_build/html/index.html
pause
