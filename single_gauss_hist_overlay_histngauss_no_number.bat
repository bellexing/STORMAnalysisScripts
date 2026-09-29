@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if "%~1"=="" (
    echo Drag and drop the 568 histogram CSV onto this BAT file.
    echo.
    echo Example expected input:
    echo out\8_Em568_Scar647_standalone_section1_568_table_signed_dist_hist_density_bin5nm.csv
    pause
    exit /b
)

set "CSV568=%~1"
set "CSV647=%CSV568:_568_=_647_%"
set "OUTDIR=%~dp0output_single_gauss"

if not exist "%CSV568%" (
    echo 568 histogram not found:
    echo %CSV568%
    pause
    exit /b
)

if /I "%CSV568%"=="%CSV647%" (
    echo Could not infer the matching 647 file from this filename:
    echo %CSV568%
    echo.
    echo The file should contain _568_ in its name.
    pause
    exit /b
)

if not exist "%CSV647%" (
    echo Matching 647 histogram not found.
    echo.
    echo 568 file:
    echo %CSV568%
    echo.
    echo Looked for:
    echo %CSV647%
    pause
    exit /b
)

if not exist "%OUTDIR%" mkdir "%OUTDIR%"

echo Using 568 histogram:
echo %CSV568%
echo.
echo Using 647 histogram:
echo %CSV647%
echo.

py -3 single_gauss_hist_overlay_histngauss_no_number.py ^
  --csv568 "%CSV568%" ^
  --csv647 "%CSV647%" ^
  --outdir "%OUTDIR%"

echo.
echo Done. Results are in: %OUTDIR%
pause
endlocal