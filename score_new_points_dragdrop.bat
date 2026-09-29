@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"

if "%~1"=="" (
    echo Drag and drop a points CSV onto this BAT file.
    pause
    exit /b
)

set "POINTS=%~1"
set "OUTDIR=%~dp0out"

if not exist "%OUTDIR%" (
    echo out folder not found:
    echo %OUTDIR%
    pause
    exit /b
)

set "CENTERLINE_NAME="
for /f "delims=" %%F in ('dir /b /a-d /o-d "%OUTDIR%\*_centerline_band*.csv" 2^>nul') do (
    set "CENTERLINE_NAME=%%F"
    goto :got_centerline
)

:got_centerline
if not defined CENTERLINE_NAME (
    echo No *_centerline_band*.csv found in:
    echo %OUTDIR%
    pause
    exit /b
)

set "CENTERLINE=%OUTDIR%\%CENTERLINE_NAME%"
set "PARAMS_NAME=!CENTERLINE_NAME:_centerline_band=_params_band!"
set "PARAMS=%OUTDIR%\!PARAMS_NAME!"

if not exist "!PARAMS!" (
    echo Matching params file not found for:
    echo %CENTERLINE%
    echo.
    echo Looked for:
    echo !PARAMS!
    pause
    exit /b
)

echo Using centerline:
echo %CENTERLINE%
echo.
echo Using params:
echo !PARAMS!
echo.
echo Scoring points file:
echo %POINTS%
echo.

python score_points_to_saved_centerline_magenta.py ^
  --centerline "%CENTERLINE%" ^
  --params "!PARAMS!" ^
  --points "%POINTS%" ^
  --outdir "%OUTDIR%"

pause