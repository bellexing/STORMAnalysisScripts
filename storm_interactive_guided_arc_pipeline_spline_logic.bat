@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
    echo Drag and drop your arc CSV onto this BAT file.
    pause
    exit /b
)

set "ARC1=%~1"
set "OUTDIR=%~dp0out"

python storm_interactive_guided_arc_pipeline_spline_logic_green.py ^
  --arc1 "%ARC1%" ^
  --points "%ARC1%" ^
  --outdir "%OUTDIR%" ^
  --guide_band 120 ^
  --band 80 ^
  --circle_thresh 35 ^
  --min_radius 300 ^
  --smooth_factor 1.0 ^
  --n_samp 1000 ^
  --bin_w 5

pause