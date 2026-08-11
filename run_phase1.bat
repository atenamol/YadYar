@echo off
python run_phase1.py --data-dir data --output-dir outputs --model-dir models
if errorlevel 1 exit /b %errorlevel%
echo Phase 1 baseline completed successfully.
