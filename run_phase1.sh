#!/usr/bin/env bash
set -euo pipefail
python run_phase1.py --data-dir data --output-dir outputs --model-dir models
echo "Phase 1 baseline completed successfully."
