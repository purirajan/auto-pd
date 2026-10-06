#!/bin/zsh
set -e

echo "== 1. Training =="
python train.py

echo "== 2. Validation tests =="
python test_validation.py

echo "== 3. Batch scoring =="
python score.py

echo "Pipeline finished successfully."