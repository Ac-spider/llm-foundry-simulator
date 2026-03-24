#!/bin/bash
set -e

echo "=== Stage: Data Pipeline ==="

# Check prerequisites
if [ ! -f "configs/data.yaml" ]; then
    echo "Error: configs/data.yaml not found"
    exit 1
fi

if [ ! -f "run.py" ]; then
    echo "Error: run.py not found"
    exit 1
fi

# Check if verify.py exists
VERIFY_SCRIPT="reproduce/verify.py"
if [ ! -f "$VERIFY_SCRIPT" ]; then
    echo "Warning: $VERIFY_SCRIPT not found, skipping verification"
    VERIFY_AVAILABLE=false
else
    VERIFY_AVAILABLE=true
fi

# Run data pipeline
echo "Running data pipeline..."
python run.py data --config configs/data.yaml

# Verify results if verify.py is available
if [ "$VERIFY_AVAILABLE" = true ]; then
    echo "Verifying results..."
    python "$VERIFY_SCRIPT" --stage data
    EXIT_CODE=$?
else
    echo "Data pipeline completed (verification skipped)"
    EXIT_CODE=0
fi

exit $EXIT_CODE
