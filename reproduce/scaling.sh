#!/bin/bash
set -e

echo "=== Stage: Scaling ==="

# Run scaling experiments
echo "Running scaling experiments..."
python run.py scaling --config configs/scaling.yaml

# Verify results
echo "Verifying results..."
python reproduce/verify.py --stage scaling

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Scaling stage passed"
else
    echo "✗ Scaling stage failed"
fi

exit $EXIT_CODE
