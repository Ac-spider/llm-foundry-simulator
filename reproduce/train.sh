#!/bin/bash
set -e

echo "=== Stage: Train ==="

# Run training
echo "Running training..."
python run.py train --config configs/train.yaml

# Verify results
echo "Verifying results..."
python reproduce/verify.py --stage train

EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Train stage passed"
else
    echo "✗ Train stage failed"
fi

exit $EXIT_CODE
