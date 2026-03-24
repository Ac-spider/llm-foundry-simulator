#!/bin/bash
set -e

echo "=== Stage: Alignment Training ==="

# Check prerequisites
if [ ! -f "configs/align.yaml" ]; then
    echo "Error: configs/align.yaml not found"
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

# Parse method from config (sft, dpo, or grpo)
METHOD=$(python -c "import yaml; print(yaml.safe_load(open('configs/align.yaml'))['method'])" 2>/dev/null || echo "sft")
echo "Alignment method: $METHOD"

# Run alignment training
echo "Running alignment training with method: $METHOD..."
python run.py align --config configs/align.yaml

# Verify results if verify.py is available
if [ "$VERIFY_AVAILABLE" = true ]; then
    echo "Verifying results..."
    python "$VERIFY_SCRIPT" --stage align
    EXIT_CODE=$?
else
    echo "Alignment training completed (verification skipped)"
    EXIT_CODE=0
fi

exit $EXIT_CODE
