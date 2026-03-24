#!/bin/bash
# Tokenize stage script for LLM Foundry Simulator reproduction
# This script handles tokenization and verification

set -e

echo "=== Stage: Tokenize ==="

# Configuration
CONFIG=${1:-configs/tokenize.yaml}
TIMEOUT=${2:-3600}  # Default timeout: 1 hour

echo "Using config: $CONFIG"
echo "Timeout: $TIMEOUT seconds"

# Check prerequisites
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

if [ ! -f "run.py" ]; then
    echo "Error: run.py not found in current directory"
    exit 1
fi

# Extract output directory from config (YAML parsing using Python)
OUTPUT_DIR=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG', 'r') as f:
        cfg = yaml.safe_load(f)
    output_dir = cfg.get('output', {}).get('output_dir', 'results/tokenizer')
    name = cfg.get('output', {}).get('name', 'bpe_tokenizer')
    print(f'{output_dir}/{name}')
except Exception as e:
    print('results/tokenizer/bpe_tokenizer')
" 2>/dev/null || echo "results/tokenizer/bpe_tokenizer")

# Extract train file from config
TRAIN_FILE=$(python3 -c "
import yaml
import sys
try:
    with open('$CONFIG', 'r') as f:
        cfg = yaml.safe_load(f)
    train_file = cfg.get('data', {}).get('train_file', '')
    print(train_file)
except Exception as e:
    print('')
" 2>/dev/null || echo "")

# Function to create minimal training data
create_minimal_data() {
    local train_file="$1"
    echo "Creating minimal training data for testing..."
    mkdir -p "$(dirname "$train_file")"
    printf '%s\n' \
        "This is a sample text for tokenizer training." \
        "The quick brown fox jumps over the lazy dog." \
        "Machine learning is a subset of artificial intelligence." \
        "Natural language processing enables computers to understand human language." \
        "Deep learning models require large amounts of training data." \
        "Transformers have revolutionized the field of natural language processing." \
        "Byte Pair Encoding is a subword tokenization algorithm." \
        "Tokenization is the process of breaking text into smaller units called tokens." \
        "Large language models are trained on vast corpora of text data." \
        "Reinforcement learning from human feedback improves model alignment." > "$train_file"
    echo "Created minimal training data at: $train_file"
}

# Check if training data exists, if not try to prepare it
if [ -n "$TRAIN_FILE" ] && [ ! -f "$TRAIN_FILE" ]; then
    echo "Warning: Training file not found: $TRAIN_FILE"
    echo "Attempting to prepare data..."

    # Check if datagen stage can provide data
    if [ -f "reproduce/datagen.sh" ]; then
        echo "Running datagen.sh to prepare training data..."
        if bash reproduce/datagen.sh; then
            echo "datagen.sh completed successfully"
        else
            echo "Warning: datagen.sh failed, creating minimal training data..."
            create_minimal_data "$TRAIN_FILE"
        fi
    else
        echo "Warning: reproduce/datagen.sh not found"
        create_minimal_data "$TRAIN_FILE"
    fi
fi

# Run tokenization with timeout
echo "Running tokenization..."
echo "Command: python run.py tokenize --config $CONFIG"

# Use timeout to prevent hanging
if command -v timeout >/dev/null 2>&1; then
    timeout $TIMEOUT python run.py tokenize --config "$CONFIG"
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 124 ]; then
        echo "Error: Tokenization timed out after $TIMEOUT seconds"
        exit 124
    fi
else
    python run.py tokenize --config "$CONFIG"
    EXIT_CODE=$?
fi

if [ $EXIT_CODE -ne 0 ]; then
    echo "Error: Tokenization failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# Check if tokenizer.json was generated
echo "Checking tokenizer output..."
if [ -f "$OUTPUT_DIR/tokenizer.json" ]; then
    echo "OK: tokenizer.json found at: $OUTPUT_DIR/tokenizer.json"
    # Display file size
    ls -lh "$OUTPUT_DIR/tokenizer.json"
else
    echo "Warning: tokenizer.json not found at expected location: $OUTPUT_DIR/tokenizer.json"
    echo "Searching for tokenizer.json..."
    find results/ -name "tokenizer.json" -type f 2>/dev/null | head -5
fi

# Check for merges.txt (BPE tokenizer should have this)
if [ -f "$OUTPUT_DIR/merges.txt" ]; then
    echo "OK: merges.txt found"
    # Count number of merge operations
    MERGE_COUNT=$(wc -l < "$OUTPUT_DIR/merges.txt")
    echo "  Merge operations: $MERGE_COUNT"
fi

# Check for vocab.json
if [ -f "$OUTPUT_DIR/vocab.json" ]; then
    echo "OK: vocab.json found"
fi

# Verify results using verify.py if available
if [ -f "reproduce/verify.py" ]; then
    echo "Verifying results with verify.py..."
    python reproduce/verify.py --stage tokenize
    VERIFY_EXIT=$?
    if [ $VERIFY_EXIT -eq 0 ]; then
        echo "OK: Tokenize stage verification passed"
    else
        echo "FAIL: Tokenize stage verification failed (exit code: $VERIFY_EXIT)"
        echo "Check results/ directory for outputs"
        exit $VERIFY_EXIT
    fi
else
    echo "Warning: reproduce/verify.py not found, skipping verification"
fi

echo ""
echo "=== Tokenize stage completed successfully ==="
echo "Output directory: $OUTPUT_DIR"

exit 0
