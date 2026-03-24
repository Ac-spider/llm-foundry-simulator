# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LLM Foundry Simulator is a unified CLI pipeline for LLM training workflows, integrating Stanford CS336 course assignments into a complete pipeline:

- **Stage 0** (`stage0_datagen/`): Data generation (SFT/GRPO data via API)
- **Stage 1** (`stage1_tokenize/`): BPE tokenizer training and pretokenization
- **Stage 2** (`stage2_train/`): Transformer training with DDP and Flash Attention
- **Stage 3** (`stage3_scaling/`): Chinchilla scaling laws and IsoFLOPs analysis
- **Stage 4** (`stage4_data/`): Data quality filtering pipeline
- **Stage 5** (`stage5_align/`): Alignment (SFT, DPO, GRPO)

## Common Commands

### Installation
```bash
pip install -e .
pip install -e ".[dev]"  # with dev dependencies
```

### Environment Check
```bash
python run.py env
```

### Running Pipeline Stages
```bash
# Data generation (requires SJTU_API_KEY env var)
python run.py datagen --config configs/datagen.yaml

# Tokenization
python run.py tokenize --config configs/tokenize.yaml

# Training
python run.py train --config configs/train.yaml
python run.py train --config configs/train.yaml --set training.lr=1e-4

# Distributed training
torchrun --nproc_per_node=4 run.py train --config configs/train.yaml --ddp

# Scaling laws analysis
python run.py scaling --config configs/scaling.yaml

# Data quality filtering
python run.py data --config configs/data.yaml

# Alignment (SFT/DPO/GRPO)
python run.py align --config configs/align.yaml --method sft
python run.py align --config configs/align.yaml --method dpo
python run.py align --config configs/align.yaml --method grpo
```

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_model.py -v
pytest tests/stage2_train/test_trainer.py -v

# Run tests for a specific stage
pytest tests/stage1_tokenize/ -v
pytest tests/stage3_scaling/ -v
```

### Code Quality
```bash
black llm_foundry/ tests/
isort llm_foundry/ tests/
mypy llm_foundry/
```

## Architecture

### Backend Abstraction (`llm_foundry/backends/`)

**Attention** (`attention.py`, `attention_triton.py`):
- Three-level fallback: Triton Flash Attention → torch.compile SDPA → F.sdpa
- Use `get_attention_fn(use_flash_attn=True)` to get best available backend
- Attention injection system in `stage2_train/attention_inject.py` for patching models

**Inference** (`inference.py`):
- Two-level fallback: vLLM → HuggingFace generate
- Used by GRPO for rollout generation

### Common Components (`llm_foundry/common/`)

**Model** (`model.py`): `BasicsTransformerLM` - Transformer with RoPE, supports flash attention via backend injection.

**Optimizer** (`optimizer.py`): AdamW with cosine schedule and `ShardedOptimizer` (ZeRO-1).

**Config** (`config.py`): YAML loading with `load_config()` returning SimpleNamespace for attribute access.

**Data** (`data.py`): `get_batch()` for memory-mapped token files, `load_tokens()` for binarized data.

### Stage Organization

Each stage follows a consistent pattern:
- Main logic in `llm_foundry/stageX_name/`
- Tests in `tests/stageX_name/`
- Config in `configs/*.yaml`

**Key interfaces**:
- Stage 0: `DataGenConfig.from_yaml()` → `run_datagen(cfg)`
- Stage 1: `BPETokenizer.train()` / `BPETokenizer.load()` → `tokenizer.encode/decode`
- Stage 2: `Trainer(cfg_dict).train()` - handles DDP internally via `distributed.py`
- Stage 3: `ScalingAnalyzer(cfg).run(experiments_data)`
- Stage 4: `DataPipeline(cfg).process_file(input, output)`
- Stage 5: `SFTTrainer`, `DPOTrainer`, `GRPOTrainer` with common interface

### Configuration System

Configs are YAML files with nested structure:
```yaml
model:
  vocab_size: 10000
  d_model: 256
training:
  batch_size: 16
  lr: 3e-4
```

Command-line overrides in `run.py`:
- `--set key=value` for arbitrary config overrides (e.g., `--set training.lr=1e-4`)
- `--data`, `--output` for common path overrides
- `--flash-attn` to force enable Flash Attention

### Reference Resources

`reference_resource/` contains original CS336 assignment implementations for reference when adapting code. Key mappings:
- Assignment1 → stage1_tokenize (BPE), common/model.py (Transformer)
- Assignment2 → backends/attention.py (Triton), common/optimizer.py (ZeRO-1)
- Assignment3 → stage3_scaling/ (Chinchilla fitting)
- Assignment4 → stage4_data/ (Gopher filters)
- Assignment5 → stage5_align/ (SFT, DPO, GRPO)

### Testing Strategy

Tests are organized by stage with integration tests for cross-module workflows:
- Unit tests: `test_*.py` for individual components
- Integration tests: `test_integration.py` for stage workflows
- Use mocked API calls for Stage 0 tests

## Development Notes

- **No stubs policy**: Only implemented code exists, no placeholder files
- **Backend detection**: Runtime detection of Triton/Flash Attention availability
- **DDP handling**: Training code checks `RANK` env var; use `torchrun` for multi-GPU
- **Commit protocol**: Use git trailers (see oh-my-claudecode commit_protocol in global CLAUDE.md)
