# LLM Foundry Simulator 使用指南

本文档提供 LLM Foundry Simulator 各命令的详细使用说明。

## 目录

- [环境检查](#环境检查)
- [数据生成](#数据生成)
- [分词](#分词)
- [训练](#训练)
- [缩放律分析](#缩放律分析)
- [数据质量过滤](#数据质量过滤)
- [对齐训练](#对齐训练)

---

## 环境检查

检查硬件环境和依赖是否满足要求。

```bash
python run.py env
```

输出信息包括：
- Python 版本
- PyTorch 版本和 CUDA 可用性
- GPU 信息（型号、显存）
- 可选依赖检查（Triton、Flash Attention 等）

---

## 数据生成

使用 DeepSeek API 生成 SFT 和 GRPO 训练数据。

### 基本用法

```bash
python run.py datagen --config configs/datagen.yaml
```

### 配置文件说明 (`configs/datagen.yaml`)

```yaml
datagen:
  sft_n: 500              # SFT 指令数据条数
  grpo_n: 500             # GRPO 数学推理数据条数
  sft_output: results/datagen/sft_data.jsonl
  grpo_output: results/datagen/grpo_data.jsonl
```

### 环境变量

使用前需设置 API Key：

```bash
export SJTU_API_KEY=your_key
```

### 输出

- SFT 数据：`results/datagen/sft_data.jsonl`
- GRPO 数据：`results/datagen/grpo_data.jsonl`

---

## 分词

训练 BPE Tokenizer 并对文本进行编码。

### 基本用法

```bash
python run.py tokenize --config configs/tokenize.yaml
```

### 配置文件说明 (`configs/tokenize.yaml`)

```yaml
data:
  train_file: "data/raw/openwebtext_train.txt"  # 训练 tokenizer 的原始文本
  val_file: null                                  # 可选验证文件
  test_file: null                                 # 可选测试文件

training:
  vocab_size: 10000                               # 词表大小
  special_tokens:                                 # 特殊 token
    - "<|endoftext|>"
    - "<|padding|>"
    - "<|unk|>"
  pre_tokenize_pattern: "'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"

output:
  output_dir: "results/tokenizer"
  name: "bpe_tokenizer"
  save_checkpoints: false

encoding:
  allow_unk: true
  max_length: 1024

logging:
  level: "INFO"
  verbose: true
```

### 输出

- Tokenizer 保存目录：`results/tokenizer/bpe_tokenizer/`
- 包含词汇表文件和合并规则

---

## 训练

训练 Transformer 语言模型。

### 基本用法

```bash
python run.py train --config configs/train.yaml
```

### 配置文件说明 (`configs/train.yaml`)

```yaml
model:
  vocab_size: 10000       # 词表大小
  context_length: 256     # 上下文窗口长度
  d_model: 256            # 模型隐层维度
  num_layers: 4           # Transformer 层数
  num_heads: 4            # 注意力头数
  d_ff: 1024              # FFN 中间层维度
  rope_theta: 10000.0     # RoPE 基础频率
  use_flash_attn: false   # 是否使用 Flash Attention

training:
  data_path: data/train_tokens.bin  # 分词后的数据文件
  device: auto                      # 自动选择设备
  batch_size: 16                    # 批次大小
  gradient_accumulation_steps: 4    # 梯度累积步数
  max_steps: 10000                  # 总训练步数
  lr: 3.0e-4                        # 学习率
  min_lr: 3.0e-5                    # 最小学习率
  warmup_steps: 500                 # 预热步数
  weight_decay: 0.1                 # 权重衰减
  grad_clip: 1.0                    # 梯度裁剪
  save_interval: 500                # 保存间隔
  log_interval: 10                  # 日志间隔

output:
  base_dir: results/                # 输出根目录
```

### 命令行参数覆盖

```bash
# 覆盖数据路径
python run.py train --config configs/train.yaml --data data/train.npy

# 覆盖输出目录
python run.py train --config configs/train.yaml --output ./my_results

# 强制启用 Flash Attention
python run.py train --config configs/train.yaml --flash-attn

# 覆盖任意配置项
python run.py train --config configs/train.yaml --set training.lr=1e-4
python run.py train --config configs/train.yaml --set model.num_layers=6

# 多卡训练（DDP）
torchrun --nproc_per_node=2 run.py train --config configs/train.yaml --ddp
```

### 预设配置

- `configs/train.yaml` - 默认配置（适合 8GB 显存）
- `configs/train_small.yaml` - 小模型配置（~30M 参数）
- `configs/train_demo.yaml` - 演示配置（50 步快速测试）

---

## 缩放律分析

运行 Chinchilla 缩放律实验和 IsoFLOPs 分析。

### 基本用法

```bash
python run.py scaling --config configs/scaling.yaml
```

### 配置文件说明 (`configs/scaling.yaml`)

```yaml
scaling:
  experiments_file: data/scaling_experiments.json    # 实验数据文件
  compute_budgets: [1.0e18, 1.0e19, 1.0e20, 1.0e21]  # FLOPs 预算点
  min_params: 1.0e7                                  # 最小参数量（10M）
  max_params: 1.0e10                                 # 最大参数量（10B）
  experiments_per_budget: 5                          # 每个预算点的实验数

  training_overrides:                                # 覆盖训练配置
    max_steps: 5000
    warmup_steps: 250
    save_interval: 1000
    log_interval: 50

output:
  base_dir: results/
  save_plots: true
  plot_format: png
```

### 输出

- 拟合参数：`results/scaling_params.json`
- 可视化图表：`results/plots/`

---

## 数据质量过滤

运行数据质量过滤流程（Gopher 规则 + 去重）。

### 基本用法

```bash
python run.py data --config configs/data.yaml
```

### 配置文件说明 (`configs/data.yaml`)

```yaml
data:
  input_file: "data/raw/openwebtext_train.txt"
  output_file: "data/filtered/openwebtext_filtered.txt"

filter:
  min_length: 100
  max_length: 100000
  enable_gopher_filter: true      # 启用 Gopher 质量规则
  enable_deduplication: true      # 启用重复检测

logging:
  level: "INFO"
  verbose: true
```

### 过滤统计

处理完成后会输出：
- 总文档数
- 保留文档数
- 过滤率
- 各过滤规则的统计

---

## 对齐训练

运行 SFT、DPO 或 GRPO 对齐训练。

### 基本用法

```bash
# SFT 训练
python run.py align --config configs/align.yaml --method sft

# DPO 训练
python run.py align --config configs/align.yaml --method dpo

# GRPO 训练
python run.py align --config configs/align.yaml --method grpo
```

### 配置文件说明 (`configs/align.yaml`)

```yaml
model:
  name: "results/train/7bb145fb/checkpoints/step_001000.pt"  # 模型检查点
  vocab_size: 10000
  context_length: 256
  d_model: 256
  num_layers: 4
  num_heads: 4
  d_ff: 1024

tokenizer:
  path: "results/tokenizer/bpe_tokenizer"

method: sft  # 训练方法：sft, dpo, grpo

# SFT 配置
sft:
  data_path: "data/sft_train.jsonl"
  eval_data_path: "data/sft_eval.jsonl"
  batch_size: 2
  gradient_accumulation_steps: 2
  learning_rate: 1e-4
  num_epochs: 3
  max_length: 256
  save_steps: 5
  eval_steps: 5

# DPO 配置
dpo:
  data_path: "data/dpo_train.jsonl"
  eval_data_path: "data/dpo_eval.jsonl"
  ref_model_path: "checkpoints/sft_final"
  batch_size: 2
  gradient_accumulation_steps: 2
  learning_rate: 5e-5
  num_epochs: 1
  beta: 0.1
  max_length: 256
  save_steps: 10
  eval_steps: 5

# GRPO 配置
grpo:
  data_path: "data/grpo_train.jsonl"
  eval_data_path: "data/grpo_eval.jsonl"
  batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 1e-5
  num_steps: 20
  group_size: 4
  max_new_tokens: 64
  temperature: 1.0
  epochs_per_rollout: 1
  eval_every_n_steps: 5

inference:
  backend: "hf"                   # 推理后端：hf, vllm
  device: "cuda"
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.5

output:
  base_dir: "results/align"
  run_name: null
  save_total_limit: 3

logging:
  use_wandb: false
  wandb_project: "llm-foundry-align"
  log_steps: 1
```

### HuggingFace 版本配置 (`configs/align_hf.yaml`)

使用 HuggingFace Transformers 实现的对齐训练：

```yaml
model:
  name: "gpt2"  # HF 模型名称

tokenizer:
  name: "gpt2"  # HF tokenizer 名称
```

### 命令行参数

```bash
# 覆盖训练方法
python run.py align --config configs/align.yaml --method dpo

# 覆盖模型路径
python run.py align --config configs/align.yaml --model checkpoints/my_model

# 覆盖数据路径
python run.py align --config configs/align.yaml --data data/my_data.jsonl

# 覆盖输出目录
python run.py align --config configs/align.yaml --output ./my_outputs

# 使用 HuggingFace 实现
python run.py align --config configs/align_hf.yaml --hf
```

---

## 完整流程示例

```bash
# 1. 检查环境
python run.py env

# 2. 生成训练数据
python run.py datagen --config configs/datagen.yaml

# 3. 训练 tokenizer
python run.py tokenize --config configs/tokenize.yaml

# 4. 预训练模型
python run.py train --config configs/train.yaml

# 5. SFT 对齐
python run.py align --config configs/align.yaml --method sft

# 6. DPO 对齐
python run.py align --config configs/align.yaml --method dpo
```

---

## 故障排除

### CUDA 内存不足

- 减小 `batch_size`
- 增大 `gradient_accumulation_steps`
- 减小 `context_length` 或模型尺寸
- 使用 `configs/train_small.yaml`

### Tokenizer 错误

- 确保已运行 tokenize 步骤
- 检查 `tokenizer.path` 配置是否正确

### 数据文件不存在

- 检查 `data_path` 路径是否正确
- 确保已运行前置步骤（如 tokenize）

### Flash Attention 不可用

- 系统会自动 fallback 到 SDPA
- 如需使用 Flash Attention，请安装 `flash-attn` 包
