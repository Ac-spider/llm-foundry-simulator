# LLM Foundry Simulator

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/pytorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

LLM Foundry Simulator 是一个统一的命令行工具，用于模拟和运行大型语言模型训练工作流。该项目整合了 Stanford CS336 课程的所有作业内容，提供了一个完整的 LLM 训练流水线。

## 项目概述

LLM Foundry Simulator 实现了从数据生成到模型对齐的完整 LLM 训练流程：

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LLM Foundry Simulator Pipeline                       │
├─────────────┬─────────────┬─────────────┬─────────────┬─────────────────────┤
│  Stage 0    │  Stage 1    │  Stage 2    │  Stage 3    │  Stage 4 & 5        │
│  DataGen    │  Tokenize   │   Train     │   Scaling   │  Data & Align       │
├─────────────┼─────────────┼─────────────┼─────────────┼─────────────────────┤
│ • Wikipedia │ • BPE       │ • Pretrain  │ • Chinchilla│ • Quality filtering │
│ • BPE train │ • Pretoken  │ • DDP       │ • IsoFLOPs  │ • SFT               │
│ • SFT gen   │ • Binarize  │ • FlashAttn │ • Scaling   │ • DPO/GRPO          │
└─────────────┴─────────────┴─────────────┴─────────────┴─────────────────────┘
```

### 核心特性

- **统一 CLI**: 单一入口点 `run.py` 管理所有阶段
- **后端抽象**: 自动选择最佳注意力实现 (Triton → torch.compile → SDPA)
- **配置驱动**: YAML 配置文件支持哈希验证和缓存
- **分布式训练**: 支持 DDP 和 ZeRO-1 分片优化器
- **Scaling Laws**: 实现 Chinchilla 缩放定律和 IsoFLOPs 实验
- **对齐方法**: 支持 SFT、DPO、PPO、GRPO 等对齐技术

## 安装说明

### 环境要求

- Python 3.10+
- PyTorch 2.x
- CUDA 11.8+ (用于 GPU 训练)

### 安装步骤

1. **克隆仓库**

```bash
git clone <repository-url>
cd LLM_Foundry_Simulator
```

2. **创建虚拟环境**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**

```bash
pip install -e .
```

4. **安装可选依赖**

```bash
# 开发依赖
pip install -e ".[dev]"

# 高性能注意力 (推荐)
pip install triton

# vLLM 推理后端
pip install vllm
```

### 环境检查

```bash
python run.py env
```

## 快速开始指南

### 1. 数据生成

```bash
# 下载 Wikipedia 数据并训练 BPE tokenizer
python run.py datagen --config configs/datagen.yaml
```

### 2. 数据预处理

```bash
# Tokenize 和 binarize 数据
python run.py tokenize --config configs/tokenize.yaml
```

### 3. 模型训练

```bash
# 单卡训练
python run.py train --config configs/train.yaml

# 分布式训练
python -m torch.distributed.run --nproc_per_node=4 run.py train --config configs/train.yaml
```

### 4. 模型推理

```bash
# 使用训练好的模型生成文本
python run.py generate --model_path outputs/model.pt --prompt "Once upon a time"
```

## 各阶段使用说明

### Stage 0: 数据生成 (`llm_foundry/stage0_datagen/`)

数据生成模块负责准备训练数据：

```python
from llm_foundry.stage0_datagen.datagen import download_wikipedia, train_bpe_tokenizer

# 下载 Wikipedia 数据
download_wikipedia(output_dir="data/wikipedia", language="en")

# 训练 BPE tokenizer
train_bpe_tokenizer(
    input_files=["data/wikipedia/train.txt"],
    vocab_size=32000,
    output_path="tokenizers/bpe_32k.model"
)
```

### Stage 1: Tokenization (`llm_foundry/stage1_tokenize/`)

Tokenization 模块将文本转换为模型可处理的二进制格式：

```bash
# Pretokenization - 使用 BPE 编码文本
python run.py tokenize pretokenize \
    --input data/wikipedia/train.txt \
    --output data/train.tokens \
    --tokenizer tokenizers/bpe_32k.model

# Binarization - 转换为内存映射格式
python run.py tokenize binarize \
    --input data/train.tokens \
    --output data/train.bin
```

### Stage 2: 训练 (`llm_foundry/stage2_train/`)

训练模块支持多种配置：

```yaml
# configs/train.yaml
model:
  vocab_size: 50257
  context_length: 1024
  d_model: 768
  num_layers: 12
  num_heads: 12
  d_ff: 3072

training:
  batch_size: 32
  learning_rate: 3e-4
  max_iters: 100000
  warmup_iters: 2000

system:
  device: cuda
  compile: true
  use_flash_attn: true
```

### Stage 3: Scaling Laws (`llm_foundry/stage3_scaling/`)

运行 IsoFLOPs 实验来研究模型缩放：

```bash
# 运行 IsoFLOPs 实验
python run.py scaling --config configs/scaling.yaml

# 可视化结果
python run.py scaling plot --results outputs/scaling_results.json
```

### Stage 4: 数据质量 (`llm_foundry/stage4_data/`)

数据质量管道：

```bash
# 运行质量过滤管道
python run.py data pipeline \
    --input data/raw/ \
    --output data/filtered/ \
    --config configs/data_quality.yaml
```

### Stage 5: 对齐 (`llm_foundry/stage5_align/`)

支持多种对齐方法：

```bash
# SFT (Supervised Fine-Tuning)
python run.py align sft --config configs/sft.yaml

# DPO (Direct Preference Optimization)
python run.py align dpo --config configs/dpo.yaml

# GRPO (Group Relative Policy Optimization)
python run.py align grpo --config configs/grpo.yaml
```

## 配置文件示例

### 基础训练配置

```yaml
# configs/train_small.yaml
model:
  vocab_size: 50257
  context_length: 512
  d_model: 384
  num_layers: 6
  num_heads: 6
  d_ff: 1536
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 5e-4
  min_lr: 5e-5
  weight_decay: 0.01
  max_iters: 50000
  warmup_iters: 1000
  eval_interval: 500
  eval_iters: 50

system:
  device: cuda
  compile: true
  use_flash_attn: true

paths:
  data_path: data/train.bin
  output_dir: outputs/small_model
```

### 分布式训练配置

```yaml
# configs/train_ddp.yaml
model:
  vocab_size: 50257
  context_length: 1024
  d_model: 768
  num_layers: 12
  num_heads: 12
  d_ff: 3072

training:
  batch_size: 64  # 全局批次大小
  learning_rate: 3e-4
  max_iters: 100000

system:
  device: cuda
  compile: true
  use_flash_attn: true
  sharded_optimizer: true  # 启用 ZeRO-1

distributed:
  backend: nccl
  find_unused_parameters: false

paths:
  data_path: data/train.bin
  output_dir: outputs/ddp_model
```

### Scaling 实验配置

```yaml
# configs/scaling.yaml
isoflops:
  target_flops:
    - 1e18
    - 1e19
    - 1e20
  model_sizes:
    - d_model: 256
      num_layers: 4
      num_heads: 4
    - d_model: 512
      num_layers: 8
      num_heads: 8
    - d_model: 768
      num_layers: 12
      num_heads: 12

training:
  tokens_per_step: 524288  # 512 * 1024
  learning_rate: 3e-4

paths:
  data_path: data/train.bin
  output_dir: outputs/scaling
```

## 项目结构说明

```
LLM_Foundry_Simulator/
├── llm_foundry/              # 主代码库
│   ├── __init__.py
│   ├── backends/             # 硬件后端抽象
│   │   ├── attention.py      # 注意力机制 (Triton/torch.compile/SDPA)
│   │   └── inference.py      # 推理后端 (vLLM/HF)
│   ├── common/               # 共享组件
│   │   ├── model.py          # Transformer 模型
│   │   ├── optimizer.py      # AdamW + ZeRO-1
│   │   ├── data.py           # 数据加载
│   │   ├── config.py         # 配置管理
│   │   ├── env_check.py      # 环境检测
│   │   └── hashing.py        # 哈希验证
│   ├── stage0_datagen/       # 数据生成
│   ├── stage1_tokenize/      # Tokenization
│   ├── stage2_train/         # 训练
│   ├── stage3_scaling/       # Scaling laws
│   ├── stage4_data/          # 数据管道
│   └── stage5_align/         # 对齐
├── configs/                  # 配置文件
├── tests/                    # 测试套件
├── scripts/                  # 实用脚本（数据下载等）
├── reproduce/                # 复现脚本与预期输出
├── docs/                     # 文档
│   └── plans/                # 实现计划
├── run.py                    # 主入口
├── pyproject.toml            # 项目配置
└── README.md                 # 本文件
```

> CS336 原始作业参考实现见：[Stanford-CS336-LvxSeraph](https://github.com/Ac-spider/Stanford-CS336-LvxSeraph)

## 开发指南

### 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定模块测试
pytest tests/test_attention.py -v
pytest tests/stage2_train/ -v

# 运行带覆盖率报告
pytest tests/ --cov=llm_foundry --cov-report=html
```

### 代码风格

项目使用 Black 和 isort 进行代码格式化：

```bash
# 格式化代码
black llm_foundry/ tests/
isort llm_foundry/ tests/

# 类型检查
mypy llm_foundry/
```

### 添加新功能

1. **新 Stage**: 在 `llm_foundry/` 下创建 `stageX_name/` 目录
2. **新后端**: 在 `llm_foundry/backends/` 添加实现
3. **新配置**: 在 `configs/` 添加 YAML 示例
4. **新测试**: 在 `tests/` 添加对应测试

### 调试技巧

```bash
# 启用详细日志
LOG_LEVEL=DEBUG python run.py train --config configs/train.yaml

# 检查环境
python run.py env --verbose

# 测试注意力后端
python -c "from llm_foundry.backends.attention import get_attention_backend_name; print(get_attention_backend_name())"
```

## 技术栈

- **PyTorch 2.x**: 深度学习框架
- **einops**: 张量操作
- **jaxtyping**: 类型注解
- **PyYAML**: 配置管理
- **pytest**: 测试框架
- **Triton** (可选): 高性能 GPU 内核
- **vLLM** (可选): 高吞吐量推理

## 参考资源

- [Stanford CS336](https://stanford-cs336.github.io/): 大语言模型课程
- [Stanford CS336 作业实现](https://github.com/Ac-spider/Stanford-CS336-LvxSeraph): 本项目基于该仓库整合的 CS336 五个作业实现（Assignment 1-5）
- [Chinchilla Paper](https://arxiv.org/abs/2203.15556): 训练计算最优的大型语言模型
- [FlashAttention-2](https://arxiv.org/abs/2307.08691): 更快的注意力机制

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 贡献

欢迎提交 Issue 和 PR！请确保：

1. 代码通过所有测试
2. 遵循 Black 代码风格
3. 添加必要的文档和注释
4. 使用 git trailers 提交信息（详见 OMC commit protocol）

## 致谢

本项目基于 Stanford CS336 课程作业开发，感谢课程团队提供的优秀教学资源。
