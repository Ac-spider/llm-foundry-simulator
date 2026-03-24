# LLM Foundry Simulator 训练复现指南

> 基于 2025-03-24 的成功训练流程记录

## 环境信息

- **虚拟环境**: `/home/lvx/triton_env`
- **Python**: `/home/lvx/triton_env/bin/python`
- **显存**: 8GB
- **工作目录**: `/mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator`

---

## 快速开始（一键执行）

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
PYTHON=/home/lvx/triton_env/bin/python

# 1. 检查环境
$PYTHON run.py env

# 2. 下载数据（1万条样本，约25MB）
$PYTHON scripts/download_openwebtext.py --max-samples 10000

# 3. 训练 Tokenizer
$PYTHON run.py tokenize --config configs/tokenize.yaml

# 4. 训练模型（demo配置，50步，约5分钟）
$PYTHON run.py train --config configs/train_demo.yaml
```

---

## 详细步骤

### 1. 环境检查

```bash
/home/lvx/triton_env/bin/python run.py env
```

预期输出：
- Python 版本
- PyTorch 版本和 CUDA 可用性
- GPU 信息

---

### 2. 数据准备

下载 OpenWebText 数据集：

```bash
/home/lvx/triton_env/bin/python scripts/download_openwebtext.py \
    --max-samples 10000 \
    --output-dir data/raw
```

**参数说明**：
- `--max-samples`: 下载样本数（默认 100000，demo用 10000 足够）
- `--output-dir`: 输出目录

**输出文件**：
- `data/raw/openwebtext_train.txt` (~25MB)

---

### 3. Tokenizer 训练

```bash
/home/lvx/triton_env/bin/python run.py tokenize --config configs/tokenize.yaml
```

**关键配置**（`configs/tokenize.yaml`）：
- `vocab_size: 10000`
- `train_file: data/raw/openwebtext_train.txt`
- `output_dir: results/tokenizer`

**输出文件**：
- `results/tokenizer/bpe_tokenizer/` - Tokenizer 目录
- `data/train_tokens.bin` - 二进制编码数据 (~51MB)

---

### 4. 模型训练

#### Demo 配置（快速验证）

```bash
/home/lvx/triton_env/bin/python run.py train --config configs/train_demo.yaml
```

**配置参数**（`configs/train_demo.yaml`）：
| 参数 | 值 | 说明 |
|------|-----|------|
| d_model | 256 | 模型维度 |
| n_layers | 4 | 层数 |
| n_heads | 4 | 注意力头数 |
| batch_size | 8 | 批次大小 |
| max_steps | 50 | 训练步数 |
| lr | 3e-4 | 学习率 |

**显存占用**: ~2GB（适合8GB显卡）

#### 更大规模的训练

如需更长时间训练，可创建自定义配置：

```bash
# 复制 demo 配置并修改
sed 's/max_steps: 50/max_steps: 1000/' configs/train_demo.yaml > configs/train_1k.yaml
/home/lvx/triton_env/bin/python run.py train --config configs/train_1k.yaml
```

#### 2000步训练（推荐）

使用专门配置进行更充分的训练：

```bash
/home/lvx/triton_env/bin/python run.py train --config configs/train_2k.yaml
```

**配置参数**（`configs/train_2k.yaml`）：
| 参数 | 值 | 说明 |
|------|-----|------|
| max_steps | 2000 | 训练步数 |
| warmup_steps | 100 | 预热步数 |
| save_interval | 500 | 每500步保存检查点 |
| log_interval | 50 | 每50步输出日志 |
| lr | 3e-4 | 学习率 |

**训练结果示例**：
```
[train] step=50, loss=8.5469, lr=0.000147
[train] step=500, loss=6.0675, lr=0.000272  ✅ 检查点1
[train] step=1000, loss=5.9207, lr=0.000176 ✅ 检查点2
[train] step=1500, loss=5.4308, lr=0.000074 ✅ 检查点3
[train] step=2000, loss=5.5083, lr=0.000030 ✅ 最终模型
```

**关键指标**：
- 初始损失: ~8.55
- 最终损失: ~5.51
- 损失降低: ~36%
- 训练时间: ~10-15分钟

---

### 5. 学习率对比实验

为找到最优学习率，可同时运行多个不同学习率的实验：

#### 创建实验配置

```bash
# 创建不同学习率的配置文件
# 配置文件已创建: configs/train_2k_lr1e4.yaml, train_2k_lr5e4.yaml, train_2k_lr1e3.yaml
```

**配置对比**：

| 配置文件 | 学习率 | 最小学习率 | 策略 |
|----------|--------|------------|------|
| `train_2k_lr1e4.yaml` | 1e-4 | 1e-5 | 保守 |
| `train_2k.yaml` | 3e-4 | 3e-5 | 基准 |
| `train_2k_lr5e4.yaml` | 5e-4 | 5e-5 | 激进 |
| `train_2k_lr1e3.yaml` | 1e-3 | 1e-4 | 非常激进 |

#### 运行对比实验

```bash
# 并行运行多个实验（在不同的终端窗口中）

# 实验1: 保守学习率
/home/lvx/triton_env/bin/python run.py train --config configs/train_2k_lr1e4.yaml

# 实验2: 激进学习率
/home/lvx/triton_env/bin/python run.py train --config configs/train_2k_lr5e4.yaml

# 实验3: 非常激进学习率
/home/lvx/triton_env/bin/python run.py train --config configs/train_2k_lr1e3.yaml
```

#### 实验结果对比

| 学习率 | Step 50 | Step 500 | Step 1000 | Step 1500 | **Step 2000** |
|--------|---------|----------|-----------|-----------|---------------|
| **1e-4** | 9.05 | 6.77 | 6.28 | 6.05 | **6.25** |
| **3e-4** | 8.55 | 6.07 | 5.92 | 5.43 | **5.51** |
| **5e-4** | 8.10 | 5.86 | 5.46 | 5.34 | **5.08** |
| **1e-3** | 7.49 | 5.69 | 5.23 | 4.88 | **4.86** ⭐ |

#### 关键发现

**1. 学习率与最终损失**
- 学习率越大 → 最终损失越低
- `lr=1e-3` 比 `lr=1e-4` 损失低 1.39 (22% 改进)

**2. 收敛速度**
- `lr=1e-3` 在 500 步就达到 5.69（比 `lr=1e-4` 的 2000 步还好）
- 高学习率前期下降更快

**3. 稳定性**
- `lr=1e-4` 后期损失波动较小，更稳定
- `lr=1e-3` 在 500 步后损失出现震荡（可能不稳定）

#### 建议

**最佳学习率选择**：
- **稳定收敛**: `5e-4`（推荐，平衡性能和稳定性）
- **最低损失**: `1e-3`（注意监控过拟合）
- **保守训练**: `1e-4`（适合长文本或大数据集）

**输出目录**：
```
results/
├── train_2k_lr1e4/{hash}/checkpoints/step_002000.pt  (loss=6.25)
├── train_2k_lr5e4/{hash}/checkpoints/step_002000.pt  (loss=5.08)
└── train_2k_lr1e3/{hash}/checkpoints/step_002000.pt  (loss=4.86) ⭐ 最佳
```

---

## 训练结果

### 输出目录结构

```
results/train_demo/{hash}/
├── checkpoints/
│   └── step_000050.pt      # 最终模型检查点
├── config.yaml             # 实际使用的配置
└── logs/
    └── train.log          # 训练日志
```

### 示例训练输出

```
Loaded 31,784,354 tokens from data/train_tokens.bin
number of non-embedding parameters: 4.20M

[train] step=10, loss=9.1134, lr=0.000270
[train] step=20, loss=8.5881, lr=0.000268
[train] step=30, loss=8.1838, lr=0.000176
[train] step=40, loss=7.9834, lr=0.000077
[train] step=50, loss=7.8892, lr=0.000030

Saved checkpoint: results/train_demo/6af0f313/checkpoints/step_000050.pt
```

**关键指标**：
- 参数: 4.20M（非嵌入）
- 损失: 9.11 → 7.89（降低13%）
- 数据: 3200万 tokens

---

## 故障排除

### 1. Python 命令不存在

**问题**: `python: command not found`

**解决**: 使用完整路径：
```bash
/home/lvx/triton_env/bin/python run.py ...
```

### 2. 缺少依赖

**问题**: `ModuleNotFoundError: No module named 'torch'`

**解决**: 确保在 triton 虚拟环境中运行（见上文）

### 3. CUDA 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决**: 减小 batch_size 或模型尺寸：
```bash
# 使用命令行覆盖
/home/lvx/triton_env/bin/python run.py train \
    --config configs/train_demo.yaml \
    --set training.batch_size=4
```

### 4. 数据文件不存在

**问题**: `FileNotFoundError: data/train_tokens.bin`

**解决**: 确保已运行 tokenize 步骤

---

## 下一步

1. **文本生成** - 使用训练好的模型生成文本
2. **继续训练** - 加载检查点继续训练
3. **完整训练** - 使用更大的配置（`configs/train.yaml`）
4. **对齐训练** - 运行 SFT/DPO/GRPO

---

## 参考链接

- 项目文档: `docs/USAGE.md`
- API 文档: `docs/API.md`
- 配置示例: `configs/`

---

*记录时间: 2025-03-24*
*更新记录:*
- ✅ 50步训练完成（loss: 9.11→7.89）
- ✅ 2000步训练完成（loss: 8.55→5.51）
- ✅ 学习率对比实验完成（4组学习率，最优: lr=1e-3, loss=4.86）
