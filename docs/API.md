# LLM Foundry Simulator API 快速参考

本文档提供主要 API 的快速参考，便于开发时查阅。

## 目录

- [配置管理](#配置管理)
- [模型](#模型)
- [数据加载](#数据加载)
- [优化器](#优化器)
- [后端](#后端)
- [分词器](#分词器)
- [训练器](#训练器)

---

## 配置管理

### `llm_foundry.common.config`

#### `load_config(config_path)`

加载 YAML 配置文件。

```python
from llm_foundry.common.config import load_config

config = load_config("configs/train.yaml")
print(config.model.d_model)  # 256
```

**参数：**
- `config_path`: str | Path - YAML 文件路径

**返回：**
- SimpleNamespace - 支持属性访问的配置对象

---

#### `load_config_with_hash(config_path)`

加载配置并计算哈希值（用于缓存）。

```python
from llm_foundry.common.config import load_config_with_hash

config, config_hash = load_config_with_hash("configs/train.yaml")
```

**返回：**
- Tuple[SimpleNamespace, str] - (配置对象, 哈希字符串)

---

#### `namespace_to_dict(ns)`

将 SimpleNamespace 转换为字典。

```python
from llm_foundry.common.config import namespace_to_dict

cfg_dict = namespace_to_dict(config)
```

---

#### `merge_configs(base, override)`

合并两个配置，override 优先级更高。

```python
from llm_foundry.common.config import merge_configs

merged = merge_configs(base_config, override_config)
```

---

#### `TrainConfig`

训练配置数据类。

```python
from llm_foundry.common.config import TrainConfig

cfg = TrainConfig(
    vocab_size=10000,
    context_length=256,
    d_model=256,
    num_layers=4,
    num_heads=4,
    batch_size=16,
    learning_rate=3e-4,
)
```

**主要字段：**
| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| vocab_size | int | 50257 | 词表大小 |
| context_length | int | 1024 | 上下文长度 |
| d_model | int | 768 | 模型维度 |
| num_layers | int | 12 | 层数 |
| num_heads | int | 12 | 注意力头数 |
| d_ff | int | 3072 | FFN 维度 |
| batch_size | int | 32 | 批次大小 |
| learning_rate | float | 3e-4 | 学习率 |
| max_iters | int | 100000 | 最大迭代次数 |

---

## 模型

### `llm_foundry.common.model`

#### `ModelConfig`

模型配置数据类。

```python
from llm_foundry.common.model import ModelConfig

cfg = ModelConfig(
    vocab_size=10000,
    context_length=256,
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=1024,
    rope_theta=10000.0,
    dropout=0.0,
)
```

---

#### `create_model(config, use_flash_attn=True)`

创建 Transformer 模型。

```python
from llm_foundry.common.model import ModelConfig, create_model

cfg = ModelConfig(vocab_size=10000, d_model=256, num_layers=4)
model = create_model(cfg, use_flash_attn=True)
```

**参数：**
- `config`: ModelConfig - 模型配置
- `use_flash_attn`: bool - 是否使用 Flash Attention

**返回：**
- BasicsTransformerLM - 初始化后的模型

---

#### `BasicsTransformerLM`

Transformer 语言模型类。

```python
from llm_foundry.common.model import BasicsTransformerLM

model = BasicsTransformerLM(
    vocab_size=10000,
    context_length=256,
    d_model=256,
    num_layers=4,
    num_heads=4,
    d_ff=1024,
    rope_theta=10000.0,
)

# 前向传播
logits = model(input_ids)

# 生成文本
output_ids = model.generate(
    input_ids,
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
    eos_token_id=50256,
)

# 获取参数量
n_params = model.get_num_params(non_embedding=True)
```

**方法：**
| 方法 | 说明 |
|------|------|
| `forward(x)` | 前向传播，返回 logits |
| `generate(x, max_new_tokens, ...)` | 自回归文本生成 |
| `get_num_params(non_embedding=True)` | 获取参数量 |
| `from_pretrained(path)` | 从目录加载预训练模型 |

---

## 数据加载

### `llm_foundry.common.data`

#### `get_batch(dataset, batch_size, context_length, device)`

从内存映射数据集采样一个批次。

```python
from llm_foundry.common.data import get_batch
import numpy as np

dataset = np.memmap("data/tokens.bin", dtype=np.uint16, mode="r")
x, y = get_batch(dataset, batch_size=16, context_length=256, device="cuda")
# x: (batch_size, context_length) - 输入
# y: (batch_size, context_length) - 目标（x 偏移一位）
```

**参数：**
- `dataset`: np.ndarray - 内存映射的 token 数组
- `batch_size`: int - 批次大小
- `context_length`: int - 序列长度
- `device`: str - 目标设备

**返回：**
- Tuple[Tensor, Tensor] - (输入, 目标)

---

#### `load_tokens(data_path, dtype=np.uint16)`

加载分词后的数据文件。

```python
from llm_foundry.common.data import load_tokens

tokens = load_tokens("data/train_tokens.bin", dtype=np.uint16)
print(f"Total tokens: {len(tokens):,}")
```

---

#### `create_data_loader(data_path, batch_size, context_length, device, num_batches=None)`

创建数据加载器（无限迭代）。

```python
from llm_foundry.common.data import create_data_loader

loader = create_data_loader(
    data_path="data/train_tokens.bin",
    batch_size=16,
    context_length=256,
    device="cuda",
)

for x, y in loader:
    # 训练步骤
    pass
```

---

## 优化器

### `llm_foundry.common.optimizer`

#### `create_optimizer(params, lr, weight_decay=0.1)`

创建 AdamW 优化器。

```python
from llm_foundry.common.optimizer import create_optimizer

optimizer = create_optimizer(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1,
)
```

---

#### `ShardedOptimizer`

分片优化器（用于大模型训练）。

```python
from llm_foundry.common.optimizer import ShardedOptimizer

optimizer = ShardedOptimizer(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1,
)
```

---

#### `get_cosine_lr(step, warmup_steps, max_steps, lr, min_lr)`

计算余弦退火学习率。

```python
from llm_foundry.common.optimizer import get_cosine_lr

current_lr = get_cosine_lr(
    step=1000,
    warmup_steps=500,
    max_steps=10000,
    lr=3e-4,
    min_lr=3e-5,
)
```

---

## 后端

### `llm_foundry.backends.attention`

#### `get_attention_fn(use_flash_attn=True)`

获取注意力函数（自动 fallback）。

```python
from llm_foundry.backends.attention import get_attention_fn

attn_fn = get_attention_fn(use_flash_attn=True)
# 优先级: Triton Flash Attention -> torch.compile SDPA -> F.sdpa

output = attn_fn(Q, K, V, is_causal=True)
```

---

### `llm_foundry.backends.inference`

#### `get_inference_backend(backend_type, ...)`

获取推理后端。

```python
from llm_foundry.backends.inference import get_inference_backend, GenerationConfig

backend = get_inference_backend(
    backend_type="auto",  # "auto", "vllm", "hf"
    model_name_or_path="gpt2",
    device="cuda",
)

gen_config = GenerationConfig(
    max_new_tokens=100,
    temperature=1.0,
    top_k=50,
)

output = backend.generate(prompt, gen_config)
```

---

## 分词器

### `llm_foundry.stage1_tokenize`

#### `BPETokenizer`

BPE 分词器。

```python
from llm_foundry.stage1_tokenize import BPETokenizer

# 训练 tokenizer
tokenizer = BPETokenizer.train(
    input_path="data/train.txt",
    vocab_size=10000,
    special_tokens=["<|endoftext|>", "<|padding|>"],
)

# 编码/解码
token_ids = tokenizer.encode("Hello world")
text = tokenizer.decode(token_ids)

# 保存/加载
tokenizer.save("results/tokenizer/bpe_tokenizer")
tokenizer = BPETokenizer.load("results/tokenizer/bpe_tokenizer")

# 获取词表大小
vocab_size = len(tokenizer)
```

---

## 训练器

### `llm_foundry.stage2_train.trainer`

#### `Trainer`

主训练器类。

```python
from llm_foundry.stage2_train.trainer import Trainer

trainer = Trainer(config_dict)
trainer.train()
```

---

### `llm_foundry.stage5_align`

#### `SFTTrainer`

监督微调训练器。

```python
from llm_foundry.stage5_align import SFTTrainer, SFTDataset

dataset = SFTDataset(data_path, tokenizer, max_length=256)
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    device="cuda",
    config=config,
)
metrics = trainer.train_epoch(dataloader)
```

---

#### `DPOTrainer`

直接偏好优化训练器。

```python
from llm_foundry.stage5_align import DPOTrainer, DPODataset

dataset = DPODataset(data_path, tokenizer)
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    optimizer=optimizer,
    beta=0.1,
    device="cuda",
    config=config,
)
trainer.train(dataloader, output_dir="./outputs")
```

---

#### `GRPOTrainer`

组相对策略优化训练器。

```python
from llm_foundry.stage5_align import GRPOTrainer, PromptDataset
from llm_foundry.backends.inference import get_inference_backend

dataset = PromptDataset(data_path)
inference_backend = get_inference_backend("hf", model=model, tokenizer=tokenizer)

trainer = GRPOTrainer(
    model=model,
    ref_model=ref_model,
    optimizer=optimizer,
    inference_backend=inference_backend,
    reward_fn=reward_fn,
    group_size=4,
    device="cuda",
    config=config,
)
trainer.train(dataset, output_dir="./outputs")
```

---

## 数据生成

### `llm_foundry.stage0_datagen`

#### `run_datagen(config)`

运行数据生成。

```python
from llm_foundry.stage0_datagen import DataGenConfig, run_datagen

cfg = DataGenConfig.from_yaml("configs/datagen.yaml")
result = run_datagen(cfg)
# result: {'sft': 500, 'grpo': 500}
```

---

## 缩放律

### `llm_foundry.stage3_scaling`

#### `ScalingAnalyzer`

缩放律分析器。

```python
from llm_foundry.stage3_scaling import ScalingAnalyzer

analyzer = ScalingAnalyzer(config_dict)
result = analyzer.run(experiments_data)

# result 包含：
# - chinchilla: 拟合参数 (E, A, alpha, B, beta)
# - isoflops_optimal: 最优配置列表
# - plots_dir: 图表目录
```

---

## 数据管道

### `llm_foundry.stage4_data`

#### `DataPipeline`

数据质量过滤管道。

```python
from llm_foundry.stage4_data import DataPipeline, DataPipelineConfig

cfg = DataPipelineConfig(
    min_length=100,
    max_length=100000,
    enable_gopher_filter=True,
    enable_deduplication=True,
)

pipeline = DataPipeline(cfg)
stats = pipeline.process_file(
    input_path="input.txt",
    output_path="output.txt",
    doc_separator="\n\n",
)
# stats: {'total': 1000, 'kept': 800, ...}
```

---

## 环境检查

### `llm_foundry.common.env_check`

#### `check_environment()`

检查运行环境。

```python
from llm_foundry.common.env_check import check_environment

status = check_environment()
status.print_report()

# 属性：
# - status.cuda_available: bool
# - status.gpu_count: int
# - status.gpu_name: str
# - status.torch_version: str
```

---

## 哈希工具

### `llm_foundry.common.hashing`

#### `compute_config_hash(config_dict)`

计算配置哈希。

```python
from llm_foundry.common.hashing import compute_config_hash

config_hash = compute_config_hash(config_dict)
```

---

#### `compute_file_hash(file_path)`

计算文件哈希。

```python
from llm_foundry.common.hashing import compute_file_hash

file_hash = compute_file_hash("data.txt")
```
