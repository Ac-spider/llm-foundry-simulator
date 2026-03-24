# 项目代码与 Reference 代码详细对比报告

## 概述

本报告详细对比 LLM Foundry Simulator 项目代码与 Stanford CS336 课程 Assignment 参考代码之间的差异。

### Reference 代码来源

| Assignment | 目录 | 内容 |
|-----------|------|------|
| Assignment1-basics | `reference_resource/Assignment1-basics/` | BPE Tokenizer, Transformer基础, 训练循环 |
| Assignment2-system | `reference_resource/Assignment2-system/` | 系统优化 (Triton, DDP, ZeRO) |
| Assignment3-scaling | `reference_resource/Assignment3-scaling/` | Chinchilla缩放定律 |
| Assignment4-data | `reference_resource/Assignment4-data/` | 数据管道, 质量过滤 |
| Assignment5-alignment | `reference_resource/Assignment5-alignment/` | SFT, DPO, PPO, GRPO |

---

## 一、Assignment1-basics 对比

### 1.1 BPE Tokenizer

| 属性 | Reference (bpe.py) | 项目 (stage1_tokenize/tokenizer.py) |
|------|-------------------|-----------------------------------|
| **文件位置** | `Assignment1-basics/Transformer/bpe.py` | `llm_foundry/stage1_tokenize/tokenizer.py` |
| **状态** | 重构并增强 | 重构并增强 |
| **类名** | 无类，纯函数 | `BPETokenizer` 类 |

**函数/类映射：**

| Reference 函数 | 项目对应 | 状态 |
|---------------|---------|------|
| `find_chunk_boundaries()` | `find_chunk_boundaries()` | 完整迁移 |
| `_process_chunk()` | `_process_chunk()` | 完整迁移 |
| `train_bpe()` | `BPETokenizer.train()` | 重构为类方法 |
| `encode()` | `BPETokenizer.encode()` | 重构为类方法 |
| `decode()` | `BPETokenizer.decode()` | 重构为类方法 |

**新增功能：**
- `BPETokenizer` 类封装，支持 save/load
- `TokenizerConfig` 配置类，支持 YAML 加载
- 完整的 docstring 文档
- `__len__()` 和 `__repr__()` 方法

**修改/重构：**
- 从纯函数式改为面向对象设计
- 添加了类型注解
- 代码组织更清晰

---

### 1.2 Tokenizer (tokenizer.py)

| 属性 | Reference (tokenizer.py) | 项目 (stage1_tokenize/tokenizer.py) |
|------|-------------------------|-----------------------------------|
| **文件位置** | `Assignment1-basics/Transformer/tokenizer.py` | `llm_foundry/stage1_tokenize/tokenizer.py` |
| **状态** | 合并到 BPE | 合并到 BPE |

Reference 中的 `Tokenizer` 类被合并到项目的 `BPETokenizer` 中。

---

### 1.3 Model

| 属性 | Reference (model.py) | 项目 (common/model.py) |
|------|---------------------|----------------------|
| **文件位置** | `Assignment1-basics/Transformer/model.py` | `llm_foundry/common/model.py` |
| **状态** | 增强 | 增强 |

**类/函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| `AttentionHead` | `AttentionHead` | 完整迁移 |
| `MultiHeadAttention` | `MultiHeadAttention` | 完整迁移 |
| `FeedForward` | `FeedForward` | 完整迁移 |
| `TransformerBlock` | `TransformerBlock` | 重构 |
| `BasicsTransformerLM` | `BasicsTransformerLM` | 增强 |

**修改/重构：**
- `TransformerBlock` 添加了 `use_flash_attn` 参数支持
- `BasicsTransformerLM` 添加了 RoPE 位置编码选项
- 添加了 `ModelConfig` dataclass 配置

**新增功能：**
- `apply_rope()` 函数 (Rotary Position Embedding)
- `create_model()` 工厂函数
- 完整的类型注解和文档

---

### 1.4 Generate

| 属性 | Reference (generate.py) | 项目 (backends/inference.py) |
|------|------------------------|----------------------------|
| **文件位置** | `Assignment1-basics/Transformer/generate.py` | `llm_foundry/backends/inference.py` |
| **状态** | 重构 | 重构 |

**函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| `generate()` | `HFInferenceBackend.generate()` | 重构为类方法 |

**修改/重构：**
- 从函数改为 `InferenceBackend` 抽象基类
- 支持两种后端：vLLM 和 HuggingFace

**新增功能：**
- `InferenceBackend` 抽象基类
- `HFInferenceBackend` HuggingFace 实现
- `VLLMInferenceBackend` vLLM 实现
- `GenerationConfig` 配置类
- `get_inference_backend()` 工厂函数

---

### 1.5 Train

| 属性 | Reference (train.py) | 项目 (stage2_train/trainer.py) |
|------|---------------------|------------------------------|
| **文件位置** | `Assignment1-basics/Transformer/train.py` | `llm_foundry/stage2_train/trainer.py` |
| **状态** | 重构 | 重构 |

**函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| `train_step()` | `Trainer.train_step()` | 重构为类方法 |
| 训练循环 | `Trainer.train_epoch()` | 重构为类方法 |

**修改/重构：**
- 从脚本改为 `Trainer` 类
- 支持梯度累积
- 支持混合精度训练

**新增功能：**
- `Trainer` 类封装
- 支持 checkpoint 保存/加载
- 支持 validation
- 完整的配置管理

---

### 1.6 Pretokenization Example

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment1-basics/Transformer/pretokenization_example.py` | 无直接对应 |
| **状态** | 未实现 | - |

这是一个示例脚本，项目中没有直接对应。功能被整合到 `binarize.py` 和 `tokenizer.py` 中。

---

### 1.7 Read PKL

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment1-basics/Transformer/read_pkl.py` | 无直接对应 |
| **状态** | 未实现 | - |

这是一个调试工具脚本，项目中没有直接对应。

---

### 1.8 Run BPE Experiments

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment1-basics/Transformer/run_bpe_experiments.py` | 无直接对应 |
| **状态** | 未实现 | - |

这是一个实验脚本，项目中没有直接对应。

---

### 1.9 Run Tokenizer Experiments

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment1-basics/Transformer/run_tokenizer_experiments.py` | 无直接对应 |
| **状态** | 未实现 | - |

这是一个实验脚本，项目中没有直接对应。

---

## 二、Assignment2-system 对比

### 2.1 cs336-basics/model.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336-basics/cs336_basics/model.py` | `llm_foundry/common/model.py` |
| **状态** | 增强 | 增强 |

与 Assignment1 的 model.py 基本相同，项目代码基于 Assignment2 版本。

**新增功能（相比 Assignment1）：**
- RMSNorm 支持
- 更好的初始化

---

### 2.2 cs336-basics/optimizer.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336-basics/cs336_basics/optimizer.py` | `llm_foundry/common/optimizer.py` |
| **状态** | 增强 | 增强 |

**类映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| 无显式类 | `AdamW` | 新增封装 |
| 无显式类 | `ShardedOptimizer` (ZeRO-1) | 新增封装 |
| 无显式类 | `CosineAnnealingWarmup` | 新增封装 |

**新增功能：**
- `AdamW` 类包装器
- `ShardedOptimizer` ZeRO-1 实现
- `CosineAnnealingWarmup` 学习率调度
- `create_optimizer()` 工厂函数

---

### 2.3 cs336-basics/data.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336-basics/cs336_basics/data.py` | `llm_foundry/common/data.py` |
| **状态** | 完整迁移 | 完整迁移 |

**函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| `get_batch()` | `get_batch()` | 完整迁移 |
| `load_tokens()` | `load_tokens()` | 完整迁移 |

**新增功能：**
- `create_data_loader()` 函数

---

### 2.4 cs336-basics/nn_utils.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336-basics/cs336_basics/nn_utils.py` | 无直接对应 |
| **状态** | 部分实现 | - |

**内容分析：**
- Reference 包含 `RMSNorm`, `Softmax`, `GeLU` 等基础组件
- 项目中这些组件被整合到 `common/model.py` 中

---

### 2.5 cs336_systems/attention_triton.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336_systems/attention_triton.py` | `llm_foundry/backends/attention_triton.py` |
| **状态** | 完整迁移 | 完整迁移 |

**函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| Triton attention kernel | Triton attention kernel | 完整迁移 |

---

### 2.6 cs336_systems/ddp_training.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336_systems/ddp_training.py` | `llm_foundry/stage2_train/distributed.py` |
| **状态** | 重构 | 重构 |

**函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| `setup_ddp()` | `setup_ddp()` | 完整迁移 |
| `cleanup_ddp()` | `cleanup_ddp()` | 完整迁移 |
| `get_rank()` | `get_rank()` | 完整迁移 |
| `get_world_size()` | `get_world_size()` | 完整迁移 |
| `all_reduce()` | `all_reduce()` | 完整迁移 |

**新增功能：**
- `DDPTrainer` 类
- `prepare_ddp_model()` 函数
- `create_distributed_dataloader()` 函数

---

### 2.7 cs336_systems/sharded_optimizer.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336_systems/sharded_optimizer.py` | `llm_foundry/common/optimizer.py` |
| **状态** | 整合 | 整合 |

**类映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| `ShardedOptimizer` | `ShardedOptimizer` | 完整迁移到 optimizer.py |

---

### 2.8 cs336_systems/benchmark.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336_systems/benchmark.py` | 无直接对应 |
| **状态** | 未实现 | - |

这是一个基准测试脚本，项目中没有直接对应。

---

### 2.9 cs336_systems/benchmark_flash.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336_systems/benchmark_flash.py` | 无直接对应 |
| **状态** | 未实现 | - |

这是一个 Flash Attention 基准测试脚本，项目中没有直接对应。

---

### 2.10 cs336_systems/test.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336_systems/test.py` | 无直接对应 |
| **状态** | 未实现 | - |

这是一个测试脚本，项目中没有直接对应。项目的测试在 `tests/` 目录下。

---

### 2.11 cs336_systems/test_triton.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment2-system/cs336_systems/test_triton.py` | 无直接对应 |
| **状态** | 未实现 | - |

这是一个 Triton 测试脚本，项目中没有直接对应。

---

## 三、Assignment3-scaling 对比

### 3.1 chinchilla_scaling_laws_fitting.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment3-scaling/cs336_scaling/chinchilla_scaling_laws_fitting.py` | `llm_foundry/stage3_scaling/fitting.py` |
| **状态** | 重构 | 重构 |

**函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| 拟合函数 | `fit_scaling_law()` | 重构 |
| 幂律计算 | `compute_flops()` | 重构 |

**新增功能：**
- `FittingConfig` 配置类
- 更完整的可视化支持

---

### 3.2 chinchilla_isoflops_scaling.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment3-scaling/cs336_scaling/chinchilla_isoflops_scaling.py` | `llm_foundry/stage3_scaling/isoflops.py` |
| **状态** | 重构 | 重构 |

**函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| IsoFLOPs 实验 | `run_isoflops_experiment()` | 重构 |
| 配置计算 | `compute_model_config()` | 重构 |

---

### 3.3 cs336_scaling/model.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment3-scaling/cs336_scaling/model.py` | `llm_foundry/common/model.py` |
| **状态** | 整合 | 整合 |

这个文件与 Assignment2 的 model.py 相同，项目中使用统一的 `common/model.py`。

---

### 3.4 项目新增文件

| 文件 | 说明 |
|------|------|
| `stage3_scaling/scaling.py` | 缩放定律核心实现，整合 fitting 和 isoflops |
| `stage3_scaling/runner.py` | 实验运行器，新增 |
| `stage3_scaling/visualization.py` | 可视化工具，新增 |

---

## 四、Assignment4-data 对比

### 4.1 cs336_data/filters.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment4-data/cs336_data/filters.py` | `llm_foundry/stage4_data/pipeline.py` |
| **状态** | 整合 | 整合 |

**函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| Gopher 规则过滤器 | `QualityPipeline` 类方法 | 重构为类 |
| 质量评分 | `score_document()` | 重构 |

---

### 4.2 cs336_data/tokenize_data.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment4-data/cs336_data/tokenize_data.py` | `llm_foundry/stage1_tokenize/binarize.py` + `pretokenize.py` |
| **状态** | 拆分 | 拆分 |

功能被拆分为：
- `binarize.py`: 二进制化
- `pretokenize.py`: 预分词

---

### 4.3 cs336_data/run_pipeline.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment4-data/cs336_data/run_pipeline.py` | `llm_foundry/stage4_data/pipeline.py` |
| **状态** | 重构 | 重构 |

**新增功能：**
- `QualityPipeline` 类
- 流式处理支持
- 配置驱动

---

### 4.4 cs336_data/build_quality_dataset.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment4-data/cs336_data/build_quality_dataset.py` | `llm_foundry/stage4_data/pipeline.py` |
| **状态** | 整合 | 整合 |

功能被整合到 `QualityPipeline` 中。

---

### 4.5 cs336-basics/data.py (Assignment4 版本)

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment4-data/cs336-basics/cs336_basics/data.py` | `llm_foundry/common/data.py` |
| **状态** | 增强 | 增强 |

与 Assignment2 版本相比，Assignment4 版本添加了：
- 更多的数据加载工具
- 数据集类支持

项目中保留了核心函数并添加了 `create_data_loader()`。

---

### 4.6 cs336-basics/ddp_utils.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment4-data/cs336-basics/cs336_basics/ddp_utils.py` | `llm_foundry/stage2_train/distributed.py` |
| **状态** | 整合 | 整合 |

功能被整合到 `distributed.py` 中。

---

### 4.7 scripts/train.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment4-data/cs336-basics/scripts/train.py` | `run.py train` |
| **状态** | 重构 | 重构 |

从独立脚本改为统一 CLI 的子命令。

---

### 4.8 scripts/generate_with_gpt2_tok.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment4-data/cs336-basics/scripts/generate_with_gpt2_tok.py` | 无直接对应 |
| **状态** | 未实现 | - |

这是一个示例脚本，项目中没有直接对应。

---

## 五、Assignment5-alignment 对比

### 5.1 sft_train.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/sft_train.py` | `llm_foundry/stage5_align/sft.py` |
| **状态** | 重构 | 重构 |

**类/函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| SFT 训练循环 | `SFTTrainer` 类 | 重构为类 |
| `tokenize_prompt_and_output()` | `tokenize_prompt_and_output()` | 完整迁移到 utils.py |

**新增功能：**
- `SFTTrainer` 类封装
- `SFTDataset` 类
- `collate_fn()` 函数

---

### 5.2 sft_instruction_tune.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/sft_instruction_tune.py` | `llm_foundry/stage5_align/sft.py` |
| **状态** | 整合 | 整合 |

功能被整合到 `SFTTrainer` 中。

---

### 5.3 dpo_train.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/dpo_train.py` | `llm_foundry/stage5_align/dpo.py` |
| **状态** | 重构 | 重构 |

**类映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| DPO 训练循环 | `DPOTrainer` 类 | 重构为类 |

**新增功能：**
- `DPOTrainer` 类封装
- `DPODataset` 类

---

### 5.4 grpo_train.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/grpo_train.py` | `llm_foundry/stage5_align/grpo.py` |
| **状态** | 重构 | 重构 |

**类映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| GRPO 训练循环 | `GRPOTrainer` 类 | 重构为类 |

---

### 5.5 grpo_off_policy_train.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/grpo_off_policy_train.py` | `llm_foundry/stage5_align/grpo.py` |
| **状态** | 整合 | 整合 |

Off-policy GRPO 功能被整合到 `GRPOTrainer` 中。

---

### 5.6 expert_iteration.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/expert_iteration.py` | 无直接对应 |
| **状态** | 未实现 | - |

Expert Iteration 功能在项目中没有直接对应。

---

### 5.7 math_baseline.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/math_baseline.py` | 无直接对应 |
| **状态** | 部分实现 | - |

数学推理评估功能部分整合到 `grpo.py` 中。

---

### 5.8 utils.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/utils.py` | `llm_foundry/stage5_align/utils.py` |
| **状态** | 完整迁移 | 完整迁移 |

**函数映射：**

| Reference | 项目对应 | 状态 |
|----------|---------|------|
| `tokenize_prompt_and_output()` | `tokenize_prompt_and_output()` | 完整迁移 |
| `get_response_log_probs()` | `get_response_log_probs()` | 完整迁移 |
| `compute_group_normalized_rewards()` | `compute_group_normalized_rewards()` | 完整迁移 |
| `grpo_microbatch_train_step()` | `grpo_microbatch_train_step()` | 完整迁移 |
| `sft_microbatch_train_step()` | `sft_microbatch_train_step()` | 完整迁移 |

---

### 5.9 drgrpo_grader.py

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/drgrpo_grader.py` | 无直接对应 |
| **状态** | 未实现 | - |

DR-GRPO 评估器在项目中没有实现。

---

### 5.10 lab/ 目录

| 属性 | Reference | 项目 |
|------|----------|------|
| **文件位置** | `Assignment5-alignment/cs336_alignment/lab/` | 无直接对应 |
| **状态** | 未实现 | - |

Lab 目录包含实验代码，项目中没有直接对应。

---

## 六、项目中新增模块（无 Reference 对应）

### 6.1 stage0_datagen/

| 文件 | 说明 |
|------|------|
| `client.py` | DeepSeek API 客户端 |
| `datagen.py` | 数据生成主模块 |
| `sft_gen.py` | SFT 数据生成 |
| `grpo_gen.py` | GRPO 数据生成 |

**说明**：这是一个全新的模块，用于通过 DeepSeek API 生成训练数据。Reference 中没有对应代码。

---

### 6.2 stage1_tokenize/pretokenize.py

预分词功能，从 Assignment4 的 `tokenize_data.py` 中拆分出来。

---

### 6.3 stage2_train/attention_inject.py

注意力机制注入工具，用于动态替换模型的注意力实现。

**说明**：这是一个新增工具，用于实现 attention 后端的动态切换。

---

### 6.4 backends/attention.py

注意力后端统一接口，提供三层 fallback：Triton → torch.compile → SDPA。

**说明**：这是一个新的抽象层，Reference 中没有直接对应。

---

### 6.5 common/config.py

YAML 配置管理，支持哈希验证。

**说明**：这是一个全新的配置管理模块。

---

### 6.6 common/env_check.py

硬件环境检测工具。

**说明**：这是一个新增的工具模块。

---

### 6.7 common/hashing.py

SHA256 哈希工具，用于可重复性验证。

**说明**：这是一个新增的工具模块。

---

### 6.8 run.py

统一 CLI 入口点。

**说明**：这是一个全新的命令行接口，整合所有功能。

---

## 七、功能迁移统计

### 7.1 完整迁移的功能

| 功能 | Reference 位置 | 项目位置 |
|------|---------------|----------|
| BPE Tokenizer | A1/Transformer/bpe.py | stage1_tokenize/tokenizer.py |
| Transformer Model | A1-2/cs336-basics/model.py | common/model.py |
| AdamW Optimizer | A2/cs336-basics/optimizer.py | common/optimizer.py |
| Data Loading | A2-4/cs336-basics/data.py | common/data.py |
| Triton Attention | A2/cs336_systems/attention_triton.py | backends/attention_triton.py |
| DDP Training | A2/cs336_systems/ddp_training.py | stage2_train/distributed.py |
| Sharded Optimizer | A2/cs336_systems/sharded_optimizer.py | common/optimizer.py |
| Scaling Laws | A3/cs336_scaling/*.py | stage3_scaling/*.py |
| Quality Filters | A4/cs336_data/filters.py | stage4_data/pipeline.py |
| SFT Training | A5/cs336_alignment/sft_train.py | stage5_align/sft.py |
| DPO Training | A5/cs336_alignment/dpo_train.py | stage5_align/dpo.py |
| GRPO Training | A5/cs336_alignment/grpo_train.py | stage5_align/grpo.py |
| Alignment Utils | A5/cs336_alignment/utils.py | stage5_align/utils.py |

### 7.2 重构/优化的功能

| 功能 | 变化说明 |
|------|---------|
| Tokenizer | 从纯函数改为 BPETokenizer 类，支持 save/load |
| Model | 添加 RoPE 支持，更好的配置管理 |
| Training | 从脚本改为 Trainer 类，支持更多功能 |
| Inference | 支持 vLLM 和 HF 两种后端 |
| DDP | 封装为 DDPTrainer 类 |
| Scaling | 添加 runner 和 visualization |
| Data Pipeline | 封装为 QualityPipeline 类 |
| Alignment | 封装为 Trainer 类，更好的抽象 |

### 7.3 新增功能（无 Reference 对应）

| 功能 | 位置 | 说明 |
|------|------|------|
| DeepSeek API 数据生成 | stage0_datagen/ | 全新模块 |
| Attention 后端抽象 | backends/attention.py | 三层 fallback |
| 配置管理 | common/config.py | YAML + 哈希验证 |
| 环境检查 | common/env_check.py | 硬件检测 |
| 统一 CLI | run.py | 命令行接口 |
| Attention 注入 | stage2_train/attention_inject.py | 动态替换 |

### 7.4 未实现的功能（Reference 中有）

| 功能 | Reference 位置 | 说明 |
|------|---------------|------|
| Expert Iteration | A5/expert_iteration.py | 未实现 |
| DR-GRPO Grader | A5/drgrpo_grader.py | 未实现 |
| Lab 实验代码 | A5/lab/ | 未实现 |
| Benchmark 脚本 | A2/benchmark*.py | 未实现 |
| Test 脚本 | A2/test*.py | 未实现 |
| BPE 实验脚本 | A1/run_bpe_experiments.py | 未实现 |
| Tokenizer 实验脚本 | A1/run_tokenizer_experiments.py | 未实现 |

---

## 八、架构差异总结

### 8.1 代码组织方式

**Reference：**
- 每个 Assignment 独立
- 脚本式编程
- 重复代码较多

**项目：**
- 统一模块结构 (`llm_foundry/`)
- 面向对象设计
- 代码复用率高

### 8.2 配置管理

**Reference：**
- 硬编码参数
- 命令行参数

**项目：**
- YAML 配置文件
- 哈希验证
- 配置类 (dataclass)

### 8.3 后端抽象

**Reference：**
- 单一实现
- 条件判断

**项目：**
- 抽象基类
- 工厂模式
- 自动 fallback

### 8.4 训练流程

**Reference：**
- 独立脚本
- 重复代码

**项目：**
- Trainer 基类
- 统一接口
- 可扩展设计

---

## 九、总结

### 代码覆盖率

| Assignment | 主要功能覆盖率 | 未实现功能 |
|-----------|--------------|-----------|
| Assignment1-basics | ~90% | 实验脚本 |
| Assignment2-system | ~85% | Benchmark, Test |
| Assignment3-scaling | ~95% | 无 |
| Assignment4-data | ~90% | 部分脚本 |
| Assignment5-alignment | ~80% | Expert Iteration, DR-GRPO |

### 架构改进

1. **统一入口点**：`run.py` 提供一致的 CLI 体验
2. **模块化设计**：清晰的阶段划分 (stage0-5)
3. **后端抽象**：支持多种实现自动切换
4. **配置驱动**：YAML 配置支持哈希验证
5. **代码复用**：避免重复代码，提高维护性

### 新增价值

1. **数据生成模块**：DeepSeek API 集成
2. **统一后端**：Triton/torch.compile/SDPA 自动选择
3. **完整流水线**：从数据生成到模型对齐的端到端支持
4. **工程化改进**：更好的错误处理、日志、文档

---

*报告生成时间：2026-03-24*
*对比范围：全部 5 个 Assignment 的 Reference 代码*
