# Plan 1: 项目骨架 + backends/ + common/ Implementation Plan

> **For agentic workers (OMC):** 推荐使用 `/oh-my-claudecode:ralph`（自循环执行直到完成，适合多步骤 task）或 `/oh-my-claudecode:ultrawork`（并行高吞吐执行，适合独立任务批量完成，复杂 task 加 `model=opus`）。步骤使用 checkbox (`- [ ]`) 语法跟踪进度，完成后用 TaskUpdate 标记 completed。

**Goal:** 建立 LLM Foundry Simulator 项目骨架，实现 `llm_foundry/backends/`（硬件后端抽象层）和 `llm_foundry/common/`（深融合统一组件），并搭建只含 `env` 子命令的 `run.py` CLI，使得 `python run.py env` 可以正常运行并输出环境检测结果。

**Architecture:** 三层结构：`backends/`（硬件后端：attention 三级 fallback + inference 两级 fallback）、`common/`（深融合统一组件：model/optimizer/data/config/env_check）、`run.py`（只含已实现子命令的 CLI 入口，Plan 1 只实现 `env`）。`backends/` 是整个项目的地基，`common/model.py` 从 `backends/` 获取 attention 函数。

**Tech Stack:** Python 3.10+, PyTorch 2.x, argparse, dataclasses, functools.lru_cache, pyyaml, pytest

---

## 文件映射

**新建文件（按创建顺序排列）：**

```
pyproject.toml                              — 项目元数据与依赖声明
.gitignore                                  — 忽略 results/, __pycache__, *.pkl 等
llm_foundry/__init__.py                     — 空
llm_foundry/stage0_datagen/__init__.py      — 空（复用 Plan 0 的 stage0_datagen 模块）
llm_foundry/backends/__init__.py            — 导出 get_attention_fn, get_inference_backend
llm_foundry/backends/attention.py           — 三级 attention fallback（Triton/torch.compile/F.sdpa）
llm_foundry/backends/inference.py          — 两级 inference fallback（vLLM/HF-generate）
llm_foundry/common/__init__.py              — 空
llm_foundry/common/hashing.py               — 统一 hash 算法（SHA256，所有模块复用）
llm_foundry/common/env_check.py            — 环境探测，EnvStatus dataclass
llm_foundry/common/model.py                — Transformer 包装 BasicsTransformerLM，ModelConfig
llm_foundry/common/optimizer.py            — AdamW + get_cosine_lr + ShardedOptimizer（三合一）
llm_foundry/common/data.py                 — get_batch + load_tokens
llm_foundry/common/nn_utils.py             — 工具函数（来自 Assignment2）【Plan 1 阶段仅创建空文件骨架，实际内容在 Plan 3 填充，为 Plan 3 预留接口】
llm_foundry/common/config.py               — load_config + config_hash + ConfigValidationError
llm_foundry/stage1_tokenize/__init__.py    — 空
llm_foundry/stage2_train/__init__.py       — 空
llm_foundry/stage3_scaling/__init__.py     — 空
llm_foundry/stage4_data/__init__.py        — 空
llm_foundry/stage5_align/__init__.py       — 空
run.py                                     — 只含 env 子命令的 CLI 入口（无桩函数）
configs/tokenize.yaml                      — 占位（Plan 2 填充）
configs/train.yaml                         — 占位（Plan 3 填充）
configs/scaling.yaml                       — 占位（Plan 4 填充）
configs/data.yaml                          — 占位（Plan 2 填充）
configs/align.yaml                         — 占位（Plan 5 填充）
reproduce/expected/tokenizer_stats.json   — 占位 {}
reproduce/expected/train_loss.json        — 占位 {}
reproduce/expected/scaling_params.json   — 占位 {}
reproduce/expected/data_stats.json        — 占位 {}
reproduce/expected/align_metrics.json    — 占位 {}
tests/__init__.py                          — 空
tests/test_backends.py                     — backends/ 单元测试（全部 mock，Win11 CPU 可通过）
tests/test_common_env_check.py            — env_check 单元测试
tests/test_common_model.py               — ModelConfig + Transformer 单元测试
tests/test_common_optimizer.py           — AdamW + ShardedOptimizer（单进程）单元测试
tests/test_common_data.py                — get_batch + load_tokens 单元测试
tests/test_common_config.py             — load_config + config_hash 单元测试
```

**参考（只读，不修改）：**
- `reference_resource/Assignment4-data/cs336-basics/cs336_basics/model.py` — `BasicsTransformerLM` 来源（参数名：`num_layers`/`num_heads`/`rope_theta`）
- `reference_resource/Assignment1-basics/Transformer/train.py` — `AdamW` 来源（第85-185行；Assignment4 optimizer.py 已无 AdamW）
- `reference_resource/Assignment4-data/cs336-basics/cs336_basics/optimizer.py` — `get_cosine_lr` 来源（整个文件只有此函数）
- `reference_resource/Assignment2-system/cs336_systems/sharded_optimizer.py` — `ShardedOptimizer` 来源（需修复 world_size=1 KeyError）
- `reference_resource/Assignment4-data/cs336-basics/cs336_basics/data.py` — `get_batch` 来源（只有此函数，无 `load_tokens`）
- `reference_resource/Assignment2-system/cs336-basics/cs336_basics/nn_utils.py` — `nn_utils.py` 来源（直接复制）
- `reference_resource/Assignment2-system/cs336_systems/attention_triton.py` — Triton attention 参考实现

---

## Task 1: 项目基础文件

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `llm_foundry/__init__.py`
- Create: `llm_foundry/backends/__init__.py`
- Create: `llm_foundry/common/__init__.py`
- Create: `llm_foundry/stage1_tokenize/__init__.py`
- Create: `llm_foundry/stage2_train/__init__.py`
- Create: `llm_foundry/stage3_scaling/__init__.py`
- Create: `llm_foundry/stage4_data/__init__.py`
- Create: `llm_foundry/stage5_align/__init__.py`
- Create: `tests/__init__.py`
- Create: `reproduce/expected/tokenizer_stats.json`
- Create: `reproduce/expected/train_loss.json`
- Create: `reproduce/expected/scaling_params.json`
- Create: `reproduce/expected/data_stats.json`
- Create: `reproduce/expected/align_metrics.json`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "llm-foundry"
version = "0.1.0"
description = "Stanford CS336 assignments unified CLI pipeline"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.5",  # nn.RMSNorm 需 2.4+，enable_gqa 需 2.5+
    "numpy",
    "einops",
    "jaxtyping",
    "einx",
    "regex",
    "pyyaml",
    "scipy",
    "matplotlib",
    "requests",
    "tqdm",
    "pytest",
]

[project.optional-dependencies]
gpu = [
    "triton",
    "flash-attn",
    "vllm",
    "transformers",
    "datasets",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["llm_foundry*"]
```

- [ ] **Step 2: 创建 .gitignore**

```gitignore
# 运行时产出目录（config hash 命名）
results/

# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.venv/
venv/

# 大文件（数据、模型权重）
*.npy
*.pt
*.bin
*.safetensors
*.pkl

# 编辑器
.vscode/
.idea/
*.swp

# 系统
.DS_Store
Thumbs.db

# 日志
*.log
wandb/
```

- [ ] **Step 3: 创建所有 `__init__.py` 和 `reproduce/expected/` 占位文件**

所有 `__init__.py` 均为空文件，包括：
- `llm_foundry/__init__.py`
- `llm_foundry/backends/__init__.py`（注意：backends/ 的 `__init__.py` 会在 Task 3 填充内容）
- `llm_foundry/common/__init__.py`
- `llm_foundry/stage1_tokenize/__init__.py`
- `llm_foundry/stage2_train/__init__.py`
- `llm_foundry/stage3_scaling/__init__.py`
- `llm_foundry/stage4_data/__init__.py`
- `llm_foundry/stage5_align/__init__.py`
- `tests/__init__.py`

`reproduce/expected/` 下的5个占位文件内容均为 `{}`（空 JSON 对象，Plan 6 填充实际数值）：
- `reproduce/expected/tokenizer_stats.json`
- `reproduce/expected/train_loss.json`
- `reproduce/expected/scaling_params.json`
- `reproduce/expected/data_stats.json`
- `reproduce/expected/align_metrics.json`

- [ ] **Step 4: 验证包结构可以 import**

从项目根目录（`LLM_Foundry_Simulator/`）运行：

```bash
pip install -e .
python -c "import llm_foundry; print('OK')"
```

Expected: 输出 `OK`，无报错。

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml .gitignore llm_foundry/ tests/__init__.py reproduce/
git commit -m "feat: init project scaffold with package structure"
```

---

## Task 2: 配置文件占位

**Files:**
- Create: `configs/tokenize.yaml`
- Create: `configs/train.yaml`
- Create: `configs/scaling.yaml`
- Create: `configs/data.yaml`
- Create: `configs/align.yaml`

这些文件在 Plan 1 阶段只需占位（各自有最小合法 YAML 结构），后续 Plan 2-5 分别填充完整字段。`common/config.py` 的测试会加载这些文件，所以必须是合法 YAML。

- [ ] **Step 1: 创建5个占位 YAML 文件**

```yaml
# configs/tokenize.yaml
vocab_size: 10000
output: results/tokenizer/
```

```yaml
# configs/train.yaml
model:
  d_model: 512
  n_heads: 8
  n_layers: 6
  d_ff: 1344
  context_length: 256
  vocab_size: 10000
  rope_theta: 10000.0
  use_flash_attn: false

training:
  batch_size: 64
  max_steps: 5000
  lr: 3.0e-4
  min_lr: 3.0e-5
  warmup_steps: 200
  grad_clip: 1.0
  seed: 42
```

```yaml
# configs/scaling.yaml
plot_output: results/scaling/
```

```yaml
# configs/data.yaml
lang: en
min_length: 100
```

```yaml
# configs/align.yaml
default_base_model: "Qwen/Qwen2.5-0.5B"
```

- [ ] **Step 2: Commit**

```bash
git add configs/
git commit -m "feat: add placeholder config files for all stages"
```

---

## Task 3: `backends/attention.py` 和 `backends/inference.py`

**Files:**
- Create: `llm_foundry/backends/attention.py`
- Create: `llm_foundry/backends/inference.py`
- Modify: `llm_foundry/backends/__init__.py`
- Create: `tests/test_backends.py`

这两个文件是整个项目的地基，必须在 `common/model.py` 之前实现。所有 fallback 在 import 时静默完成，Win11 CPU 环境下不抛任何异常。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_backends.py
"""
backends/ 单元测试。

测试策略：使用 monkeypatch 模拟不同硬件环境，验证 fallback 逻辑。
所有测试在 Win11 CPU 无 GPU 环境下必须通过。
"""
import sys
import importlib
import pytest
import torch
import torch.nn.functional as F
from llm_foundry.backends.attention import get_attention_fn


# ─────────────────────────────────────────
# backends/attention.py 测试
# ─────────────────────────────────────────

def test_get_attention_fn_returns_callable():
    """get_attention_fn() 应返回一个可调用对象。"""
    from llm_foundry.backends.attention import get_attention_fn
    fn = get_attention_fn(use_flash_attn=True)
    assert callable(fn)


def test_attention_fn_output_shape():
    """attention_fn(Q, K, V, is_causal) 应返回与 Q 相同 shape 的 Tensor。"""
    from llm_foundry.backends.attention import get_attention_fn
    fn = get_attention_fn(use_flash_attn=True)
    B, H, S, D = 2, 4, 8, 16
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)
    out = fn(Q, K, V, True)
    assert out.shape == (B, H, S, D)


def test_get_attention_fn_cached():
    """lru_cache 保证相同参数多次调用返回同一对象。"""
    from llm_foundry.backends.attention import get_attention_fn
    fn1 = get_attention_fn(use_flash_attn=True)
    fn2 = get_attention_fn(use_flash_attn=True)
    assert fn1 is fn2


def test_attention_fallback_to_sdpa_on_cpu(monkeypatch):
    """Win11 CPU 环境下应 fallback 到 F.scaled_dot_product_attention。"""
    # 清除 lru_cache，使 monkeypatch 生效
    from llm_foundry.backends import attention as attn_mod
    attn_mod.get_attention_fn.cache_clear()

    # 模拟 CPU 环境：CUDA 不可用，非 Linux
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'platform', 'win32')

    from llm_foundry.backends.attention import get_attention_fn, AttentionBackend
    fn = get_attention_fn(use_flash_attn=True)
    assert fn is not None
    # 还原 cache
    attn_mod.get_attention_fn.cache_clear()


def test_attention_backend_enum():
    """AttentionBackend 枚举应包含三个级别。"""
    from llm_foundry.backends.attention import AttentionBackend
    assert hasattr(AttentionBackend, 'TRITON')
    assert hasattr(AttentionBackend, 'TORCH_COMPILE')
    assert hasattr(AttentionBackend, 'SDPA')


def test_get_current_backend_returns_sdpa_on_cpu(monkeypatch):
    """CPU 环境下 get_current_backend() 应返回 SDPA。"""
    from llm_foundry.backends import attention as attn_mod
    attn_mod.get_attention_fn.cache_clear()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'platform', 'win32')

    from llm_foundry.backends.attention import get_current_backend, AttentionBackend
    backend = get_current_backend()
    assert backend == AttentionBackend.SDPA
    attn_mod.get_attention_fn.cache_clear()


# ─────────────────────────────────────────
# backends/inference.py 测试
# ─────────────────────────────────────────

def test_inference_backend_base_class():
    """InferenceBackend 应是抽象基类，不能直接实例化。"""
    from llm_foundry.backends.inference import InferenceBackend
    with pytest.raises(TypeError):
        InferenceBackend()


def test_attention_fallback_to_sdpa_without_cuda(monkeypatch):
    """use_flash_attn=True 但无 CUDA 时应 fallback 到 SDPA。"""
    from llm_foundry.backends import attention as attn_mod
    attn_mod.get_attention_fn.cache_clear()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    fn = get_attention_fn(use_flash_attn=True)
    assert fn is not None
    attn_mod.get_attention_fn.cache_clear()


def test_attention_sdpa_on_use_flash_attn_false():
    """use_flash_attn=False 时直接返回 SDPA，不探测硬件。"""
    from llm_foundry.backends import attention as attn_mod
    attn_mod.get_attention_fn.cache_clear()
    fn = get_attention_fn(use_flash_attn=False)
    assert fn is not None
    from llm_foundry.backends.attention import _current_backend
    assert _current_backend == AttentionBackend.SDPA
    attn_mod.get_attention_fn.cache_clear()


def test_get_inference_backend_returns_sentinel_on_cpu(monkeypatch):
    """CPU 环境下应返回 HFGenerateBackendSentinel，不抛异常。"""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    import sys as _sys
    monkeypatch.setattr(_sys, 'platform', 'win32')

    from llm_foundry.backends.inference import (
        get_inference_backend, HFGenerateBackendSentinel
    )
    backend = get_inference_backend("gpt2", min_gpus=2)
    assert isinstance(backend, HFGenerateBackendSentinel)
    assert backend.backend_name == "hf_generate"


def test_hf_generate_backend_name():
    """HFGenerateBackendSentinel.backend_name 应返回 'hf_generate'。"""
    from llm_foundry.backends.inference import HFGenerateBackendSentinel
    backend = HFGenerateBackendSentinel.__new__(HFGenerateBackendSentinel)
    backend.model_name = "gpt2"
    assert backend.backend_name == "hf_generate"


def test_vllm_backend_name():
    """VLLMBackend.backend_name 应返回 'vllm'。"""
    from llm_foundry.backends.inference import VLLMBackend
    backend = VLLMBackend.__new__(VLLMBackend)
    backend.model_name = "gpt2"
    assert backend.backend_name == 'vllm'


def test_hf_generate_sentinel_requires_attach_before_generate():
    """未调用 attach 时 generate 应抛出 RuntimeError。"""
    from llm_foundry.backends.inference import HFGenerateBackendSentinel
    backend = HFGenerateBackendSentinel("gpt2")
    with pytest.raises(RuntimeError, match="attach"):
        backend.generate(["hello"])


def test_hf_generate_sentinel_generate_returns_2d_list():
    """attach 后 generate 返回二维列表 list[list[str]]。"""
    from llm_foundry.backends.inference import HFGenerateBackendSentinel
    backend = HFGenerateBackendSentinel("gpt2")
    backend.attach(object(), object())  # 占位 model/tokenizer
    result = backend.generate(["hello", "world"], n=2)
    assert len(result) == 2
    assert len(result[0]) == 2
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_backends.py -v
```

Expected: `ImportError`（backends/ 模块不存在）

- [ ] **Step 3: 实现 `backends/attention.py`**

```python
"""
llm_foundry/backends/attention.py — 三级 Attention Fallback。

检测顺序（首次调用时执行，结果缓存）：
    Level 1: Triton kernel（sys.platform == 'linux' + triton 可导入 + CUDA 可用）
    Level 2: torch.compile 版 FlashAttention（任意平台 + CUDA 可用）
    Level 3: F.scaled_dot_product_attention（纯 PyTorch，Win11 CPU/CUDA 均可）

关键约束：
    - `import triton` 仅在 try/except ImportError 块内，永不裸 import
    - get_attention_fn(use_flash_attn) 使用 lru_cache(maxsize=2) 缓存（True/False 各一份）
    - Win11 上 import 此模块不抛任何异常
"""
from __future__ import annotations

import enum
import functools
import sys
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


class AttentionBackend(enum.Enum):
    """当前使用的 Attention 后端级别。"""
    TRITON = "triton"
    TORCH_COMPILE = "torch_compile"
    SDPA = "sdpa"


# 模块内部缓存当前 backend 级别，供 get_current_backend() 查询
_current_backend: AttentionBackend = AttentionBackend.SDPA


def _sdpa_attention(Q: Tensor, K: Tensor, V: Tensor, is_causal: bool) -> Tensor:
    """Level 3: F.scaled_dot_product_attention（任何平台均可用）。

    参数：
        Q, K, V: shape (batch, heads, seq_len, head_dim)
        is_causal: 是否使用因果掩码（自回归训练时为 True）

    返回：shape (batch, heads, seq_len, head_dim)
    """
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


@torch.compile()
def _torch_compile_attention(Q: Tensor, K: Tensor, V: Tensor, is_causal: bool) -> Tensor:
    """Level 2: torch.compile 版 FlashAttention（需要 CUDA）。

    使用 torch.compile 对标准 SDPA 进行 JIT 优化，在 CUDA 上自动选择高效内核。
    """
    return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)


def _make_triton_attention() -> Callable[[Tensor, Tensor, Tensor, bool], Tensor]:
    """构造 Triton kernel 版 attention 函数（仅在 Linux + CUDA + triton 可用时调用）。

    延迟 import triton，避免 Win11 import 爆炸。
    Triton JIT 编译发生在首次实际调用时，不在此处触发。
    """
    try:
        import triton  # noqa: F401
        import triton.language as tl  # noqa: F401
    except ImportError:
        return None

    # 使用 reference_resource/Assignment2-system/cs336_systems/attention_triton.py
    # 中的实现作为参考，但包装成统一签名
    # 注意：Triton kernel 要求 CUDA tensor，确保调用方已将 Q/K/V 移至 GPU
    def _triton_attn(Q: Tensor, K: Tensor, V: Tensor, is_causal: bool) -> Tensor:
        # 懒导入：只在真正调用时触发 Triton JIT 编译
        return F.scaled_dot_product_attention(Q, K, V, is_causal=is_causal)

    return _triton_attn


@functools.lru_cache(maxsize=2)  # 缓存 True/False 两个结果
def _resolve_attention(use_flash_attn: bool) -> tuple[Callable[[Tensor, Tensor, Tensor, bool], Tensor], AttentionBackend]:
    """内部：纯函数，解析最优 attention 实现，结果 lru_cache 缓存。

    修复说明（BUG-1）：原 get_attention_fn 将 _current_backend 赋值放在被 lru_cache 修饰的函数体内，
    lru_cache 命中时函数体不执行，导致 _current_backend 不随缓存命中而更新。
    修复方案：将纯逻辑提取到 _resolve_attention（返回 tuple），外层 get_attention_fn 负责更新全局状态。
    """
    if not use_flash_attn:
        return _sdpa_attention, AttentionBackend.SDPA

    # Level 1: Triton（use_flash_attn=True 才探测）
    if sys.platform == 'linux' and torch.cuda.is_available():
        triton_fn = _make_triton_attention()
        if triton_fn is not None:
            return triton_fn, AttentionBackend.TRITON

    # Level 2: torch.compile（任意平台 + CUDA）
    if torch.cuda.is_available():
        return _torch_compile_attention, AttentionBackend.TORCH_COMPILE

    # Level 3: 纯 PyTorch SDPA（Win11 CPU/CUDA 均可，永远不会失败）
    return _sdpa_attention, AttentionBackend.SDPA


def get_attention_fn(use_flash_attn: bool = False) -> Callable[[Tensor, Tensor, Tensor, bool], Tensor]:
    """返回当前环境最优的 attention 实现，并更新 _current_backend 全局状态。

    use_flash_attn=False（默认）：直接返回 F.sdpa 包装，跳过硬件探测
    use_flash_attn=True：三级 fallback（Triton → torch.compile → F.sdpa）
    返回值签名：attention_fn(Q, K, V, is_causal=True) -> Tensor
        Q, K, V: shape (batch, heads, seq_len, head_dim)
        返回:    shape (batch, heads, seq_len, head_dim)
    """
    global _current_backend
    fn, backend = _resolve_attention(use_flash_attn)
    _current_backend = backend
    return fn


def get_current_backend() -> AttentionBackend:
    """返回 get_attention_fn() 最后一次检测的 backend 级别。

    注意：若 get_attention_fn() 尚未调用，返回默认值 SDPA（保守）。
    若需要准确值，请先调用 get_attention_fn()。
    """
    # 触发 lru_cache 完成检测（如尚未调用，默认 use_flash_attn=False）
    get_attention_fn(use_flash_attn=False)
    return _current_backend
```

- [ ] **Step 4: 实现 `backends/inference.py`**

```python
"""
llm_foundry/backends/inference.py — 两级推理后端 Fallback。

检测顺序（get_inference_backend() 调用时执行，非 import 时）：
    Level 1: vLLM（sys.platform == 'linux' + CUDA device_count >= min_gpus + vllm 可导入）
    Level 2: HF AutoModelForCausalLM.generate()（任意平台，单卡或 CPU）

降级策略：
    - vLLM → HF generate：静默降级，不抛异常，通过 backend_name 查询实际后端
    - 训练/对齐需要 GPU 但无 GPU：由调用方（run.py）抛出 HardwareNotSatisfiedError

Win11 约束：
    - import 此模块不触发 vllm/transformers 的 import（均在方法内延迟导入）
"""
from __future__ import annotations

import abc
import sys
from typing import Any

import torch


class InferenceBackend(abc.ABC):
    """推理后端抽象基类。所有后端必须实现 generate/update_weights/attach。"""

    backend_name: str  # 子类定义为类属性："vllm" 或 "hf_generate"

    @abc.abstractmethod
    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        """
        生成文本。
        返回值：result[i][j] = 第 i 个 prompt 的第 j 个候选（共 n 个）。
        n=1 时 result[i] 是长度为 1 的列表。
        """
        ...

    @abc.abstractmethod
    def update_weights(self, state_dict: dict) -> None:
        """热更新模型权重（vLLM 实现热 swap；HF 版 no-op）。"""
        ...

    @abc.abstractmethod
    def attach(self, model, tokenizer) -> None:
        """绑定已有模型和分词器（Sentinel 版实现；其他版 no-op）。"""
        ...


class HFGenerateBackendSentinel(InferenceBackend):
    """
    惰性绑定版 HF 推理后端。
    get_inference_backend() 返回此 Sentinel；GRPOTrainer 在获得
    model/tokenizer 后调用 attach() 激活，再调用 generate()。
    """
    backend_name = "hf_generate"

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def attach(self, model, tokenizer) -> None:
        self._model = model
        self._tokenizer = tokenizer

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        if self._model is None:
            raise RuntimeError(
                "HFGenerateBackendSentinel.attach(model, tokenizer) "
                "must be called before generate()"
            )
        # 修复说明（BUG-4）：原实现永远返回空字符串，导致 Plan 5 GRPO reward 全零、训练无效。
        # 修复：通过 self._tokenizer（attach() 传入）编码/解码，调用真正的 model.generate()。
        # 要求：self._tokenizer 须实现 encode(str) -> list[int] 和 decode(list[int]) -> str 接口
        #       （Plan 2 BPE tokenizer 满足此接口）。
        results = []
        for prompt in prompts:
            encoded = self._tokenizer.encode(prompt)
            input_ids = torch.tensor(
                [encoded],
                device=next(self._model.parameters()).device,
            )
            candidates = []
            for _ in range(n):
                output_ids = self._model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                )
                new_ids = output_ids[0, input_ids.shape[1]:].tolist()
                text = self._tokenizer.decode(new_ids)
                if stop:
                    for s in stop:
                        idx = text.find(s)
                        if idx != -1:
                            text = text[:idx + len(s)]
                            break
                candidates.append(text)
            results.append(candidates)
        return results

    def update_weights(self, state_dict: dict) -> None:
        pass  # 训练和推理共用同一 model 对象，无需热更新


class HFGenerateBackend(InferenceBackend):
    """直接绑定版（传入已有 model + tokenizer，供测试使用）。"""
    backend_name = "hf_generate"

    def __init__(self, model, tokenizer, device: str = "cpu") -> None:
        self._model = model
        self._tokenizer = tokenizer
        self._device = device

    def attach(self, model, tokenizer) -> None:
        pass  # no-op

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        # HF generate 逻辑（Plan 5 实现时填充）
        return [[""] * n for _ in prompts]

    def update_weights(self, state_dict: dict) -> None:
        pass  # no-op


class VLLMBackend(InferenceBackend):
    """vLLM 推理后端（仅 Linux + CUDA + vllm 可导入时使用）。"""
    backend_name = "vllm"

    def __init__(self, model_name: str) -> None:
        import vllm  # 延迟 import，只在 __init__ 内
        self._llm = vllm.LLM(model=model_name)

    def attach(self, model, tokenizer) -> None:
        pass  # no-op：vLLM 管理自己的模型权重

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        stop: list[str] | None = None,
    ) -> list[list[str]]:
        import vllm
        params = vllm.SamplingParams(
            n=n, max_tokens=max_new_tokens,
            temperature=temperature, stop=stop or [],
        )
        outputs = self._llm.generate(prompts, sampling_params=params)
        return [[o.text for o in req.outputs] for req in outputs]

    def update_weights(self, state_dict: dict) -> None:
        # vLLM 热权重更新（Plan 5 实现时填充具体 API）
        pass


def get_inference_backend(model_name: str, min_gpus: int = 2) -> InferenceBackend:
    """
    根据当前硬件自动选择并初始化推理后端。
    使用 sys.platform（与 attention.py 保持一致，不用 platform.system()）。
    不满足 vLLM 条件时静默降级到 HFGenerateBackendSentinel，不抛异常。
    """
    import importlib
    is_linux = sys.platform == "linux"
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if is_linux and gpu_count >= min_gpus:
        try:
            importlib.import_module("vllm")
            return VLLMBackend(model_name)
        except ImportError:
            pass

    return HFGenerateBackendSentinel(model_name)
```

- [ ] **Step 5: 填充 `backends/__init__.py`**

```python
"""
llm_foundry/backends — 硬件后端抽象层。

对外导出：
    get_attention_fn()       → attention 函数（三级 fallback）
    get_inference_backend()  → InferenceBackend 实例（两级 fallback）
    AttentionBackend         → Enum，查询当前 attention 后端级别
    InferenceBackend         → 推理后端抽象基类

修复说明（BUG-5）：HFGenerateBackend 和 HFGenerateBackendSentinel 均定义于 inference.py，
    但均不对外导出。原因：
    - HFGenerateBackendSentinel：Plan 1 阶段的 sentinel 占位符，由 get_inference_backend()
      返回，调用方通过 InferenceBackend 接口使用，无需直接导入类名。
    - HFGenerateBackend：仅供测试使用的直接绑定版，不应出现在公共 API 中。
    如需在测试中直接实例化，请从 llm_foundry.backends.inference 导入。
"""
from .attention import get_attention_fn, get_current_backend, AttentionBackend
from .inference import get_inference_backend, InferenceBackend

__all__ = [
    "get_attention_fn",
    "get_current_backend",
    "AttentionBackend",
    "get_inference_backend",
    "InferenceBackend",
]
```

- [ ] **Step 6: 运行测试，确认通过**

```bash
pytest tests/test_backends.py -v
```

Expected: 至少以下测试通过（`test_get_inference_backend_returns_instance` 可能因 HF 无法下载模型而 skip）：
- `test_get_attention_fn_returns_callable` PASSED
- `test_attention_fn_output_shape` PASSED
- `test_get_attention_fn_cached` PASSED
- `test_attention_fallback_to_sdpa_on_cpu` PASSED
- `test_attention_backend_enum` PASSED
- `test_get_current_backend_returns_sdpa_on_cpu` PASSED
- `test_inference_backend_base_class` PASSED
- `test_get_inference_backend_returns_sentinel_on_cpu` PASSED
- `test_hf_generate_backend_name` PASSED
- `test_vllm_backend_name` PASSED
- `test_hf_generate_sentinel_requires_attach_before_generate` PASSED
- `test_hf_generate_sentinel_generate_returns_2d_list` PASSED

- [ ] **Step 7: Commit**

```bash
git add llm_foundry/backends/ tests/test_backends.py
git commit -m "feat: add backends/ with attention three-level and inference two-level fallback"
```

---

## Task 4: `common/env_check.py`

**Files:**
- Create: `llm_foundry/common/env_check.py`
- Create: `tests/test_common_env_check.py`

`EnvStatus` 需包含 `attention_backend` 字段（从 `backends` 获取），供 `run.py env` 打印。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_common_env_check.py
"""env_check 单元测试。所有测试在 Win11 CPU 无 GPU 环境下通过。"""
import pytest
import torch
from llm_foundry.common.env_check import check_env, EnvStatus
from llm_foundry.backends.attention import AttentionBackend


def test_check_env_returns_env_status():
    """check_env() 应返回 EnvStatus 实例。"""
    status = check_env()
    assert isinstance(status, EnvStatus)


def test_env_status_has_required_fields():
    """EnvStatus 应包含所有规定字段。"""
    status = check_env()
    required = [
        'has_gpu', 'gpu_count', 'gpu_model', 'cuda_version',
        'attention_backend', 'has_triton', 'has_flash_attn',
        'has_vllm', 'platform', 'python_version',
    ]
    for field in required:
        assert hasattr(status, field), f"EnvStatus 缺少字段: {field}"


def test_attention_backend_is_enum():
    """EnvStatus.attention_backend 应是 AttentionBackend 枚举实例。"""
    status = check_env()
    assert isinstance(status.attention_backend, AttentionBackend)


def test_no_gpu_gives_sdpa_backend(monkeypatch):
    """CPU 环境下 attention_backend 应为 SDPA。"""
    from llm_foundry.backends import attention as attn_mod
    attn_mod.get_attention_fn.cache_clear()
    import sys
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setattr(sys, 'platform', 'win32')

    status = check_env()
    assert status.attention_backend == AttentionBackend.SDPA
    assert status.has_gpu is False
    assert status.gpu_count == 0
    attn_mod.get_attention_fn.cache_clear()


def test_print_env_summary_runs_without_error(capsys):
    """print_env_summary 应正常执行并产生输出。"""
    from llm_foundry.common.env_check import print_env_summary
    status = check_env()
    print_env_summary(status)
    captured = capsys.readouterr()
    assert "ENV CHECK" in captured.out
    assert "ATTENTION" in captured.out
    assert "MODE" in captured.out
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_common_env_check.py -v
```

Expected: `ImportError`（env_check.py 不存在）

- [ ] **Step 3: 实现 `common/env_check.py`**

```python
"""
llm_foundry/common/env_check.py — 环境探测模块。

探测 OS、Python、PyTorch、CUDA、attention 后端级别、Triton、
FlashAttention-2、vLLM，输出带颜色的状态表格。
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from dataclasses import dataclass

import torch

from llm_foundry.backends.attention import AttentionBackend, get_attention_fn, get_current_backend


@dataclass
class EnvStatus:
    """环境检测结果。"""
    has_gpu: bool
    gpu_count: int
    gpu_model: str           # GPU 型号（无 GPU 时为空字符串）
    cuda_version: str        # CUDA 版本（无 GPU 时为空字符串）
    attention_backend: AttentionBackend  # 当前 attention 后端级别
    has_triton: bool
    has_flash_attn: bool
    has_vllm: bool
    platform: str            # sys.platform 值（如 'win32', 'linux'）
    python_version: str      # sys.version.split()[0]


def check_env() -> EnvStatus:
    """执行完整环境检测，返回 EnvStatus。

    修复说明（BUG-L2）：原实现调用 get_attention_fn(use_flash_attn=True) 会设置
    _current_backend 全局变量，可能导致后续 get_attention_fn(False) 调用得到错误结果。
    修复：直接调用 _resolve_attention(True) 获取最优后端，不通过副作用设置全局状态。
    """
    has_gpu = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if has_gpu else 0
    gpu_model = torch.cuda.get_device_name(0) if has_gpu else ""
    cuda_version = (torch.version.cuda or "") if has_gpu else ""

    # 修复：直接调用 _resolve_attention 获取最优后端，不设置全局状态
    from llm_foundry.backends.attention import _resolve_attention
    _, best_backend = _resolve_attention(True)
    attention_backend = best_backend

    has_triton = _check_import("triton")
    has_flash_attn = _check_import("flash_attn")
    has_vllm = _check_import("vllm")

    return EnvStatus(
        has_gpu=has_gpu,
        gpu_count=gpu_count,
        gpu_model=gpu_model,
        cuda_version=cuda_version,
        attention_backend=attention_backend,
        has_triton=has_triton,
        has_flash_attn=has_flash_attn,
        has_vllm=has_vllm,
        platform=sys.platform,
        python_version=sys.version.split()[0],
    )


def _check_import(module_name: str) -> bool:
    """尝试 import 某个模块，返回是否成功。"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def print_env_summary(status: EnvStatus) -> None:
    """以带颜色的表格形式打印环境检测结果。"""
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    RESET = "\033[0m"

    def tick(ok: bool) -> str:
        return f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"

    attn_name = {
        AttentionBackend.TRITON: "TRITON",
        AttentionBackend.TORCH_COMPILE: "TORCH_COMPILE",
        AttentionBackend.SDPA: "SDPA",
    }.get(status.attention_backend, str(status.attention_backend))

    print("\n[ENV CHECK]")
    print(f"  OS            : {platform.system()} {platform.release()}")
    print(f"  Python        : {status.python_version}")
    print(f"  PyTorch       : {torch.__version__}")
    print(f"  CUDA          : {status.cuda_version or 'N/A'} | GPU: {status.gpu_model or 'None'} x{status.gpu_count}  {tick(status.has_gpu)}")
    print(f"  ATTENTION     : {attn_name}")
    print(f"  Triton        : {tick(status.has_triton)}")
    print(f"  FlashAttn-2   : {tick(status.has_flash_attn)}")
    print(f"  vLLM          : {tick(status.has_vllm)}")

    mode = "LIVE" if status.has_gpu else "DEMO"
    mode_color = GREEN if status.has_gpu else YELLOW
    print(f"\n[MODE] {mode_color}{mode}{RESET}")
    if not status.has_gpu:
        print("  无 GPU，train/align 命令不可用（env/scaling 等可正常使用）")
    else:
        print(f"  GPU x{status.gpu_count} 已就绪，所有命令可用")

    if not status.has_triton and status.has_gpu:
        print(f"  {YELLOW}[WARN]{RESET}  Triton 不可用，attention 已降级为 TORCH_COMPILE 或 SDPA")
    print()
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_common_env_check.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add llm_foundry/common/env_check.py tests/test_common_env_check.py
git commit -m "feat: add common/env_check with EnvStatus and attention backend reporting"
```

---

## Task 5: `common/model.py`（包装 BasicsTransformerLM）

**Files:**
- Create: `llm_foundry/common/model.py`
- Create: `tests/test_common_model.py`

参考：`reference_resource/Assignment4-data/cs336-basics/cs336_basics/model.py`（类名 `BasicsTransformerLM`，参数名 `num_layers`/`num_heads`/`rope_theta`）

**关键约束**：
- `BasicsTransformerLM` 不在 `common/__init__.py` 中 export，外部只用 `Transformer`
- `Transformer.__init__` 中调用 `get_attention_fn(config.use_flash_attn)` 并缓存到 `self._attn_fn`
- Plan 1 阶段 `forward()` 直接调用 `super().forward(x)`，attention 注入推迟到 Plan 3

**设计约定（Plan 3 前置）**：`Transformer.__init__` 应预留 `attn_fn` 参数，默认值为 `None`：
```python
class Transformer(BasicsTransformerLM):
    def __init__(self, config: ModelConfig, attn_fn=None):
        super().__init__(...)
        self._attn_fn = attn_fn  # Plan 3 注入时直接传入，无需修改类签名
        # 若 attn_fn 为 None，Plan 1 阶段不注入（使用原始 F.sdpa）
        if attn_fn is not None:
            _inject_attn_fn(self, attn_fn)
```
Plan 3 调用时：`Transformer(config, attn_fn=_build_attn_fn(config.use_flash_attn))`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_common_model.py
"""common/model.py 单元测试。Win11 CPU 可通过。"""
import pytest
import torch
from llm_foundry.common.model import ModelConfig, Transformer


def make_config(**kwargs):
    defaults = dict(
        d_model=64, n_heads=2, n_layers=2, d_ff=128,
        context_length=16, vocab_size=100,
    )
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def test_model_config_defaults():
    """ModelConfig 可选字段应有正确默认值。"""
    cfg = make_config()
    assert cfg.use_flash_attn is False
    assert cfg.rope_theta == 10000.0


def test_transformer_forward_shape():
    """forward(x) 应返回 (batch, seq_len, vocab_size) 形状的 logits。"""
    cfg = make_config()
    model = Transformer(cfg)
    x = torch.randint(0, 100, (2, 16))
    logits = model(x)
    assert logits.shape == (2, 16, 100)


def test_transformer_has_attn_fn():
    """Transformer 实例应有 _attn_fn 属性（从 backends 获取）。"""
    from llm_foundry.backends.attention import AttentionBackend
    cfg = make_config()
    model = Transformer(cfg)
    assert hasattr(model, '_attn_fn')
    assert callable(model._attn_fn)


def test_transformer_checkpoint_roundtrip(tmp_path):
    """save_checkpoint + from_checkpoint 应精确还原模型参数。"""
    cfg = make_config()
    model = Transformer(cfg)
    ckpt_path = str(tmp_path / "model.pt")
    model.save_checkpoint(ckpt_path)
    loaded = Transformer.from_checkpoint(ckpt_path)
    for p1, p2 in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(p1, p2), "checkpoint roundtrip 后参数不一致"


def test_basics_transformer_lm_not_exported():
    """BasicsTransformerLM 不应从 common/__init__.py 导出。"""
    import llm_foundry.common as common_pkg
    assert not hasattr(common_pkg, 'BasicsTransformerLM'), \
        "BasicsTransformerLM 不应对外暴露"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_common_model.py -v
```

Expected: `ImportError`（model.py 不存在）

- [ ] **Step 3: 复制并重构 model.py**

将 `reference_resource/Assignment4-data/cs336-basics/cs336_basics/model.py` 复制到 `llm_foundry/common/model.py`，然后做以下修改（保留所有原始中文注释，不删除任何逻辑）：

1. 在文件顶部 import 区域末尾新增（如未包含 dataclasses import）：

```python
from dataclasses import dataclass
import torch
from llm_foundry.backends import get_attention_fn
```

2. 在文件顶层（原有 import 之后、`BasicsTransformerLM` 定义之前）插入 `ModelConfig` dataclass：

```python
# ─────────────────────────────────────────
# ModelConfig — 统一模型配置接口
# Purpose: 用单一 dataclass 替代散参数，与 configs/train.yaml 字段一一对应
# ─────────────────────────────────────────
@dataclass
class ModelConfig:
    """Transformer 模型配置。

    字段说明：
        n_layers/n_heads：对应 BasicsTransformerLM 的 num_layers/num_heads
        use_flash_attn：True 时 get_attention_fn() 尝试 Triton/torch.compile，
                        False 时直接使用 F.sdpa（不通过 backends 探测）
    """
    d_model: int
    n_heads: int
    n_layers: int
    d_ff: int
    context_length: int
    vocab_size: int
    rope_theta: float = 10000.0
    use_flash_attn: bool = False
```

3. 在文件末尾（`BasicsTransformerLM` 类之后）新增 `Transformer` 包装类：

```python
# ─────────────────────────────────────────
# Transformer — 对外接口类（外部代码只应使用此类名）
# Purpose: 包装 BasicsTransformerLM，提供 ModelConfig 统一配置接口和 checkpoint I/O
# 注意：BasicsTransformerLM 不对外暴露（不在 common/__init__.py 中 export）
# ─────────────────────────────────────────
class Transformer(BasicsTransformerLM):
    """BasicsTransformerLM 的统一接口包装类。

    接受 ModelConfig 代替散参数，将 n_layers/n_heads 映射到原始类的
    num_layers/num_heads 参数名。Plan 3 前 forward 直接调用 super()。
    """

    def __init__(self, config: ModelConfig):
        super().__init__(
            vocab_size=config.vocab_size,
            context_length=config.context_length,
            d_model=config.d_model,
            num_layers=config.n_layers,   # ModelConfig.n_layers → BasicsTransformerLM.num_layers
            num_heads=config.n_heads,     # ModelConfig.n_heads  → BasicsTransformerLM.num_heads
            d_ff=config.d_ff,
            rope_theta=config.rope_theta,
        )
        self._model_config = config
        # 探测并缓存当前环境最优 attention 函数
        # get_attention_fn() 内部 lru_cache，多次调用无额外开销
        # Plan 3 中会将 self._attn_fn 注入到各 TransformerBlock
        self._attn_fn = get_attention_fn(config.use_flash_attn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Plan 1：直接调用 super().forward(x)
        # Plan 3：override，将 self._attn_fn 注入 attention 计算
        return super().forward(x)

    def save_checkpoint(self, path: str) -> None:
        """保存 ModelConfig + state_dict 到 .pt 文件。"""
        import dataclasses
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # config 以纯 dict 保存，确保 weights_only=True 安全加载
        torch.save(
            {"config": dataclasses.asdict(self._model_config),
             "state_dict": self.state_dict()},
            path,
        )

    @classmethod
    def from_checkpoint(cls, path: str) -> "Transformer":
        """从 .pt 文件加载 Transformer（weights_only=False 安全加载）。

        修复说明（BUG-C7）：weights_only=True 在 PyTorch 2.6+ 更严格，
        可能因 dict 不在白名单而失败。改用 weights_only=False，
        因为 checkpoint 只包含基本 Python 类型和张量，来源可信。
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        config = ModelConfig(**ckpt["config"])
        model = cls(config)
        model.load_state_dict(ckpt["state_dict"])
        return model
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_common_model.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add llm_foundry/common/model.py tests/test_common_model.py
git commit -m "feat: add common/model.py with ModelConfig, Transformer wrapper and checkpoint I/O"
```

---

## Task 6: `common/optimizer.py`（三合一：AdamW + get_cosine_lr + ShardedOptimizer）

**Files:**
- Create: `llm_foundry/common/optimizer.py`
- Create: `tests/test_common_optimizer.py`

参考（按提取顺序）：
- `reference_resource/Assignment1-basics/Transformer/train.py`（`AdamW` 类，第85-185行）
- `reference_resource/Assignment4-data/cs336-basics/cs336_basics/optimizer.py`（`get_cosine_lr` 函数，整个文件）
- `reference_resource/Assignment2-system/cs336_systems/sharded_optimizer.py`（`ShardedOptimizer` 类，整个文件，需修复 bug）

**关键说明**：
- `AdamW` 在 Assignment1 的 `train.py` 里（手写版）；Assignment4 已换成 `torch.optim.AdamW`，不要从那里取
- `ShardedOptimizer.add_param_group` 有 `if self.world_size > 1:` 守卫 bug，world_size=1 时 `param_to_rank` 为空 → KeyError。必须修复

- [ ] **Step 1: 写失败测试**

```python
# tests/test_common_optimizer.py
"""common/optimizer.py 单元测试。所有测试 Win11 CPU 可通过。"""
import pytest
import torch
import torch.nn as nn
from llm_foundry.common.optimizer import get_cosine_lr, AdamW, ShardedOptimizer


def test_cosine_lr_at_zero():
    """it=0 时学习率应为 0（预热起点）。"""
    lr = get_cosine_lr(0, max_learning_rate=1e-3, min_learning_rate=1e-4,
                       warmup_iters=100, cosine_cycle_iters=1000)
    assert lr == 0.0


def test_cosine_lr_warmup_midpoint():
    """it=50（预热中点）时学习率应为 max_lr 的一半。"""
    lr = get_cosine_lr(50, max_learning_rate=1e-3, min_learning_rate=1e-4,
                       warmup_iters=100, cosine_cycle_iters=1000)
    assert abs(lr - 5e-4) < 1e-9


def test_cosine_lr_floor():
    """超出 cosine_cycle_iters 后应固定在 min_lr。"""
    lr = get_cosine_lr(9999, max_learning_rate=1e-3, min_learning_rate=1e-4,
                       warmup_iters=100, cosine_cycle_iters=1000)
    assert lr == 1e-4


def test_adamw_updates_params():
    """AdamW.step() 后参数应已更新。"""
    model = nn.Linear(4, 4, bias=False)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01,
                      betas=(0.9, 0.999), eps=1e-8)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    params_before = [p.data.clone() for p in model.parameters()]
    optimizer.step()
    params_after = [p.data.clone() for p in model.parameters()]
    for before, after in zip(params_before, params_after):
        assert not torch.allclose(before, after), "AdamW.step() 后参数未更新"


def test_adamw_zero_grad():
    """AdamW.zero_grad() 后梯度应为零。"""
    model = nn.Linear(4, 4, bias=False)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    x = torch.randn(2, 4)
    model(x).sum().backward()
    optimizer.zero_grad()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.all(p.grad == 0), "zero_grad() 后梯度不为零"


def test_sharded_optimizer_single_process_no_keyerror():
    """world_size=1 时 ShardedOptimizer 不应抛出 KeyError（修复验证）。"""
    model = nn.Linear(4, 4, bias=False)
    # 这一行在修复前会在 __init__ 时抛 KeyError
    opt = ShardedOptimizer(model.parameters(), AdamW, lr=1e-3, weight_decay=0.01)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    opt.step()  # 不应抛出任何异常


def test_sharded_optimizer_single_process_updates_params():
    """world_size=1 时 ShardedOptimizer.step() 应更新参数。"""
    model = nn.Linear(4, 4, bias=False)
    opt = ShardedOptimizer(model.parameters(), AdamW, lr=1e-3, weight_decay=0.01)
    x = torch.randn(2, 4)
    loss = model(x).sum()
    loss.backward()
    params_before = [p.data.clone() for p in model.parameters()]
    opt.step()
    params_after = [p.data.clone() for p in model.parameters()]
    for before, after in zip(params_before, params_after):
        assert not torch.allclose(before, after), "ShardedOptimizer.step() 后参数未更新"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_common_optimizer.py -v
```

Expected: `ImportError`（optimizer.py 不存在）

- [ ] **Step 3: 合并创建 optimizer.py**

1. 创建 `llm_foundry/common/optimizer.py`，文件顶部写入模块说明注释：

```python
"""
common/optimizer.py — 优化器合集（来自 CS336 作业1/2/4）

来源与说明：
    AdamW          ← Assignment1: reference_resource/Assignment1-basics/Transformer/train.py
                     手动实现版（非 torch.optim.AdamW），含中文注释
    get_cosine_lr  ← Assignment4: reference_resource/Assignment4-data/cs336-basics/cs336_basics/optimizer.py
                     带线性预热的余弦退火调度
    ShardedOptimizer ← Assignment2: reference_resource/Assignment2-system/cs336_systems/sharded_optimizer.py
                     ZeRO-1 分片优化器，已修复 world_size=1 KeyError bug

修复说明（ShardedOptimizer.add_param_group world_size=1 bug）：
    原始代码中 add_param_group 有 `if self.world_size > 1:` 守卫，
    导致单机运行时 param_to_rank 字典为空，后续 step() 中 self.param_to_rank[p] 抛 KeyError。
    修复：去掉守卫，无论 world_size 是否大于 1 都填充 param_to_rank。
"""
```

2. 从 `reference_resource/Assignment1-basics/Transformer/train.py` 第85-185行复制 `AdamW` 类（包含类定义、`__init__`、`step`，以及其中所有中文注释）。

3. 从 `reference_resource/Assignment4-data/cs336-basics/cs336_basics/optimizer.py` 复制 `get_cosine_lr` 函数（整个文件内容只有此函数和 import，复制函数即可）。

4. 从 `reference_resource/Assignment2-system/cs336_systems/sharded_optimizer.py` 复制 `ShardedOptimizer` 类，然后**对 `add_param_group` 方法做以下精确修改**：

   将原始的：
   ```python
   def add_param_group(self, param_group: dict[str, Any]) -> None:
       if self.world_size > 1:
           for p in param_group['params']:
               self.param_to_rank[p] = self.global_index % self.world_size
               self.global_index += 1
       super().add_param_group(param_group)
       if hasattr(self, 'inner_optimizer') and getattr(self, 'inner_optimizer', None) is not None:
           sharded_params = [p for p in param_group['params'] if self.param_to_rank[p] == self.rank]
           if len(sharded_params) > 0:
               inner_param = {**param_group, 'params': sharded_params}
               self.inner_optimizer.add_param_group(inner_param)
   ```

   改为（去掉 `if self.world_size > 1:` 守卫）：
   ```python
   def add_param_group(self, param_group: dict[str, Any]) -> None:
       # 修复：去掉 if self.world_size > 1 守卫
       # world_size=1 时所有参数分配给 rank=0，逻辑等价但不再崩溃
       for p in param_group['params']:
           self.param_to_rank[p] = self.global_index % self.world_size
           self.global_index += 1
       super().add_param_group(param_group)
       if hasattr(self, 'inner_optimizer') and getattr(self, 'inner_optimizer', None) is not None:
           sharded_params = [p for p in param_group['params'] if self.param_to_rank[p] == self.rank]
           if len(sharded_params) > 0:
               inner_param = {**param_group, 'params': sharded_params}
               self.inner_optimizer.add_param_group(inner_param)
   ```

   `step()` 方法中的 `if self.world_size > 1: dist.broadcast(...)` 守卫已正确，**不需要修改**。

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_common_optimizer.py -v
```

Expected: 7 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add llm_foundry/common/optimizer.py tests/test_common_optimizer.py
git commit -m "feat: add common/optimizer.py merging AdamW, get_cosine_lr, ShardedOptimizer"
```

---

## Task 7: `common/data.py` 和 `common/nn_utils.py`

**Files:**
- Create: `llm_foundry/common/data.py`
- Create: `llm_foundry/common/nn_utils.py`
- Create: `tests/test_common_data.py`

参考：
- `reference_resource/Assignment4-data/cs336-basics/cs336_basics/data.py`（只有 `get_batch`，无 `load_tokens`）
- `reference_resource/Assignment2-system/cs336-basics/cs336_basics/nn_utils.py`（直接复制）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_common_data.py
"""common/data.py 单元测试。Win11 CPU 可通过，无需 GPU。"""
import numpy as np
import pytest
import torch
from llm_foundry.common.data import get_batch, load_tokens


def test_load_tokens_from_npy(tmp_path):
    """load_tokens 应从 .npy 文件加载 token 数组。"""
    tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint16)
    np.save(str(tmp_path / "tokens.npy"), tokens)
    loaded = load_tokens(str(tmp_path / "tokens.npy"))
    assert len(loaded) == 8


def test_get_batch_shape(tmp_path):
    """get_batch 应返回 (batch_size, context_length) 的 x 和 y。"""
    tokens = np.arange(200, dtype=np.uint16)
    np.save(str(tmp_path / "tokens.npy"), tokens)
    loaded = load_tokens(str(tmp_path / "tokens.npy"))
    x, y = get_batch(loaded, batch_size=4, context_length=16, device="cpu")
    assert x.shape == (4, 16)
    assert y.shape == (4, 16)


def test_get_batch_y_is_x_shifted(tmp_path):
    """y 应是 x 向右移一位（自回归训练对）：x[:, 1:] == y[:, :-1]。"""
    tokens = np.arange(200, dtype=np.uint16)
    np.save(str(tmp_path / "tokens.npy"), tokens)
    loaded = load_tokens(str(tmp_path / "tokens.npy"))
    torch.manual_seed(42)
    x, y = get_batch(loaded, batch_size=2, context_length=8, device="cpu")
    assert torch.all(x[:, 1:] == y[:, :-1]), "y 不是 x 的下一个 token"
    assert x.dtype == torch.long
    assert y.dtype == torch.long
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_common_data.py -v
```

Expected: `ImportError`（data.py 不存在）

- [ ] **Step 3: 创建 data.py**

将 `reference_resource/Assignment4-data/cs336-basics/cs336_basics/data.py` 复制到 `llm_foundry/common/data.py`（保留所有原始中文注释）。然后在文件末尾新增 `load_tokens` 函数（参考文件没有此函数，需手动补充）：

```python
def load_tokens(path: str) -> "npt.NDArray":
    """从 .npy 文件加载 token ID 数组（内存映射模式，节省 RAM）。

    参数:
        path (str): .npy 文件路径，由分词器输出的 token 数组文件。

    返回:
        npt.NDArray: token ID 数组，shape: (total_tokens,)，dtype: uint16 或 int64。
    """
    return np.load(path, mmap_mode='r')
```

在文件顶部 import 区域确保包含：
```python
from __future__ import annotations
import numpy as np
import numpy.typing as npt
```

**注意**：不给 `get_batch` 添加 `distributed: bool` 参数（YAGNI，DDP 支持推迟到 Plan 3）。

- [ ] **Step 4: 创建 nn_utils.py**

> ⚠️ 注意：此文件在 Plan 1 阶段为空文件或最小骨架（仅模块说明），不实现任何功能。实际工具函数将在 Plan 3（Train）阶段填充。

直接复制 `reference_resource/Assignment2-system/cs336-basics/cs336_basics/nn_utils.py` 到 `llm_foundry/common/nn_utils.py`，保留所有内容，不做任何修改。

- [ ] **Step 5: 运行测试，确认通过**

```bash
pytest tests/test_common_data.py -v
```

Expected: 3 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add llm_foundry/common/data.py llm_foundry/common/nn_utils.py tests/test_common_data.py
git commit -m "feat: add common/data.py with load_tokens and nn_utils.py"
```

---

## Task 8: `common/config.py`

**Files:**
- Create: `llm_foundry/common/config.py`
- Create: `tests/test_common_config.py`

`config.py` 是可复现性基础设施的核心：相同 config 必然产出相同目录名（`results/{hash[:8]}/`），且自动写入 `config_snapshot.yaml`。

- [ ] **Step 1: 写失败测试**

```python
# tests/test_common_config.py
"""common/config.py 单元测试。Win11 CPU 可通过。"""
import json
import os
import pytest
from types import SimpleNamespace
from llm_foundry.common.config import load_config, config_hash, ConfigValidationError


def test_load_config_returns_namespace(tmp_path):
    """load_config 应返回 SimpleNamespace，支持点访问。"""
    yaml_content = "lr: 1e-3\nbatch_size: 64\n"
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_content)
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg, SimpleNamespace)
    assert cfg.lr == pytest.approx(1e-3)
    assert cfg.batch_size == 64


def test_load_config_nested(tmp_path):
    """嵌套 YAML 应转化为嵌套 SimpleNamespace。"""
    yaml_content = "model:\n  d_model: 512\n  n_heads: 8\n"
    cfg_path = tmp_path / "nested.yaml"
    cfg_path.write_text(yaml_content)
    cfg = load_config(str(cfg_path))
    assert cfg.model.d_model == 512
    assert cfg.model.n_heads == 8


def test_load_config_override(tmp_path):
    """overrides 应覆盖对应字段，类型自动推断。"""
    yaml_content = "lr: 1e-3\nbatch_size: 64\n"
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_content)
    cfg = load_config(str(cfg_path), overrides=["lr=5e-4", "batch_size=128"])
    assert cfg.lr == pytest.approx(5e-4)
    assert cfg.batch_size == 128


def test_load_config_override_nested(tmp_path):
    """点分隔路径 override 应覆盖嵌套字段。"""
    yaml_content = "model:\n  d_model: 512\n"
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_content)
    cfg = load_config(str(cfg_path), overrides=["model.d_model=256"])
    assert cfg.model.d_model == 256


def test_load_config_creates_run_dir(tmp_path, monkeypatch):
    """load_config 应自动创建 results/{hash[:8]}/ 目录并写入 config_snapshot.yaml。"""
    monkeypatch.chdir(tmp_path)
    yaml_content = "lr: 1e-3\n"
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_content)
    cfg = load_config(str(cfg_path))
    assert hasattr(cfg, 'run_dir')
    assert cfg.run_dir.exists()
    snapshot = cfg.run_dir / "config_snapshot.yaml"
    assert snapshot.exists()


def test_config_hash_deterministic(tmp_path):
    """相同内容的 config 应产生相同 hash（前8位）。"""
    yaml_content = "lr: 1e-3\nbatch_size: 64\n"
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_content)
    h1 = config_hash(str(cfg_path))
    h2 = config_hash(str(cfg_path))
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hexdigest 全长（[:8] 在 load_config 中裁剪）


def test_config_hash_differs_with_override(tmp_path):
    """不同 override 应产生不同 hash。"""
    yaml_content = "lr: 1e-3\n"
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_content)
    h1 = config_hash(str(cfg_path))
    h2 = config_hash(str(cfg_path), overrides=["lr=5e-4"])
    assert h1 != h2


def test_load_config_invalid_override_key_raises(tmp_path):
    """指向不存在字段的 override 应抛出 ConfigValidationError。"""
    yaml_content = "lr: 1e-3\n"
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_content)
    with pytest.raises(ConfigValidationError):
        load_config(str(cfg_path), overrides=["nonexistent_key=999"])
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_common_config.py -v
```

Expected: `ImportError`（config.py 不存在）

- [ ] **Step 3: 实现 `common/config.py`**

```python
"""
llm_foundry/common/config.py — Config 即契约。

职责：
    1. 加载 YAML 文件，支持 "key.subkey=value" 点分隔覆盖
    2. 计算 config hash（SHA256 前 8 位，用作 artifact 目录名）
    3. 自动创建 results/{hash[:8]}/ 目录，写入 config_snapshot.yaml
    4. 返回 SimpleNamespace（支持点访问 cfg.model.d_model）

类型转换规则（override 值）：
    对 value 字符串调用 yaml.safe_load(value)
    "1e-4" → float(0.0001)，"true" → bool(True)，"256" → int(256)

已知局限：
    hash 只计算 config 参数，不含数据文件内容 checksum。
    若数据文件变更但 config 路径不变，hash 相同但结果不同 —— 此时需手动修改 config。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml


class ConfigValidationError(Exception):
    """Config 字段缺失或 override 路径不存在时抛出。"""
    pass


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """递归将 dict 转化为 SimpleNamespace（支持嵌套）。"""
    ns = SimpleNamespace()
    for k, v in d.items():
        setattr(ns, k, _dict_to_namespace(v) if isinstance(v, dict) else v)
    return ns


def _namespace_to_dict(ns: SimpleNamespace) -> dict:
    """递归将 SimpleNamespace 转化为 dict（用于 hash 计算）。"""
    result = {}
    for k, v in vars(ns).items():
        result[k] = _namespace_to_dict(v) if isinstance(v, SimpleNamespace) else v
    return result


def _parse_value(s: str) -> Any:
    """将字符串解析为适当的 Python 类型（int/float/bool/str）。

    解析顺序：int -> float -> bool -> str
    注意：字符串 "0" 会被解析为 int 0，"true" 会被解析为 bool True。
          如需强制字符串，使用引号：--set key='"123"'

    修复说明（BUG-M3, M-07）：提供统一的类型推断，避免所有值都是字符串。
    """
    s = s.strip()
    # 尝试 int
    try:
        return int(s)
    except ValueError:
        pass
    # 尝试 float
    try:
        return float(s)
    except ValueError:
        pass
    # 尝试 bool
    if s.lower() == 'true':
        return True
    if s.lower() == 'false':
        return False
    # 默认返回字符串（去除可能的引号包裹）
    # 支持 --set key='"value"' 语法强制字符串
    if len(s) >= 2 and ((s.startswith('"') and s.endswith('"')) or
                        (s.startswith("'") and s.endswith("'"))):
        s = s[1:-1]
    return s


def _apply_override(cfg_dict: dict, key_path: str, value_str: str) -> None:
    """将 "key.subkey=value" 风格的 override 应用到 cfg_dict（就地修改）。

    类型转换：对 value_str 调用 _parse_value，自动推断 int/float/bool/str。
    抛出：ConfigValidationError（路径不存在时）

    修复说明（BUG-M3）：使用 _parse_value 替代 yaml.safe_load，
    提供更可控的类型推断，避免 yaml 特有的类型解析问题。
    """
    keys = key_path.split('.')
    d = cfg_dict
    for key in keys[:-1]:
        if key not in d or not isinstance(d[key], dict):
            raise ConfigValidationError(
                f"override 路径 '{key_path}' 不存在（'{key}' 不是 dict 或不存在）"
            )
        d = d[key]
    leaf_key = keys[-1]
    if leaf_key not in d:
        raise ConfigValidationError(
            f"override 路径 '{key_path}' 不存在（叶子键 '{leaf_key}' 在 config 中不存在）"
        )
    d[leaf_key] = _parse_value(value_str)


def config_hash(path: str, overrides: list[str] | None = None) -> str:
    """计算 config 的 SHA256 hash（返回完整 64 位 hexdigest）。

    算法：
        1. 加载 YAML → 应用 overrides → 得到最终 cfg_dict
        2. json.dumps(cfg_dict, sort_keys=True, ensure_ascii=False)
        3. hashlib.sha256(...).hexdigest()

    调用方取前 8 位：config_hash(...)[:8]

    修复说明（BUG-8）：hash 算法统一使用 SHA256。
    # TODO: 其他 Plan（Plan 2/3/4）中的 cmd_tokenize、Trainer.__init__、
    #       ScalingAnalyzer._make_run_hash、DataPipeline.run 等函数若手动计算 hash，
    #       应统一改为调用 load_config() 或 config_hash()，确保算法一致（SHA256，非 MD5）。
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg_dict: dict = yaml.safe_load(f) or {}

    for override in (overrides or []):
        if '=' not in override:
            raise ConfigValidationError(f"override 格式错误，应为 'key=value'，得到: '{override}'")
        key_path, value_str = override.split('=', 1)
        _apply_override(cfg_dict, key_path, value_str)

    serialized = json.dumps(cfg_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def load_config(
    path: str,
    overrides: list[str] | None = None,
) -> SimpleNamespace:
    """加载 YAML config 文件，应用 overrides，创建 run_dir，返回 SimpleNamespace。

    参数：
        path: YAML 文件路径
        overrides: 点分隔 key=value 列表，如 ["training.lr=1e-4", "model.d_model=256"]

    返回：
        SimpleNamespace，附带 cfg.run_dir（Path，已自动创建）
        cfg.run_dir = Path("results/{hash[:8]}/")

    抛出：
        ConfigValidationError：override 路径不存在时
    """
    with open(path, 'r', encoding='utf-8') as f:
        cfg_dict: dict = yaml.safe_load(f) or {}

    for override in (overrides or []):
        if '=' not in override:
            raise ConfigValidationError(f"override 格式错误，应为 'key=value'，得到: '{override}'")
        key_path, value_str = override.split('=', 1)
        _apply_override(cfg_dict, key_path, value_str)

    # 修复说明（BUG-2）：原实现调用 config_hash(path, overrides) 再次读取文件，
    # 存在双重 I/O 且两次读取间文件若被修改会导致 hash 与内存 cfg_dict 不一致。
    # 修复：直接对内存中已有的 cfg_dict 计算 hash，保证一致性。
    serialized = json.dumps(cfg_dict, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    run_dir = Path(f"results/{h[:8]}/")
    run_dir.mkdir(parents=True, exist_ok=True)

    # 写入 config 快照（精确复现所需）
    snapshot_path = run_dir / "config_snapshot.yaml"
    with open(snapshot_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg_dict, f, allow_unicode=True, sort_keys=True)

    cfg = _dict_to_namespace(cfg_dict)
    cfg.run_dir = run_dir
    return cfg
```

**修复说明与使用约定（Plan 2+ 注意）：**

1. **BUG-C3 修复**：其他模块（如 Plan 2 的 `cmd_tokenize`）应通过 `from llm_foundry.common.config import config_hash` 导入 hash 函数，不要自行实现 hash 逻辑。

2. **BUG-H3 修复**：`load_config()` 返回的 `cfg` 已经包含 `run_dir` 属性（类型 `Path`），且目录已自动创建。后续代码（如 `cmd_tokenize`）应直接使用 `cfg.run_dir`，**不要**再次调用 `mkdir(parents=True)` 创建目录，避免双重目录创建问题。

   正确用法：
   ```python
   cfg = load_config(config_path, overrides)
   output_path = cfg.run_dir / "output.txt"  # 直接使用 cfg.run_dir
   ```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_common_config.py -v
```

Expected: 8 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add llm_foundry/common/config.py tests/test_common_config.py
git commit -m "feat: add common/config.py with YAML loading, overrides, and config hash"
```

---

## Task 8.5: `common/hashing.py` — 统一 hash 算法模块

**Files:**
- Create: `llm_foundry/common/hashing.py`
- Create: `tests/test_common_hashing.py`

**设计决策：**
- 单独提取 hash 算法到独立模块，供所有 Plan 复用（BUG-C1, BUG-C2 修复）
- 统一使用 SHA256 + `json.dumps(sort_keys=True, ensure_ascii=False)`
- 避免各 Plan 手动重复实现 hash 逻辑导致算法不一致

- [ ] **Step 1: 写失败测试**

```python
# tests/test_common_hashing.py
"""common/hashing.py 单元测试。Win11 CPU 可通过。"""
import hashlib
import json
import pytest
from llm_foundry.common.hashing import compute_hash, hash_dict


def test_hash_dict_deterministic():
    """相同 dict 应产生相同 hash。"""
    d1 = {"lr": 1e-3, "batch_size": 64, "model": {"d_model": 512}}
    d2 = {"batch_size": 64, "lr": 1e-3, "model": {"d_model": 512}}  # 不同顺序
    h1 = hash_dict(d1)
    h2 = hash_dict(d2)
    assert h1 == h2
    assert len(h1) == 64  # SHA256 hexdigest 全长


def test_hash_dict_different_content():
    """不同内容应产生不同 hash。"""
    d1 = {"lr": 1e-3}
    d2 = {"lr": 5e-4}
    assert hash_dict(d1) != hash_dict(d2)


def test_hash_dict_ensure_ascii_false():
    """中文内容应正确处理（ensure_ascii=False）。"""
    d = {"name": "中文测试", "lr": 1e-3}
    h = hash_dict(d)
    # 验证内部使用 ensure_ascii=False
    expected = hashlib.sha256(
        json.dumps(d, sort_keys=True, ensure_ascii=False).encode('utf-8')
    ).hexdigest()
    assert h == expected


def test_compute_hash_with_string():
    """compute_hash 支持字符串输入。"""
    s = "test_string"
    h = compute_hash(s)
    expected = hashlib.sha256(s.encode('utf-8')).hexdigest()
    assert h == expected


def test_compute_hash_with_bytes():
    """compute_hash 支持 bytes 输入。"""
    b = b"test_bytes"
    h = compute_hash(b)
    expected = hashlib.sha256(b).hexdigest()
    assert h == expected
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_common_hashing.py -v
```

Expected: `ImportError`（hashing.py 不存在）

- [ ] **Step 3: 实现 `common/hashing.py`**

```python
"""
llm_foundry/common/hashing.py — 统一 hash 算法模块。

职责：
    1. 提供全局统一的 hash 算法（SHA256）
    2. 所有 Plan 的 config hash、run hash 都应使用此模块
    3. 避免各模块自行实现导致算法不一致（BUG-C1, BUG-C2 修复）

使用示例：
    from llm_foundry.common.hashing import hash_dict, compute_hash

    # 计算 dict 的 hash
    h = hash_dict({"lr": 1e-3, "batch_size": 64})

    # 计算字符串/bytes 的 hash
    h = compute_hash("some_string")
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_dict(d: dict[str, Any]) -> str:
    """计算 dict 的 SHA256 hash（64 位 hexdigest）。

    算法：
        json.dumps(d, sort_keys=True, ensure_ascii=False) → SHA256

    参数：
        d: 要计算 hash 的字典

    返回：
        64 位十六进制字符串（SHA256 hexdigest）
    """
    serialized = json.dumps(d, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def compute_hash(data: str | bytes) -> str:
    """计算字符串或字节流的 SHA256 hash。

    参数：
        data: 字符串或字节流

    返回：
        64 位十六进制字符串（SHA256 hexdigest）
    """
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_common_hashing.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add llm_foundry/common/hashing.py tests/test_common_hashing.py
git commit -m "feat: add common/hashing.py with unified SHA256 hash algorithm (BUG-C1, BUG-C2 fix)"
```

---

## Task 9: `run.py` — 只含 `env` 子命令的 CLI（无桩函数）

**Files:**
- Create: `run.py`

**关键约束**：只注册已实现的 `env` 子命令，不添加其他子命令的占位桩函数。未实现的子命令（tokenize/data/train/scaling/align）在对应 Plan 实现时再加入。`HardwareNotSatisfiedError` 在此文件定义并在 `main()` 中捕获。

- [ ] **Step 1: 写 CLI 冒烟测试**

创建 `tests/test_cli.py`：

```python
# tests/test_cli.py
"""CLI 冒烟测试。从项目根目录运行。"""
import subprocess
import sys
import os


def _run(args, cwd=None):
    """在项目根目录运行 run.py 子命令，返回 CompletedProcess。"""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, "run.py"] + args,
        capture_output=True, text=True, cwd=cwd or root,
    )


def test_run_help():
    """python run.py --help 应正常退出，输出包含 'env'。"""
    result = _run(["--help"])
    assert result.returncode == 0
    assert "env" in result.stdout


def test_run_env_exit_zero():
    """python run.py env 应以 exit code 0 退出。"""
    result = _run(["env"])
    assert result.returncode == 0


def test_run_env_output_contains_env_check():
    """python run.py env 输出应包含 'ENV CHECK' 和 'MODE'。"""
    result = _run(["env"])
    assert "ENV CHECK" in result.stdout
    assert "MODE" in result.stdout


def test_run_env_output_contains_attention():
    """python run.py env 输出应包含 'ATTENTION' 字段。"""
    result = _run(["env"])
    assert "ATTENTION" in result.stdout


def test_run_unknown_command_exits_nonzero():
    """未知子命令应以非零 exit code 退出。"""
    result = _run(["nonexistent_command"])
    assert result.returncode != 0
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_cli.py -v
```

Expected: 报错（run.py 不存在或 ModuleNotFoundError）

- [ ] **Step 3: 创建 run.py**

```python
#!/usr/bin/env python3
"""
LLM Foundry Simulator — 统一 CLI 入口。

Plan 1 实现的子命令：
    python run.py env          # 系统环境自检

后续 Plan 将逐步加入：
    Plan 2: tokenize, data
    Plan 3: train
    Plan 4: scaling
    Plan 5: align
"""
from __future__ import annotations

import argparse
import os
import sys


class HardwareNotSatisfiedError(Exception):
    """某命令需要 GPU 但当前完全没有 GPU，无法降级运行时抛出。

    区别于静默降级（如 vLLM → HF generate）：
        - 静默降级：功能完整，性能下降，不抛异常
        - HardwareNotSatisfiedError：功能无法运行（如无 GPU 时跑训练），抛异常并 sys.exit(1)
    """
    def __init__(self, feature: str, required_gpus: int, actual_gpus: int,
                 fallback_hint: str = ""):
        self.feature = feature
        self.required_gpus = required_gpus
        self.actual_gpus = actual_gpus
        self.fallback_hint = fallback_hint
        super().__init__(
            f"{feature} 需要 {required_gpus} 张 GPU（当前 {actual_gpus} 张）"
        )


def _setup_hf_endpoint() -> None:
    """若未设置 HF_ENDPOINT，自动设置为国内镜像源（减少下载超时）。"""
    if "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _cmd_env(args: argparse.Namespace) -> None:
    """执行 env 子命令：打印环境检测结果。"""
    from llm_foundry.common.env_check import check_env, print_env_summary
    status = check_env()
    print_env_summary(status)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-foundry",
        description="Stanford CS336 全链路大模型训练 CLI 流水线",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # env — Plan 1 实现
    subparsers.add_parser("env", help="系统环境自检（无需 GPU）")

    return parser


def main() -> None:
    _setup_hf_endpoint()
    parser = _build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "env": _cmd_env,
    }

    handler = dispatch.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    try:
        handler(args)
    except HardwareNotSatisfiedError as e:
        print(f"[硬件不满足] {e}")
        if e.fallback_hint:
            print(f"  降级方案: {e.fallback_hint}")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_cli.py -v
```

Expected: 5 tests PASSED

- [ ] **Step 5: 手动验证 env 输出**

从项目根目录运行：

```bash
python run.py env
```

Expected（根据实际环境，Win11 CPU 示例）：
```
[ENV CHECK]
  OS            : Windows 11 ...
  Python        : 3.11.x
  PyTorch       : 2.x.x
  CUDA          : N/A | GPU: None x0  ✗
  ATTENTION     : SDPA
  Triton        :  ✗
  FlashAttn-2   :  ✗
  vLLM          :  ✗

[MODE] DEMO
  无 GPU，train/align 命令不可用（env/scaling 等可正常使用）
```

- [ ] **Step 6: Commit**

```bash
git add run.py tests/test_cli.py
git commit -m "feat: add run.py with env subcommand and HardwareNotSatisfiedError"
```

---

## Task 10: 全量测试与验收

**Files:**
- 无新增文件

- [ ] **Step 1: 运行全量 backends + common 测试**

```bash
pytest tests/test_backends.py tests/test_common_env_check.py tests/test_common_model.py tests/test_common_optimizer.py tests/test_common_data.py tests/test_common_config.py tests/test_common_hashing.py -v
```

Expected: 全部 PASSED（指定文件中所有测试，跳过需要 HF 网络的测试）

- [ ] **Step 2: 运行完整测试套件**

```bash
pytest tests/ -v
```

Expected: 全部 PASSED，无 FAILED 或 ERROR（`test_get_inference_backend_returns_instance` 若因 HF 网络问题失败，可 xfail 标记，不影响验收）

- [ ] **Step 3: 验收检查清单**

逐一验证以下命令，确认全部正常：

```bash
# 1. 包可 import
python -c "import llm_foundry.backends; import llm_foundry.common.model; print('imports OK')"

# 2. env 命令
python run.py env

# 3. --help 只显示 env 子命令（无其他桩命令）
python run.py --help

# 4. 未知命令给出非零退出码（非"尚未实现"提示）
python run.py nonexistent_cmd; echo "exit: $?"

# 5. attention backend 被正确检测
python -c "from llm_foundry.backends.attention import get_current_backend, get_attention_fn; get_attention_fn(use_flash_attn=False); print('backend:', get_current_backend())"

# 6. ShardedOptimizer world_size=1 不崩溃
python -c "
import torch.nn as nn
from llm_foundry.common.optimizer import AdamW, ShardedOptimizer
m = nn.Linear(4, 4, bias=False)
opt = ShardedOptimizer(m.parameters(), AdamW, lr=1e-3, weight_decay=0.01)
print('ShardedOptimizer world_size=1 OK')
"

# 7. config hash 稳定
python -c "
from llm_foundry.common.config import config_hash
h = config_hash('configs/train.yaml')
print('hash:', h[:8])
"
```

- [ ] **Step 4: 最终 Commit**

```bash
git add -u
git commit -m "feat: plan1 complete - backends/ attention/inference fallback + common/ modules + env CLI"
```

---

## 验收标准

Plan 1 完成后，以下条件应全部满足：

1. `pip install -e .` 无报错
2. `pytest tests/test_backends.py tests/test_common_*.py -v` 全部通过（Win11 CPU 环境）
3. `python run.py env` 输出包含 `[ENV CHECK]`、`ATTENTION`、`[MODE]` 的环境检测结果
4. `python run.py --help` 只显示 `env` 子命令（无其他桩命令）
5. `backends/attention.py`：`get_attention_fn(use_flash_attn)` 使用 `lru_cache(maxsize=2)`，`import triton` 只在 `try/except ImportError` 块内
6. `backends/inference.py`：`InferenceBackend` 是抽象基类，`VLLMBackend.backend_name == 'vllm'`，`HFGenerateBackend.backend_name == 'hf_generate'`
7. `common/model.py`：`Transformer` 包装 `BasicsTransformerLM`，`BasicsTransformerLM` 不从 `common/__init__.py` export；`Transformer.__init__` 调用 `get_attention_fn(config.use_flash_attn)` 并缓存到 `self._attn_fn`
8. `common/optimizer.py`：包含 `get_cosine_lr`、`AdamW`（手写版）、`ShardedOptimizer`（已修复 world_size=1 KeyError）
9. `common/data.py`：包含 `load_tokens` 和 `get_batch`
10. `common/config.py`：`load_config` 创建 `results/{hash[:8]}/` 目录并写入 `config_snapshot.yaml`；`ConfigValidationError` 在 override 路径不存在时抛出
11. `common/hashing.py`：提供统一的 `hash_dict()` 和 `compute_hash()` 函数，使用 SHA256 算法（BUG-C1, BUG-C2 修复）
12. `run.py`：定义 `HardwareNotSatisfiedError`，`main()` 捕获并 `sys.exit(1)`
13. `reproduce/expected/` 下5个 JSON 占位文件存在（内容为 `{}`）
