# Plan 2: BPETokenizer + DataPipeline Implementation Plan

> **For agentic workers (OMC):** 推荐使用 `/oh-my-claudecode:ralph`（自循环执行直到完成，适合多步骤 task）或 `/oh-my-claudecode:ultrawork`（并行高吞吐执行，适合独立任务批量完成，复杂 task 加 `model=opus`）。步骤使用 checkbox (`- [ ]`) 语法跟踪进度，完成后用 TaskUpdate 标记 completed。

**Goal:** 实现 `stage1_tokenize/tokenizer.py`（BPETokenizer）和 `stage4_data/pipeline.py`（DataPipeline），并在 `run.py` 中新增 `tokenize` 和 `data` 子命令（Plan 1 只有 `env` 子命令），使得 `python run.py tokenize --config configs/tokenize.yaml` 和 `python run.py data --config configs/data.yaml` 均可对小样本数据正常运行并产出 artifact。

**Architecture:** 两条独立流水线：Stage 1 将 Assignment1 的 `train_bpe` + `Tokenizer` 重构为单一 `BPETokenizer` 类（train/encode/decode/save/load），Stage 4 将 Assignment4 的多个过滤函数整合为 `DataPipeline` 类（无重量级 fastText 依赖，改用纯 Python Gopher 规则 + 长度过滤 + SHA256 去重）。两条流水线共享 `run.py` 的 `--config / --set` 参数框架和 `results/{hash}/` 输出约定。

**Tech Stack:** Python 3.10+, regex, pytest, pyyaml, hashlib（内置）

---

## 前置条件

Plan 1 已完成，以下条件成立：
- `llm_foundry/` 包已安装（`pip install -e .`），且 `pyproject.toml` 的 `dependencies` 已含 `"regex>=2023.0"`；若未含，需先在 `pyproject.toml` 中添加：
  ```toml
  dependencies = [
      "pyyaml>=6.0",
      "regex>=2023.0",
  ]
  ```
  然后重新安装：`pip install -e . regex`
- `llm_foundry/stage1_tokenize/__init__.py` 已存在（空文件）
- `llm_foundry/stage4_data/__init__.py` 已存在（空文件）
- `run.py` 已存在，目前只含 `env` 子命令（Plan 1 不含 tokenize/data 子命令）
- `configs/tokenize.yaml` 和 `configs/data.yaml` 已存在（含初始占位内容）

---

## 文件映射

**新建文件：**
- `llm_foundry/stage1_tokenize/tokenizer.py` — `BPETokenizer` 类（train/encode/decode/save/from_pretrained）
- `llm_foundry/stage4_data/pipeline.py` — `DataPipeline` 类（run 方法，Gopher 过滤 + 长度过滤 + SHA256 去重）
- `tests/test_stages.py` — Stage 1 和 Stage 4 的集成测试
- `data/sample.txt` — 供 tokenize 命令使用的小样本文本（20 KB 以内）
- `data/sample.jsonl` — 供 data 命令使用的小样本 JSONL（10 条文档）

**修改文件：**
- `llm_foundry/stage1_tokenize/__init__.py` — 导出 `BPETokenizer`
- `llm_foundry/stage4_data/__init__.py` — 导出 `DataPipeline`
- `configs/tokenize.yaml` — 填充真实配置（text_file / vocab_size / output.base_dir）
- `configs/data.yaml` — 填充真实配置（input_path / filters / dedup / output.base_dir）
- `run.py` — 新增 `cmd_tokenize` 和 `cmd_data` 子命令（Plan 1 中不存在此命令）

**参考来源（只读，不修改）：**
- `reference_resource/Assignment1-basics/Transformer/bpe.py` — `train_bpe`, `find_chunk_boundaries`, `_process_chunk`
- `reference_resource/Assignment1-basics/Transformer/tokenizer.py` — `Tokenizer` 类（`_encode_chunk`, `encode`, `decode`）
- `reference_resource/Assignment4-data/cs336_data/filters.py` — `gopher_quality_filter`, `normalize_text`, `get_ngrams` 等
- `reference_resource/Assignment4-data/cs336_data/run_pipeline.py` — 过滤流水线参考结构

---

## Task 1: 创建小样本测试数据文件

**Files:**
- Create: `data/sample.txt`
- Create: `data/sample.jsonl`

- [ ] **Step 1: 创建 `data/` 目录并写入 `data/sample.txt`**

该文件用于训练 BPE 分词器，需要足够量的英文文本（约 5 KB）。

```bash
mkdir -p data
```

然后创建 `data/sample.txt`，内容为若干段英文文章拼接（段落间用 `<|endoftext|>` 分隔）：

```
The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols.
<|endoftext|>
Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves.
<|endoftext|>
Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural networks, and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, and audio recognition.
<|endoftext|>
Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them.
<|endoftext|>
A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing and computer vision. Transformers were introduced in 2017 by a team at Google Brain and are now the dominant architecture for language models.
<|endoftext|>
The attention mechanism is the core innovation in transformer models. It allows the model to focus on different parts of the input sequence when computing representations. The scaled dot-product attention computes attention weights by taking the dot product of query and key vectors, scaling by the square root of the key dimension, and applying a softmax function.
<|endoftext|>
Byte pair encoding is a data compression algorithm that iteratively replaces the most frequent pair of bytes in a sequence with a single unused byte. It has been adapted for use in natural language processing as a subword tokenization algorithm. BPE builds a vocabulary by starting with individual characters and merging the most frequent adjacent pairs until the desired vocabulary size is reached.
<|endoftext|>
The training of large language models requires massive amounts of compute and data. Typically, models are trained on trillions of tokens of text data collected from the internet. The data pipeline involves crawling, extracting, filtering, deduplicating, and tokenizing text before it can be used for training.
<|endoftext|>
Reinforcement learning from human feedback is a technique used to align language models with human values and preferences. In this approach, human raters provide feedback on model outputs, which is used to train a reward model that is then used to fine-tune the language model using reinforcement learning algorithms.
<|endoftext|>
Scaling laws describe how the performance of machine learning models improves as a function of model size, dataset size, and compute budget. Empirical research has shown that larger models trained on more data generally achieve better performance, with predictable relationships between these quantities.
<|endoftext|>
```

重复以上文本块 5 次（直接粘贴同样内容 5 遍），使文件总大小约 5 KB，确保 BPE 训练时有足够的词对频次。

- [ ] **Step 2: 创建 `data/sample.jsonl`**

每行一个 JSON 对象，格式为 `{"text": "..."}` ，共写入 10 条文档（各条长度在 50～500 词之间）。包含 2 条"应被过滤掉"的文档（长度 < 100 字符）以验证过滤逻辑。

```jsonl
{"text": "The history of artificial intelligence began in antiquity, with myths, stories and rumors of artificial beings endowed with intelligence or consciousness by master craftsmen. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as the mechanical manipulation of symbols. This laid the groundwork for what would become one of the most transformative technologies of the modern era."}
{"text": "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. Machine learning focuses on the development of computer programs that can access data and use it to learn for themselves. The process begins with observations or data, such as examples, direct experience, or instruction, so that computers can make better decisions in the future."}
{"text": "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning. Learning can be supervised, semi-supervised or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural networks, and convolutional neural networks have been applied to fields including computer vision, speech recognition, natural language processing, and audio recognition where they have produced results comparable to and in some cases surpassing human expert performance."}
{"text": "Natural language processing is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language, in particular how to program computers to process and analyze large amounts of natural language data. The goal is a computer capable of understanding the contents of documents, including the contextual nuances of the language within them. The technology can then accurately extract information and insights contained in the documents, as well as categorize and organize the documents themselves."}
{"text": "A transformer is a deep learning model that adopts the mechanism of self-attention, differentially weighting the significance of each part of the input data. It is used primarily in the fields of natural language processing and computer vision. Transformers were introduced in 2017 by a team at Google Brain and are now the dominant architecture for language models, having replaced previous recurrent neural network approaches due to their ability to process sequences in parallel during training."}
{"text": "Byte pair encoding is a data compression algorithm that has been adapted for use as a subword tokenization method in natural language processing. It builds a vocabulary by starting with individual characters and merging the most frequent adjacent pairs iteratively until the desired vocabulary size is reached. This approach allows the model to handle rare and out-of-vocabulary words by decomposing them into subword units that appear frequently in the training data."}
{"text": "Reinforcement learning from human feedback is a technique used to align language models with human values and preferences. A reward model is trained using human preference data, and then used to fine-tune a language model using proximal policy optimization or similar algorithms. This approach has been used to create models like InstructGPT and ChatGPT that follow instructions and avoid harmful outputs."}
{"text": "Scaling laws in machine learning describe how model performance improves predictably as a function of model parameters, training data size, and compute budget. Research from OpenAI and DeepMind has shown these relationships follow power laws, enabling researchers to predict the performance of larger models before training them. The Chinchilla scaling laws suggest that many large language models were undertrained relative to their parameter count."}
{"text": "short"}
{"text": "Too short to pass filter."}
```

- [ ] **Step 3: Commit**

```bash
git add data/sample.txt data/sample.jsonl
git commit -m "feat: add sample data files for tokenize and data pipeline testing"
```

---

## Task 2: `stage1_tokenize/tokenizer.py` — BPETokenizer

**Files:**
- Create: `llm_foundry/stage1_tokenize/tokenizer.py`
- Modify: `llm_foundry/stage1_tokenize/__init__.py`

参考：
- `reference_resource/Assignment1-basics/Transformer/bpe.py`（`train_bpe`, `find_chunk_boundaries`, `_process_chunk`）
- `reference_resource/Assignment1-basics/Transformer/tokenizer.py`（`Tokenizer._encode_chunk`, `encode`, `decode`）

**关键设计决策：**
1. `BPETokenizer` 是单一类，内部复用 `bpe.py` 中的 `train_bpe` 函数（不重写算法）
2. 序列化格式为 JSON（非 pickle），以保证人类可读和跨平台安全
3. JSON 中 vocab key 为 int 字符串（JSON 不支持 int key），load 时需转换
4. vocab values 为 base64 编码字符串（bytes 无法直接 JSON 序列化）
5. merges 为 `[[base64_a, base64_b], ...]` 列表

- [ ] **Step 1: 写失败测试（仅测结构，不测算法正确性）**

在 `tests/test_stages.py` 中加入 Stage 1 测试（文件不存在则新建）：

```python
# tests/test_stages.py
"""
Stage 1 (tokenizer) 和 Stage 4 (data pipeline) 集成测试。

测试策略：只验证文件存在 + 格式正确，不验证数值精度。
耗时约束：全部测试应在 CPU 上 30 秒内完成。
"""
import json
import os
import pytest
import tempfile
import pathlib


# ─────────────────────────────────────────
# Stage 1: BPETokenizer 测试
# ─────────────────────────────────────────

def test_bpe_tokenizer_import():
    """BPETokenizer 可以正常 import。"""
    from llm_foundry.stage1_tokenize import BPETokenizer
    assert BPETokenizer is not None


def test_bpe_tokenizer_train_produces_json(tmp_path):
    """train() 后 save() 产出有效 JSON 文件，包含 vocab 和 merges 字段。"""
    from llm_foundry.stage1_tokenize import BPETokenizer

    # 写一个最小化的临时文本文件（反复几段英文保证有足够词对）
    text_file = tmp_path / "corpus.txt"
    sample = (
        "the quick brown fox jumps over the lazy dog. "
        "the fox and the dog are friends. "
    ) * 30  # 重复 30 次，保证词对频次足够
    text_file.write_text(sample, encoding="utf-8")

    tok = BPETokenizer()
    tok.train(str(text_file), vocab_size=300)

    out_path = str(tmp_path / "tokenizer.json")
    tok.save(out_path)

    assert os.path.exists(out_path), "tokenizer.json 不存在"
    with open(out_path, encoding="utf-8") as f:
        data = json.load(f)

    assert "vocab" in data, "tokenizer.json 缺少 'vocab' 字段"
    assert "merges" in data, "tokenizer.json 缺少 'merges' 字段"
    assert len(data["vocab"]) >= 256, "vocab 至少应有 256 个基础字节 token"


def test_bpe_tokenizer_encode_decode_roundtrip(tmp_path):
    """train() -> encode() -> decode() 应能还原原文（英文 ASCII 文本）。"""
    from llm_foundry.stage1_tokenize import BPETokenizer

    text_file = tmp_path / "corpus.txt"
    text_file.write_text(
        ("hello world this is a test sentence for bpe tokenizer. " * 20),
        encoding="utf-8",
    )

    tok = BPETokenizer()
    tok.train(str(text_file), vocab_size=300)

    test_text = "hello world"
    ids = tok.encode(test_text)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)

    decoded = tok.decode(ids)
    assert decoded == test_text, f"解码不一致：期望 {test_text!r}，实际 {decoded!r}"


def test_bpe_tokenizer_from_pretrained_roundtrip(tmp_path):
    """save() 后 from_pretrained() 加载，encode 结果应一致。"""
    from llm_foundry.stage1_tokenize import BPETokenizer

    text_file = tmp_path / "corpus.txt"
    text_file.write_text(
        ("the quick brown fox jumps over the lazy dog. " * 30),
        encoding="utf-8",
    )

    tok = BPETokenizer()
    tok.train(str(text_file), vocab_size=300)
    out_path = str(tmp_path / "tokenizer.json")
    tok.save(out_path)

    tok2 = BPETokenizer.from_pretrained(out_path)
    test_text = "the quick"
    assert tok.encode(test_text) == tok2.encode(test_text), "加载后 encode 结果不一致"
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
pytest tests/test_stages.py::test_bpe_tokenizer_import -v
```

Expected: `ImportError`（`tokenizer.py` 不存在）

- [ ] **Step 3: 实现 `stage1_tokenize/tokenizer.py`**

创建 `llm_foundry/stage1_tokenize/tokenizer.py`，内容如下：

```python
"""
stage1_tokenize/tokenizer.py — BPETokenizer

将 Assignment1 的 train_bpe + Tokenizer 重构为单一类。
序列化格式：JSON（vocab 的 bytes value 用 base64 编码）。

参考来源：
  - reference_resource/Assignment1-basics/Transformer/bpe.py（train_bpe）
  - reference_resource/Assignment1-basics/Transformer/tokenizer.py（encode/decode）
"""
from __future__ import annotations

import base64
import json
import os
import sys

import regex as re

# ─────────────────────────────────────────
# GPT-2 预分词正则表达式
# 与 Assignment1 bpe.py / tokenizer.py 保持完全一致
# ─────────────────────────────────────────
GPT2_PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    """
    基于 BPE（Byte Pair Encoding）的分词器。

    生命周期：
      1. 实例化：`tok = BPETokenizer()`
      2. 训练：`tok.train(text_file, vocab_size=1000)`
         —— 内部调用 train_bpe（来自 Assignment1 bpe.py），学习合并规则
      3. 使用：`tok.encode(text)` / `tok.decode(ids)`
      4. 持久化：`tok.save(path)` / `BPETokenizer.from_pretrained(path)`

    序列化格式（JSON）：
      {
        "vocab":  {"0": "<base64>", "1": "<base64>", ...},   # int id -> base64(bytes)
        "merges": [["<base64_a>", "<base64_b>"], ...]         # 按合并顺序排列
      }
    """

    def __init__(self) -> None:
        # vocab: {token_id (int) -> bytes}
        self.vocab: dict[int, bytes] = {}
        # merges: [(bytes_a, bytes_b), ...] 按优先级排列（索引越小优先级越高）
        self.merges: list[tuple[bytes, bytes]] = []
        # 反向词表：bytes -> token_id，用于 encode 时快速查找
        self._inverse_vocab: dict[bytes, int] = {}
        # 合并规则排名：pair -> rank（rank 越小优先级越高）
        self._merges_rank: dict[tuple[bytes, bytes], int] = {}

    # ─────────────────────────────────────────
    # train
    # ─────────────────────────────────────────
    def train(self, text_file: str, vocab_size: int) -> None:
        """从文本文件训练 BPE 分词器。

        Args:
            text_file: UTF-8 编码的文本文件路径
            vocab_size: 目标词表大小（须 >= 256）
        """
        # 内部调用：从本地 stage1_tokenize/bpe.py 导入 train_bpe
        # DESIGN-1 修复：已删除 sys.path.insert，改为本地 import，无路径脆弱性
        vocab, merges = _train_bpe(text_file, vocab_size, special_tokens=[])
        self.vocab = vocab
        self.merges = merges
        self._build_indices()

    def _build_indices(self) -> None:
        """从 self.vocab 和 self.merges 构建查找索引，train/load 后都需调用。"""
        self._inverse_vocab = {v: k for k, v in self.vocab.items()}
        self._merges_rank = {pair: i for i, pair in enumerate(self.merges)}

    # ─────────────────────────────────────────
    # encode / decode
    # ─────────────────────────────────────────
    def encode(self, text: str) -> list[int]:
        """将文本编码为 token ID 列表。"""
        return _encode_chunk(text, self._inverse_vocab, self._merges_rank)

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        """将 token ID 列表解码为文本，支持 skip_special_tokens 参数以兼容 HF 风格调用。

        Args:
            ids: token ID 列表
            skip_special_tokens: 是否跳过特殊 token（BPETokenizer 暂无特殊 token，参数为 API 兼容性保留）
        """
        byte_parts = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(f"未知 token id: {token_id}")
            byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def tokenize_file(self, text_file: str, output_path: str, dtype: str = "uint16") -> None:
        """将文本文件编码为 token ID 数组并保存为 .npy 文件，供训练阶段使用（C-02 修复）。

        Args:
            text_file: 输入文本文件路径
            output_path: 输出 .npy 文件路径
            dtype: 数组数据类型，默认 uint16（支持 vocab_size 最大 65535）
        """
        import numpy as np

        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        ids = self.encode(text)
        np.save(output_path, np.array(ids, dtype=getattr(np, dtype)))

    # ─────────────────────────────────────────
    # save / from_pretrained
    # ─────────────────────────────────────────
    def save(self, path: str) -> None:
        """将 vocab 和 merges 序列化为 JSON 文件。

        bytes 值使用 base64 编码以满足 JSON 可序列化要求。
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        data = {
            "vocab": {
                str(k): base64.b64encode(v).decode("ascii")
                for k, v in self.vocab.items()
            },
            "merges": [
                [
                    base64.b64encode(a).decode("ascii"),
                    base64.b64encode(b).decode("ascii"),
                ]
                for a, b in self.merges
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True)

    @classmethod
    def from_pretrained(cls, path: str) -> "BPETokenizer":
        """从 JSON 文件加载已训练的分词器。"""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls()
        tok.vocab = {
            int(k): base64.b64decode(v) for k, v in data["vocab"].items()
        }
        tok.merges = [
            (base64.b64decode(a), base64.b64decode(b))
            for a, b in data["merges"]
        ]
        tok._build_indices()
        return tok


# ─────────────────────────────────────────
# 模块级私有辅助函数（不作为公开 API）
# 从 Assignment1 的 bpe.py / tokenizer.py 提取，保留原始逻辑
# ─────────────────────────────────────────

def _train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """调用本地 stage1_tokenize/bpe.py 中的 train_bpe 函数。

    DESIGN-1 修复：将 train_bpe 函数从
    reference_resource/Assignment1-basics/Transformer/bpe.py
    复制到 llm_foundry/stage1_tokenize/bpe.py，改为本地 import。
    删除了原有的 sys.path.insert 方案（路径脆弱、多线程不安全）。
    """
    from llm_foundry.stage1_tokenize.bpe import train_bpe  # 本地副本，无 sys.path 污染
    return train_bpe(input_path, vocab_size, special_tokens)


def _encode_chunk(
    text: str,
    inverse_vocab: dict[bytes, int],
    merges_rank: dict[tuple[bytes, bytes], int],
) -> list[int]:
    """对文本执行 BPE 编码（贪心合并）。

    逻辑来自 reference_resource/Assignment1-basics/Transformer/tokenizer.py
    的 Tokenizer._encode_chunk 方法，此处改写为独立函数避免重复依赖。
    """
    ids: list[int] = []

    for match in re.finditer(GPT2_PAT, text):
        token_bytes = match.group().encode("utf-8")
        b_list: list[bytes] = [bytes([b]) for b in token_bytes]

        # 反复合并 rank 最低的 pair
        while len(b_list) >= 2:
            best_pair: tuple[bytes, bytes] | None = None
            min_rank = float("inf")
            for i in range(len(b_list) - 1):
                pair = (b_list[i], b_list[i + 1])
                rank = merges_rank.get(pair, float("inf"))
                if rank < min_rank:
                    best_pair = pair
                    min_rank = rank
            if best_pair is None:
                break
            # 替换所有出现的 best_pair
            i = 0
            new_b_list: list[bytes] = []
            while i < len(b_list):
                if (
                    i < len(b_list) - 1
                    and b_list[i] == best_pair[0]
                    and b_list[i + 1] == best_pair[1]
                ):
                    new_b_list.append(b_list[i] + b_list[i + 1])
                    i += 2
                else:
                    new_b_list.append(b_list[i])
                    i += 1
            b_list = new_b_list

        for b in b_list:
            if b not in inverse_vocab:
                raise ValueError(f"BPE merge 产生了未知 token: {b!r}")
            ids.append(inverse_vocab[b])

    return ids
```

- [ ] **Step 4: 更新 `llm_foundry/stage1_tokenize/__init__.py`**

```python
from llm_foundry.stage1_tokenize.tokenizer import BPETokenizer

__all__ = ["BPETokenizer"]
```

- [ ] **Step 5: 运行 Stage 1 测试，确认通过**

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
pytest tests/test_stages.py -k "bpe" -v
```

Expected: 4 tests PASSED（`test_bpe_tokenizer_import`, `test_bpe_tokenizer_train_produces_json`, `test_bpe_tokenizer_encode_decode_roundtrip`, `test_bpe_tokenizer_from_pretrained_roundtrip`）

- [ ] **Step 6: Commit**

```bash
git add llm_foundry/stage1_tokenize/tokenizer.py llm_foundry/stage1_tokenize/__init__.py tests/test_stages.py
git commit -m "feat: add BPETokenizer with train/encode/decode/save/from_pretrained"
```

---

## Task 3: `stage4_data/pipeline.py` — DataPipeline

**Files:**
- Create: `llm_foundry/stage4_data/pipeline.py`
- Modify: `llm_foundry/stage4_data/__init__.py`

参考：
- `reference_resource/Assignment4-data/cs336_data/filters.py`（`gopher_quality_filter`, `normalize_text`, `get_ngrams`）
- `reference_resource/Assignment4-data/cs336_data/run_pipeline.py`（过滤流水线结构参考）

**关键设计决策：**
1. **不依赖 fastText 模型**：fastText 二进制模型文件不随项目分发，且在 Win11 CPU 环境安装较复杂。Plan 2 的 DataPipeline 仅实现不需要外部模型文件的过滤步骤：长度过滤、Gopher 启发式规则过滤、SHA256 精确去重。语言识别和 NSFW 过滤留待后续 Plan（需要模型文件时）启用。
2. **输入格式**：每行 `{"text": "..."}` 的 JSONL 文件
3. **输出格式**：`results/{hash}/cleaned.jsonl`（保留通过过滤的文档）+ `results/{hash}/metrics.jsonl`（统计追加写入）
4. `{hash}` 为配置文件内容的 SHA256 前 8 位，确保不同配置的输出不互相覆盖（BUG-8 修复：统一使用 SHA256）

- [ ] **Step 1: 在 `tests/test_stages.py` 中追加 Stage 4 测试**

在已有文件末尾追加：

```python

# ─────────────────────────────────────────
# Stage 4: DataPipeline 测试
# ─────────────────────────────────────────

def test_data_pipeline_import():
    """DataPipeline 可以正常 import。"""
    from llm_foundry.stage4_data import DataPipeline
    assert DataPipeline is not None


def test_data_pipeline_run_produces_cleaned_jsonl(tmp_path):
    """run() 应产出 cleaned.jsonl，且内容为有效 JSONL（每行均可 json.loads）。"""
    from llm_foundry.stage4_data import DataPipeline

    # 构造 10 条文档：8 条正常（长度 > 100 字符）、2 条太短（应被过滤）
    input_file = tmp_path / "input.jsonl"
    docs = []
    long_doc = (
        "The quick brown fox jumps over the lazy dog. "
        "This sentence is repeated to ensure sufficient length for the filter. "
    ) * 5  # ~230 字符，远超 min_length=100
    for i in range(8):
        docs.append(json.dumps({"text": long_doc + f" Document {i}."}, ensure_ascii=False))
    docs.append(json.dumps({"text": "short"}))   # 应被过滤
    docs.append(json.dumps({"text": "too short"}))  # 应被过滤
    input_file.write_text("\n".join(docs), encoding="utf-8")

    cfg = {
        "data": {
            "input_path": str(input_file),
            "filters": {"min_length": 100, "max_length": 50000},
            "dedup": False,  # 去重关闭，简化测试
        },
        "output": {"base_dir": str(tmp_path / "results")},
    }

    pipeline = DataPipeline(cfg)
    stats = pipeline.run(str(input_file), str(tmp_path / "results"))

    # 验证输出目录下存在 cleaned.jsonl
    out_dir = pathlib.Path(stats["output_dir"])
    cleaned_path = out_dir / "cleaned.jsonl"
    assert cleaned_path.exists(), f"cleaned.jsonl 不存在: {cleaned_path}"

    # 验证每行均为有效 JSON
    lines = cleaned_path.read_text(encoding="utf-8").strip().split("\n")
    for line in lines:
        obj = json.loads(line)  # 若非合法 JSON 则抛出异常
        assert "text" in obj

    # 验证统计信息合理
    assert stats["input_docs"] == 10
    assert stats["output_docs"] <= stats["input_docs"]
    assert stats["output_docs"] >= 1  # 至少有一条通过


def test_data_pipeline_filters_short_docs(tmp_path):
    """min_length 过滤应丢弃短文档，保留长文档。"""
    from llm_foundry.stage4_data import DataPipeline

    input_file = tmp_path / "input.jsonl"
    long_text = "word " * 60  # ~300 字符
    short_text = "hi"
    lines = [
        json.dumps({"text": long_text}),
        json.dumps({"text": short_text}),
    ]
    input_file.write_text("\n".join(lines), encoding="utf-8")

    cfg = {
        "data": {
            "input_path": str(input_file),
            "filters": {"min_length": 100, "max_length": 50000},
            "dedup": False,
        },
        "output": {"base_dir": str(tmp_path / "results")},
    }
    pipeline = DataPipeline(cfg)
    stats = pipeline.run(str(input_file), str(tmp_path / "results"))

    assert stats["input_docs"] == 2
    assert stats["output_docs"] == 1, "只有长文档应通过 min_length 过滤"


def test_data_pipeline_dedup_removes_exact_duplicates(tmp_path):
    """dedup=True 时，精确重复文档应只保留一份。"""
    from llm_foundry.stage4_data import DataPipeline

    long_text = ("The quick brown fox jumps over the lazy dog. " * 5)
    input_file = tmp_path / "input.jsonl"
    # 3 条相同文档
    lines = [json.dumps({"text": long_text})] * 3
    input_file.write_text("\n".join(lines), encoding="utf-8")

    cfg = {
        "data": {
            "input_path": str(input_file),
            "filters": {"min_length": 10, "max_length": 50000},
            "dedup": True,
        },
        "output": {"base_dir": str(tmp_path / "results")},
    }
    pipeline = DataPipeline(cfg)
    stats = pipeline.run(str(input_file), str(tmp_path / "results"))

    assert stats["input_docs"] == 3
    assert stats["output_docs"] == 1, "3 条重复文档去重后应只剩 1 条"
```

- [ ] **Step 2: 运行 Stage 4 测试，确认失败**

```bash
pytest tests/test_stages.py -k "data_pipeline" -v
```

Expected: `ImportError`（`pipeline.py` 不存在）

- [ ] **Step 3: 实现 `stage4_data/pipeline.py`**

创建 `llm_foundry/stage4_data/pipeline.py`，内容如下：

```python
"""
stage4_data/pipeline.py — DataPipeline

将 Assignment4 的多个过滤函数整合为单一流水线类。

本 Plan 实现的过滤步骤（不依赖外部模型文件）：
  1. 长度过滤：按字符数过滤（min_length / max_length）
  2. Gopher 启发式质量过滤：词数 / 平均词长 / 省略号行占比 / 字母词占比
  3. SHA256 精确去重（dedup=True 时启用；BUG-8 修复：统一使用 SHA256）

语言识别 / NSFW / 质量分类器过滤需要 fastText 模型文件，留待后续 Plan 扩展。

参考来源：
  - reference_resource/Assignment4-data/cs336_data/filters.py（gopher_quality_filter 等）
  - reference_resource/Assignment4-data/cs336_data/run_pipeline.py（流水线结构）
"""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace  # BUG-C2 修复：支持 SimpleNamespace 类型
from typing import Any


class DataPipeline:
    """数据清洗流水线。

    用法：
        cfg = {
            "data": {
                "input_path": "data/sample.jsonl",
                "filters": {"min_length": 100, "max_length": 50000},
                "dedup": True,
            },
            "output": {"base_dir": "results/"},
        }
        pipeline = DataPipeline(cfg)
        stats = pipeline.run(cfg["data"]["input_path"], cfg["output"]["base_dir"])
    """

    def __init__(self, cfg: SimpleNamespace | dict[str, Any]) -> None:
        """
        Args:
            cfg: 配置对象，可以是 SimpleNamespace（来自 load_config）或 dict

        BUG-C2 修复：接受 SimpleNamespace 或 dict，内部统一处理
        """
        self.cfg = cfg
        # 从 SimpleNamespace 或 dict 安全获取 data 配置
        if isinstance(cfg, SimpleNamespace):
            data_cfg = getattr(cfg, 'data', SimpleNamespace())
            filters_cfg = getattr(data_cfg, 'filters', SimpleNamespace())
            self.min_length: int = getattr(filters_cfg, 'min_length', 100)
            self.max_length: int = getattr(filters_cfg, 'max_length', 50000)
            self.do_dedup: bool = getattr(data_cfg, 'dedup', True)
        else:
            data_cfg = cfg.get("data", {})
            filters_cfg = data_cfg.get("filters", {})
            self.min_length: int = filters_cfg.get("min_length", 100)
            self.max_length: int = filters_cfg.get("max_length", 50000)
            self.do_dedup: bool = data_cfg.get("dedup", True)

    def run(self, input_path: str, output_dir: str) -> dict[str, Any]:
        """执行完整清洗流水线。

        Args:
            input_path: 输入 JSONL 文件路径（每行 {"text": "..."}）
            output_dir: 输出目录路径（由调用方负责创建，直接使用）

        Returns:
            统计字典：
            {
                "input_docs": 输入文档数,
                "output_docs": 输出文档数,
                "output_dir": 实际输出目录路径,
                "filter_stats": {
                    "length_filtered": N,      # 字段名已与 Plan 6 verify_stage4 对齐
                    "quality_filtered": N,     # 原 gopher_filter_removed
                    "dedup_filtered": N,       # 原 dedup_removed
                }
            }
        """
        # BUG-H2 修复：直接使用传入的 output_dir，不再内部创建 hash 子目录
        os.makedirs(output_dir, exist_ok=True)

        # ── 读取输入文档 ──
        docs: list[dict[str, Any]] = []
        with open(input_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    docs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # 跳过无效行

        input_count = len(docs)
        # 字段名已与 Plan 6 verify_stage4 对齐（BUG-3 修复）
        filter_stats: dict[str, int] = {
            "length_filtered": 0,
            "quality_filtered": 0,
            "dedup_filtered": 0,
        }

        # ── 步骤 1：长度过滤 ──
        after_length: list[dict[str, Any]] = []
        for doc in docs:
            text = doc.get("text", "")
            if self.min_length <= len(text) <= self.max_length:
                after_length.append(doc)
            else:
                filter_stats["length_filtered"] += 1

        # ── 步骤 2：Gopher 启发式质量过滤 ──
        after_gopher: list[dict[str, Any]] = []
        for doc in after_length:
            text = doc.get("text", "")
            if _gopher_quality_filter(text):
                after_gopher.append(doc)
            else:
                filter_stats["quality_filtered"] += 1

        # ── 步骤 3：SHA256 精确去重（可选；BUG-8 修复：统一 hash 策略）──
        if self.do_dedup:
            after_dedup: list[dict[str, Any]] = []
            seen: set[bytes] = set()
            for doc in after_gopher:
                text = doc.get("text", "")
                digest = hashlib.sha256(text.encode("utf-8")).digest()
                if digest not in seen:
                    seen.add(digest)
                    after_dedup.append(doc)
                else:
                    filter_stats["dedup_filtered"] += 1
            result_docs = after_dedup
        else:
            result_docs = after_gopher

        # ── 写出 cleaned.jsonl ──
        cleaned_path = os.path.join(output_dir, "cleaned.jsonl")
        with open(cleaned_path, "w", encoding="utf-8") as f:
            for doc in result_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")

        # ── 追加写入 metrics.jsonl ──
        metrics_path = os.path.join(output_dir, "metrics.jsonl")
        metrics = {
            "input_docs": input_count,
            "output_docs": len(result_docs),
            "filter_stats": filter_stats,
        }
        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        return {
            "input_docs": input_count,
            "output_docs": len(result_docs),
            "output_dir": output_dir,
            "filter_stats": filter_stats,
        }


# ─────────────────────────────────────────
# 私有辅助函数
# ─────────────────────────────────────────

def _config_hash(cfg: dict[str, Any]) -> str:
    """返回配置字典的 SHA256 十六进制摘要。
    BUG-8 修复：统一使用 SHA256（与 common/config.py load_config 一致，替代原 MD5）。
    """
    serialized = json.dumps(cfg, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _gopher_quality_filter(text: str) -> bool:
    """基于 DeepMind Gopher 论文的启发式文本质量过滤器。

    规则与 reference_resource/Assignment4-data/cs336_data/filters.py
    的 gopher_quality_filter 完全一致，此处内联以避免 import 依赖。

    返回 True 表示文本通过质量检查（保留）。
    """
    words = text.split()
    num_words = len(words)

    # 规则 1：词数范围 [50, 100000]
    if num_words < 50 or num_words > 100000:
        return False

    # 规则 2：平均词长范围 [3, 10]
    total_chars = sum(len(w) for w in words)
    mean_word_length = total_chars / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # 规则 3：省略号行占比 <= 30%
    lines = text.splitlines()
    if lines:
        ellipsis_lines = sum(1 for line in lines if line.strip().endswith("..."))
        if ellipsis_lines / len(lines) > 0.3:
            return False

    # 规则 4：字母词占比 >= 80%
    alpha_words = sum(1 for w in words if any(c.isalpha() for c in w))
    if (alpha_words / num_words) < 0.8:
        return False

    return True
```

- [ ] **Step 4: 更新 `llm_foundry/stage4_data/__init__.py`**

```python
from llm_foundry.stage4_data.pipeline import DataPipeline

__all__ = ["DataPipeline"]
```

- [ ] **Step 5: 运行 Stage 4 测试，确认通过**

```bash
pytest tests/test_stages.py -k "data_pipeline" -v
```

Expected: 4 tests PASSED（`test_data_pipeline_import`, `test_data_pipeline_run_produces_cleaned_jsonl`, `test_data_pipeline_filters_short_docs`, `test_data_pipeline_dedup_removes_exact_duplicates`）

- [ ] **Step 6: Commit**

```bash
git add llm_foundry/stage4_data/pipeline.py llm_foundry/stage4_data/__init__.py tests/test_stages.py
git commit -m "feat: add DataPipeline with length/gopher/dedup filters"
```

---

## Task 4: 填充真实配置文件

**Files:**
- Modify: `configs/tokenize.yaml`
- Modify: `configs/data.yaml`

- [ ] **Step 1: 覆写 `configs/tokenize.yaml`**

```yaml
# configs/tokenize.yaml
# 用法：python run.py tokenize --config configs/tokenize.yaml
# 产出：results/{hash}/tokenizer.json

tokenizer:
  text_file: data/sample.txt          # 训练语料文件路径（UTF-8 纯文本）
  vocab_size: 1000                     # 目标词表大小（256 个基础字节 + 744 个 BPE 合并 token）

output:
  base_dir: results/                   # 输出根目录，实际写入 results/{hash}/tokenizer.json
```

- [ ] **Step 2: 覆写 `configs/data.yaml`**

```yaml
# configs/data.yaml
# 用法：python run.py data --config configs/data.yaml
# 产出：results/{hash}/cleaned.jsonl + results/{hash}/metrics.jsonl

data:
  input_path: data/sample.jsonl        # 输入 JSONL 文件路径（每行 {"text": "..."}）
  filters:
    min_length: 100                    # 文档最小字符数（小于此值则丢弃）
    max_length: 50000                  # 文档最大字符数（大于此值则丢弃）
  dedup: true                          # 是否启用 SHA256 精确去重（BUG-8 修复：统一使用 SHA256）

output:
  base_dir: results/                   # 输出根目录，实际写入 results/{hash}/
```

- [ ] **Step 3: Commit**

```bash
git add configs/tokenize.yaml configs/data.yaml
git commit -m "feat: fill real config values for tokenize and data pipeline"
```

---

## Task 5: 接入 `run.py` 子命令

**Files:**
- Modify: `run.py`

**背景：** `run.py` 目前只含 `env` 子命令，尚无 `cmd_tokenize` 和 `cmd_data`。需要新增这两个子命令，解析 `--config` YAML 文件并调用对应类。

- [ ] **Step 1: 在 `tests/test_stages.py` 末尾追加 CLI 集成测试**

```python

# ─────────────────────────────────────────
# CLI 集成测试（run.py tokenize / data）
# ─────────────────────────────────────────
import pathlib
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent

def test_cli_tokenize_runs(tmp_path):
    """python run.py tokenize --config <yaml> 应正常完成并产出 tokenizer.json。"""
    import subprocess, sys, textwrap

    # 创建临时文本文件和配置文件
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        ("the quick brown fox jumps over the lazy dog. " * 40),
        encoding="utf-8",
    )
    cfg_file = tmp_path / "tokenize.yaml"
    cfg_file.write_text(
        textwrap.dedent(f"""\
            tokenizer:
              text_file: {corpus}
              vocab_size: 300
            output:
              base_dir: {tmp_path / "results"}
        """),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "run.py", "tokenize", "--config", str(cfg_file)],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"tokenize 命令失败:\n{result.stderr}"

    # 验证产出 tokenizer.json
    out_dirs = list((tmp_path / "results").iterdir())
    assert len(out_dirs) >= 1, "results/ 下应至少有一个 hash 子目录"
    tok_json = out_dirs[0] / "tokenizer.json"
    assert tok_json.exists(), f"tokenizer.json 不存在: {tok_json}"


def test_cli_data_runs(tmp_path):
    """python run.py data --config <yaml> 应正常完成并产出 cleaned.jsonl。"""
    import subprocess, sys, textwrap

    # 创建临时 JSONL 文件
    long_text = "word " * 60
    input_jsonl = tmp_path / "input.jsonl"
    lines = [json.dumps({"text": long_text + f" doc {i}."}) for i in range(5)]
    input_jsonl.write_text("\n".join(lines), encoding="utf-8")

    cfg_file = tmp_path / "data.yaml"
    cfg_file.write_text(
        textwrap.dedent(f"""\
            data:
              input_path: {input_jsonl}
              filters:
                min_length: 10
                max_length: 50000
              dedup: false
            output:
              base_dir: {tmp_path / "results"}
        """),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "run.py", "data", "--config", str(cfg_file)],
        capture_output=True, text=True,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0, f"data 命令失败:\n{result.stderr}"

    out_dirs = list((tmp_path / "results").iterdir())
    assert len(out_dirs) >= 1
    cleaned = out_dirs[0] / "cleaned.jsonl"
    assert cleaned.exists(), f"cleaned.jsonl 不存在: {cleaned}"
```

- [ ] **Step 2: 运行 CLI 测试，确认失败**

```bash
pytest tests/test_stages.py -k "cli" -v
```

Expected: FAILED（`tokenize` 和 `data` 子命令尚不存在，命令行报错 "invalid choice"）

- [ ] **Step 3: 修改 `run.py` — 新增 `cmd_tokenize`**

在 `run.py` 中新增 `cmd_tokenize` 函数（Plan 1 中不存在此函数）：

```python
def cmd_tokenize(args) -> None:
    """tokenize 子命令：从 YAML 配置读取参数，训练 BPE 并保存 tokenizer.json。"""
    from llm_foundry.stage1_tokenize import BPETokenizer
    from llm_foundry.common.config import load_config, config_hash  # BUG-C3 修复：导入 config_hash

    # 若通过子命令（train/encode）调用，保留占位符（Plan 2 仅实现 --config 路径）
    if hasattr(args, "tokenize_cmd") and args.tokenize_cmd is not None:
        print(f"[INFO] tokenize {args.tokenize_cmd} 子命令尚未实现（Plan 2 使用 --config）")
        sys.exit(0)

    if not hasattr(args, "config") or not args.config:
        print("[ERROR] --config 参数必须指定", file=sys.stderr)
        sys.exit(1)

    # BUG-H3 修复：使用 load_config 获取 cfg 和 cfg_hash，避免双重计算
    cfg, cfg_hash = load_config(args.config)  # 内部使用 SHA256，已创建 run_dir

    # 支持 --set key=value 覆盖（可选扩展，Plan 2 暂不实现）
    tok_cfg = cfg.get("tokenizer", {})
    text_file = tok_cfg.get("text_file")
    vocab_size = int(tok_cfg.get("vocab_size", 1000))

    if not text_file:
        print("[ERROR] configs/tokenize.yaml 中缺少 tokenizer.text_file", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 训练 BPETokenizer: text_file={text_file}, vocab_size={vocab_size}")
    tok = BPETokenizer()
    tok.train(text_file, vocab_size)

    # BUG-H3 修复：直接使用 cfg.run_dir，load_config 已创建该目录
    out_dir = str(cfg.run_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "tokenizer.json")
    tok.save(out_path)
    print(f"[INFO] tokenizer.json 已保存至: {out_path}
    # C-02 修复：同时生成 train.npy 供训练阶段使用
    npy_path = os.path.join(out_dir, "train.npy")
    tok.tokenize_file(text_file, npy_path)
    print(f"[INFO] train.npy 已保存至: {npy_path}")
```

- [ ] **Step 4: 修改 `run.py` — 新增 `cmd_data`**

在 `run.py` 中新增 `cmd_data` 函数（Plan 1 中不存在此函数）：

```python
def cmd_data(args) -> None:
    """data 子命令：从 YAML 配置读取参数，执行数据清洗流水线。"""
    from llm_foundry.stage4_data import DataPipeline
    from llm_foundry.common.config import load_config, config_hash  # BUG-C3 修复：导入 config_hash

    if not hasattr(args, "config") or not args.config:
        print("[ERROR] --config 参数必须指定", file=sys.stderr)
        sys.exit(1)

    # BUG-C2 修复：使用 load_config 获取 cfg 和 run_dir，统一 hash 计算
    cfg, cfg_hash = load_config(args.config)
    run_dir = str(cfg.run_dir)  # load_config 已创建 results/{hash}/ 目录

    # 从 cfg (SimpleNamespace) 读取配置值
    input_path = cfg.data.input_path if hasattr(cfg, 'data') else None

    if not input_path:
        print("[ERROR] configs/data.yaml 中缺少 data.input_path", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 运行 DataPipeline: input_path={input_path}")
    # BUG-H2 修复：直接传入 cfg（SimpleNamespace），DataPipeline 已兼容
    # BUG-C2 修复：直接传递 cfg，DataPipeline.__init__ 已支持 SimpleNamespace
    pipeline = DataPipeline(cfg)
    stats = pipeline.run(input_path, run_dir)

    print(f"[INFO] 清洗完成：输入 {stats['input_docs']} 条 -> 输出 {stats['output_docs']} 条")
    print(f"[INFO] 过滤统计：{stats['filter_stats']}")
    print(f"[INFO] 输出目录：{stats['output_dir']}")
```

- [ ] **Step 5: 同时更新 `run.py` 中 `build_parser` 的 `data_p` 和 `tok_p` 子命令，确保 `--config` 参数存在**

找到 `data_p = subparsers.add_parser("data", ...)` 段落，确认其中包含：
```python
data_p.add_argument("--config", default="configs/data.yaml")
```

找到 `tok_p = subparsers.add_parser("tokenize", ...)` 段落，在其（或 `train_tok`）的参数定义处，确认顶层 `tok_p` 有：
```python
tok_p.add_argument("--config", default="configs/tokenize.yaml")
```

如果 Plan 1 的 `run.py` 中 tokenize 只有子命令（`tokenize train` / `tokenize encode`）没有 `tok_p.add_argument("--config", ...)`，则需要新增此行。`cmd_tokenize` 函数会在 `args.tokenize_cmd` 为 `None`（即直接 `python run.py tokenize --config ...`）时使用 `--config` 路径。

具体地：将 `tok_p = subparsers.add_parser(...)` 行之后立即添加：
```python
tok_p.add_argument("--config", default="configs/tokenize.yaml",
                   help="tokenize 配置文件（当不使用子命令 train/encode 时）")
```

同时在新增的 `cmd_tokenize` 开头加入：对 `args.tokenize_cmd` 做判断——若为 `None` 则走 `--config` YAML 路径，若为 `"train"` / `"encode"` 则打印说明信息（Plan 2 仅实现 `--config` 路径，`train`/`encode` 子命令暂不实现）：

```python
def cmd_tokenize(args) -> None:
    """tokenize 子命令：从 YAML 配置读取参数，训练 BPE 并保存 tokenizer.json。"""
    from llm_foundry.stage1_tokenize import BPETokenizer
    from llm_foundry.common.config import load_config, config_hash  # BUG-C3 修复：导入 config_hash

    # 若通过子命令（train/encode）调用，保留占位符（Plan 2 仅实现 --config 路径）
    if hasattr(args, "tokenize_cmd") and args.tokenize_cmd is not None:
        print(f"[INFO] tokenize {args.tokenize_cmd} 子命令尚未实现（Plan 2 使用 --config）")
        sys.exit(0)

    if not hasattr(args, "config") or not args.config:
        print("[ERROR] --config 参数必须指定", file=sys.stderr)
        sys.exit(1)

    # BUG-H3 修复：使用 load_config 获取 cfg 和 cfg_hash，避免双重计算和目录创建
    cfg, cfg_hash = load_config(args.config)  # 内部使用 SHA256，已创建 run_dir

    tok_cfg = cfg.get("tokenizer", {})
    text_file = tok_cfg.get("text_file")
    vocab_size = int(tok_cfg.get("vocab_size", 1000))

    if not text_file:
        print("[ERROR] configs/tokenize.yaml 中缺少 tokenizer.text_file", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 训练 BPETokenizer: text_file={text_file}, vocab_size={vocab_size}")
    tok = BPETokenizer()
    tok.train(text_file, vocab_size)

    # BUG-H3 修复：直接使用 cfg.run_dir，load_config 已创建该目录
    out_dir = str(cfg.run_dir)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, "tokenizer.json")
    tok.save(out_path)
    print(f"[INFO] tokenizer.json 已保存至: {out_path}")
    # C-02 修复：同时生成 train.npy 供训练阶段使用
    npy_path = os.path.join(out_dir, "train.npy")
    tok.tokenize_file(text_file, npy_path)
    print(f"[INFO] train.npy 已保存至: {npy_path}")
```

- [ ] **Step 6: 运行 CLI 测试，确认通过**

```bash
pytest tests/test_stages.py -k "cli" -v
```

Expected: 2 tests PASSED（`test_cli_tokenize_runs`, `test_cli_data_runs`）

- [ ] **Step 7: 手动端到端验证**

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
python run.py tokenize --config configs/tokenize.yaml
```

Expected 输出（类似）：
```
[INFO] 训练 BPETokenizer: text_file=data/sample.txt, vocab_size=1000
[INFO] tokenizer.json 已保存至: results/xxxxxxxx/tokenizer.json
```

```bash
python run.py data --config configs/data.yaml
```

Expected 输出（类似）：
```
[INFO] 运行 DataPipeline: input_path=data/sample.jsonl
[INFO] 清洗完成：输入 10 条 -> 输出 N 条
[INFO] 过滤统计：{'length_filtered': M, 'quality_filtered': K, 'dedup_filtered': J}
[INFO] 输出目录：results/xxxxxxxx
```

- [ ] **Step 8: Commit**

```bash
git add run.py
git commit -m "feat: wire up run.py tokenize and data subcommands with --config YAML"
```

---

## Task 6: 运行全量测试并验收

**Files:**
- 无新增文件

- [ ] **Step 1: 运行全部测试**

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
pytest tests/ -v
```

Expected: 全部 PASSED，无 FAILED 或 ERROR

- [ ] **Step 2: 完整端到端验证**

```bash
# 确认 env 命令仍正常
python run.py env

# 验证 tokenize 产出
python run.py tokenize --config configs/tokenize.yaml
ls results/*/tokenizer.json

# 验证 data 产出
python run.py data --config configs/data.yaml
ls results/*/cleaned.jsonl
ls results/*/metrics.jsonl
```

- [ ] **Step 3: 最终 Commit**

```bash
git add -u
git commit -m "feat: plan2 complete - BPETokenizer, DataPipeline, tokenize/data CLI"
```

---

## 验收标准

Plan 2 完成后，以下条件应全部满足：

1. `pytest tests/ -v` 全部通过（包含 Plan 1 的所有测试 + Plan 2 新增测试）
2. `python run.py tokenize --config configs/tokenize.yaml` 正常完成，产出 `results/{hash}/tokenizer.json`
3. `python run.py data --config configs/data.yaml` 正常完成，产出 `results/{hash}/cleaned.jsonl` 和 `results/{hash}/metrics.jsonl`
4. `tokenizer.json` 内容为合法 JSON，包含 `vocab`（字典，key 为字符串整数，value 为 base64）和 `merges`（列表）
5. `cleaned.jsonl` 内容为合法 JSONL，每行均可 `json.loads` 且包含 `text` 字段
6. `BPETokenizer.from_pretrained(path).encode(text)` 与训练后直接调用 `encode(text)` 结果一致
7. `DataPipeline` 在 `dedup=True` 时去重，3 条完全相同文档只保留 1 条
8. 所有测试在 Win11 CPU 上运行完毕耗时 < 30 秒

---

## 参考资源路径

| 组件 | 参考文件 | 使用方式 |
|------|---------|---------|
| BPE 训练算法 | `reference_resource/Assignment1-basics/Transformer/bpe.py` | 将 `train_bpe` 函数复制到 `stage1_tokenize/bpe.py`，`_train_bpe` 本地 import（DESIGN-1 修复：删除 sys.path.insert） |
| BPE 编码逻辑 | `reference_resource/Assignment1-basics/Transformer/tokenizer.py` | 内联复制 `_encode_chunk` 逻辑为模块级私有函数 |
| Gopher 质量过滤 | `reference_resource/Assignment4-data/cs336_data/filters.py` | 内联复制 `gopher_quality_filter` 逻辑（约 30 行）|
| 流水线结构参考 | `reference_resource/Assignment4-data/cs336_data/run_pipeline.py` | 参考三阶段串联结构（过滤 -> 精确去重 -> 近似去重）|
