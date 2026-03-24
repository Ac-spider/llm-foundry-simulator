# Plan 0: Stage0 DataGen (DeepSeek API 合成数据) Implementation Plan

> **For agentic workers (OMC):** 推荐使用 `/oh-my-claudecode:ralph`（自循环执行直到完成，适合多步骤 task）或 `/oh-my-claudecode:ultrawork`（并行高吞吐执行，适合独立任务批量完成，复杂 task 加 `model=opus`）。步骤使用 checkbox (`- [ ]`) 语法跟踪进度，完成后用 TaskUpdate 标记 completed。

**Goal:** 新增 `llm_foundry/stage0_datagen/` 模块，通过交大 DeepSeek-V3 API 批量合成 SFT 指令数据（500 条）和 GRPO 数学推理数据（500 条），产出两个 JSONL 文件，供 Stage 5 对齐训练使用。

**Architecture:** `client.py` 封装异步 HTTP 请求（aiohttp + asyncio），使用全局令牌桶严格限速 ≤80 req/min，Session 在 client 生命周期内复用；`sft_gen.py` 和 `grpo_gen.py` 各持有一个 client 实例，定义各自的 prompt 模板和输出解析逻辑；generate() 使用 while 循环保证恰好写入 n 条；`datagen.py` 作为统一入口协调两者；`run.py` 新增 `datagen` 子命令。API key 从环境变量 `SJTU_API_KEY` 读取，绝不硬编码。

**Tech Stack:** Python 3.10+, aiohttp, asyncio, tqdm, pyyaml, pytest, pytest-asyncio

---

## 文件映射

**新建文件：**

```
llm_foundry/stage0_datagen/__init__.py      — Task 5 完成后填充导出；Task 1 先建空文件
llm_foundry/stage0_datagen/client.py        — DeepSeekClient：异步HTTP封装，令牌桶限速+重试，Session复用
llm_foundry/stage0_datagen/sft_gen.py       — SFTGenerator：生成 instruction-response 对
llm_foundry/stage0_datagen/grpo_gen.py      — GRPOGenerator：生成数学推理题
llm_foundry/stage0_datagen/datagen.py       — DataGenConfig + run_datagen() 统一入口
configs/datagen.yaml                         — 数据生成配置（数量、模型、输出路径等）
tests/test_datagen.py                        — 单元测试（mock HTTP，不发真实请求）
```

**修改文件：**

```
run.py                                       — 新增 datagen 子命令
pyproject.toml                               — 新增 aiohttp, tqdm, pytest-asyncio 依赖
```

---

## 前置条件

在开始任何 Task 之前，确保以下条件均已满足：

```bash
# 1. 安装项目（确保 llm_foundry 包可被 import）
pip install -e .

# 2. 设置 DeepSeek API key（必须，否则所有 API 调用均失败）
API Base URL: https://models.sjtu.edu.cn/api/v1/chat/completions
API Key: sk-rymX7XZk1zK4rrhnraA_sQ
模型名: deepseek-v3
```

---

## Task 1: 安装依赖 + 创建模块骨架

**Files:**
- Modify: `pyproject.toml`
- Create: `llm_foundry/stage0_datagen/__init__.py`

- [ ] **Step 1: 在 pyproject.toml 的 dependencies 里加入新依赖，同时加入 pytest 配置**

```toml
# 在 [project] dependencies 列表中添加：
"aiohttp>=3.9",
"tqdm>=4.66",
"pyyaml>=6.0",

# 在 [project.optional-dependencies] 或 dev 依赖中添加：
# "pytest-asyncio>=0.23",

# 新增（或追加到已有的 [tool.pytest.ini_options]）：
[tool.pytest.ini_options]
asyncio_mode = "auto"
```

- [ ] **Step 2: 创建 `llm_foundry/stage0_datagen/__init__.py`（空文件，Task 5 完成后填充）**

```python
# llm_foundry/stage0_datagen/__init__.py
# 导出在 Task 5 完成后填充
```

- [ ] **Step 3: 安装依赖**

```bash
pip install aiohttp tqdm pytest-asyncio
```

Expected: 安装成功，无报错

- [ ] **Step 4: 验证 import 不报错**

```bash
python -c "import aiohttp; import tqdm; print('OK')"
```

Expected: 输出 `OK`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml llm_foundry/stage0_datagen/__init__.py
git commit -m "feat(datagen): add stage0_datagen module skeleton and dependencies"
```

---

## Task 2: 实现 `client.py` — DeepSeek API 异步客户端

**Files:**
- Create: `llm_foundry/stage0_datagen/client.py`
- Test: `tests/test_datagen.py`（本 Task 写 client 相关测试）

### 设计说明

- API key 从环境变量 `SJTU_API_KEY` 读取
- **限速（令牌桶）**：全局 `asyncio.Queue` 令牌桶，每 0.75s 投放一个令牌（≤80 req/min），所有请求消费令牌后才发送
- **Session 复用**：`aiohttp.ClientSession` 在 `DeepSeekClient` 整个生命周期内只创建一次，用完后显式 `close()`
- 重试：最多 3 次，指数退避（1s, 2s, 4s），HTTP 429/5xx 触发重试

- [ ] **Step 1: 写 client 的失败测试**

```python
# tests/test_datagen.py
import os
import pathlib
import subprocess
import sys
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

PROJECT_ROOT = pathlib.Path(__file__).parent.parent

def test_client_reads_api_key_from_env(monkeypatch):
    """client 从环境变量读取 API key"""
    monkeypatch.setenv("SJTU_API_KEY", "test-key-123")
    from llm_foundry.stage0_datagen.client import DeepSeekClient
    client = DeepSeekClient()
    assert client.api_key == "test-key-123"

def test_client_raises_if_no_api_key(monkeypatch):
    """没有设置环境变量时抛出 ValueError"""
    monkeypatch.delenv("SJTU_API_KEY", raising=False)
    from llm_foundry.stage0_datagen.client import DeepSeekClient
    with pytest.raises(ValueError, match="SJTU_API_KEY"):
        DeepSeekClient()
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_datagen.py::test_client_reads_api_key_from_env tests/test_datagen.py::test_client_raises_if_no_api_key -v
```

Expected: FAIL with `ModuleNotFoundError` 或 `ImportError`

- [ ] **Step 3: 实现 `client.py`**

```python
# llm_foundry/stage0_datagen/client.py
"""
DeepSeek API 异步客户端。
API key 从环境变量 SJTU_API_KEY 读取，绝不硬编码。
限速：全局令牌桶，≤80 req/min（每 0.75s 投放一个令牌）。
Session：整个 client 生命周期内复用同一个 aiohttp.ClientSession。
"""
from __future__ import annotations

import asyncio
import os
from typing import Any

import aiohttp

BASE_URL = "https://models.sjtu.edu.cn/api/v1/chat/completions"
MODEL = "deepseek-v3"
MAX_RETRIES = 3
# 令牌桶：每 0.75s 投放一个令牌，对应 ≤80 req/min
TOKEN_INTERVAL = 0.75


class DeepSeekClient:
    """
    异步 DeepSeek API 客户端，内置令牌桶限速与重试。
    使用方式：
        client = DeepSeekClient()
        try:
            result = await client.chat(messages)
        finally:
            await client.close()
    """

    def __init__(self, model: str = MODEL) -> None:
        api_key = os.environ.get("SJTU_API_KEY")
        if not api_key:
            raise ValueError(
                "环境变量 SJTU_API_KEY 未设置。"
                "请先执行: export SJTU_API_KEY=your_key"
            )
        self.api_key = api_key
        self.model = model
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Session 在整个生命周期内复用
        self._session: aiohttp.ClientSession | None = None
        # 令牌桶：容量为 1，由后台 _refill_task 持续投放
        self._token_bucket: asyncio.Queue[None] = asyncio.Queue(maxsize=1)
        self._refill_task: asyncio.Task | None = None

    async def _ensure_started(self) -> None:
        """懒初始化：确保 session 和令牌桶已启动（首次 chat 调用时触发）。"""
        if self._session is None:
            self._session = aiohttp.ClientSession()
        if self._refill_task is None:
            self._refill_task = asyncio.create_task(self._refill_tokens())

    async def _refill_tokens(self) -> None:
        """后台任务：每 TOKEN_INTERVAL 秒往令牌桶投放一个令牌。"""
        while True:
            try:
                self._token_bucket.put_nowait(None)
            except asyncio.QueueFull:
                pass  # 桶已满，丢弃（防止积压）
            await asyncio.sleep(TOKEN_INTERVAL)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.8,
        max_tokens: int = 1024,
    ) -> str:
        """
        发送一次 chat 请求，返回 assistant 回复内容字符串。
        调用前会从令牌桶取令牌（限速），内置重试（最多 3 次，指数退避）。
        超出重试次数后抛出 RuntimeError。
        """
        await self._ensure_started()
        # 从令牌桶取令牌（限速）
        await self._token_bucket.get()

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        for attempt in range(MAX_RETRIES):
            try:
                async with self._session.post(
                    BASE_URL,
                    json=payload,
                    headers=self._headers,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == MAX_RETRIES - 1:
                    raise RuntimeError(
                        f"DeepSeek API 请求失败（已重试 {MAX_RETRIES} 次）: {e}"
                    ) from e
                await asyncio.sleep(2 ** attempt)
        raise RuntimeError("DeepSeek API 请求失败（超出重试次数）")

    async def close(self) -> None:
        """关闭 session 和令牌桶任务，释放资源。"""
        if self._refill_task is not None:
            self._refill_task.cancel()
            self._refill_task = None
        if self._session is not None:
            await self._session.close()
            self._session = None
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_datagen.py::test_client_reads_api_key_from_env tests/test_datagen.py::test_client_raises_if_no_api_key -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add llm_foundry/stage0_datagen/client.py tests/test_datagen.py
git commit -m "feat(datagen): implement DeepSeekClient with token-bucket rate limiting and session reuse"
```

---

## Task 3: 实现 `sft_gen.py` — SFT 指令数据生成器

**Files:**
- Create: `llm_foundry/stage0_datagen/sft_gen.py`
- Test: `tests/test_datagen.py`（追加测试）

### 输出格式

每行一个 JSON 对象：
```json
{"prompt": "...", "response": "..."}
```

### Prompt 模板

```
请生成一条高质量的中文指令跟随样本，包含一个用户指令和一个详细的助手回复。
要求：
1. 指令要多样，涵盖写作、分析、问答、代码、翻译等不同任务类型
2. 回复要完整、有帮助、内容充实（不少于100字）
3. 严格按以下JSON格式输出，不要输出其他内容：
{"prompt": "用户指令内容", "response": "助手回复内容"}
```

- [ ] **Step 1: 写 SFTGenerator 的失败测试**

```python
# 追加到 tests/test_datagen.py

def test_sft_generator_parses_response(monkeypatch):
    """SFTGenerator 能正确解析 API 返回的 JSON（同步方法，无需 asyncio）"""
    monkeypatch.setenv("SJTU_API_KEY", "test-key")
    from llm_foundry.stage0_datagen.sft_gen import SFTGenerator
    gen = SFTGenerator()
    fake_response = '{"prompt": "解释牛顿第一定律", "response": "牛顿第一定律..."}'
    result = gen._parse(fake_response)
    assert result["prompt"] == "解释牛顿第一定律"
    assert "response" in result

def test_sft_generator_skips_invalid_json(monkeypatch):
    """解析失败时返回 None，不抛出异常"""
    monkeypatch.setenv("SJTU_API_KEY", "test-key")
    from llm_foundry.stage0_datagen.sft_gen import SFTGenerator
    gen = SFTGenerator()
    assert gen._parse("这不是JSON格式") is None
    assert gen._parse('{"prompt": "只有指令"}') is None  # 缺少 response 字段
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_datagen.py::test_sft_generator_parses_response tests/test_datagen.py::test_sft_generator_skips_invalid_json -v
```

Expected: FAIL

- [ ] **Step 3: 实现 `sft_gen.py`**

```python
# llm_foundry/stage0_datagen/sft_gen.py
"""
SFT 指令跟随数据生成器。
输出格式（每行）：{"prompt": "...", "response": "..."}
generate() 使用 while 循环，确保恰好写入 n 条（不因解析失败而少写）。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tqdm.asyncio import tqdm as atqdm

from .client import DeepSeekClient

SYSTEM_PROMPT = "你是一个专业的数据标注员，负责生成高质量的中文指令跟随训练数据。"

USER_PROMPT = """请生成一条高质量的中文指令跟随样本，包含一个用户指令和一个详细的助手回复。
要求：
1. 指令要多样，涵盖写作、分析、问答、代码、翻译等不同任务类型
2. 回复要完整、有帮助、内容充实（不少于100字）
3. 严格按以下JSON格式输出，不要输出其他内容：
{"prompt": "用户指令内容", "response": "助手回复内容"}"""


class SFTGenerator:
    """批量生成 SFT 指令-回复对。"""

    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self.client = client or DeepSeekClient()

    def _parse(self, text: str) -> dict[str, str] | None:
        """解析 API 返回的 JSON 字符串，失败返回 None。"""
        text = text.strip()
        # 兼容模型在 JSON 前后输出多余文字的情况
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            data = json.loads(text[start:end])
            if "prompt" in data and "response" in data:
                return {
                    "prompt": str(data["prompt"]),
                    "response": str(data["response"]),
                }
        except json.JSONDecodeError:
            pass
        return None

    async def _generate_one(self) -> dict[str, str] | None:
        """生成单条 SFT 样本，最多尝试 3 次解析，失败返回 None。"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
        for _ in range(3):
            try:
                text = await self.client.chat(messages, temperature=0.9, max_tokens=1024)
                result = self._parse(text)
                if result:
                    return result
            except RuntimeError:
                pass
        return None

    async def generate(self, n: int, output_path: str) -> int:
        """
        生成恰好 n 条 SFT 数据，写入 output_path（JSONL 格式）。
        支持断点续传：若文件已存在，跳过已有行数。
        使用 while 循环保证最终写入数量恰好为 n（解析失败时自动补充请求）。
        返回实际写入的新条数。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # 断点续传：统计已有行数
        existing = 0
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing = sum(1 for line in f if line.strip())

        remaining = n - existing
        if remaining <= 0:
            print(f"[SFT] {output_path} 已有 {existing} 条，无需继续生成。")
            return 0

        print(f"[SFT] 已有 {existing} 条，还需生成 {remaining} 条 → {output_path}")

        written = 0
        max_attempts = remaining * 5
        attempts = 0
        with open(path, "a", encoding="utf-8") as f, \
             atqdm(total=remaining, desc="SFT生成") as pbar:
            while written < remaining and attempts < max_attempts:
                # 每轮批量发 min(8, 剩余量) 个并发请求
                batch = min(8, remaining - written)
                results = await asyncio.gather(
                    *[self._generate_one() for _ in range(batch)],
                    return_exceptions=False,
                )
                for result in results:
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        written += 1
                        pbar.update(1)
                        if written >= remaining:
                            break
                attempts += batch

        if written < remaining:
            raise RuntimeError(
                f"[SFT] 已尝试 {max_attempts} 次，仅成功生成 {written}/{remaining} 条。"
                "请检查 API 返回格式是否符合预期。"
            )
        print(f"[SFT] 完成，新写入 {written} 条")
        return written
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_datagen.py::test_sft_generator_parses_response tests/test_datagen.py::test_sft_generator_skips_invalid_json -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add llm_foundry/stage0_datagen/sft_gen.py tests/test_datagen.py
git commit -m "feat(datagen): implement SFTGenerator with guaranteed n-count and resume support"
```

---

## Task 4: 实现 `grpo_gen.py` — 数学推理数据生成器

**Files:**
- Create: `llm_foundry/stage0_datagen/grpo_gen.py`
- Test: `tests/test_datagen.py`（追加测试）

### 输出格式

每行一个 JSON 对象：
```json
{"problem": "...", "solution": "...", "answer": "..."}
```

### Prompt 模板

```
请生成一道中学或大学数学题，并给出详细解题过程和最终答案。
要求：
1. 题目难度适中（中学竞赛到大学微积分水平）
2. 解题过程要详细，逐步推导（不少于50字）
3. 答案要精确（数字或数学表达式）
4. 严格按以下JSON格式输出，不要输出其他内容：
{"problem": "题目内容", "solution": "详细解题过程", "answer": "最终答案"}
```

- [ ] **Step 1: 写 GRPOGenerator 的失败测试**

```python
# 追加到 tests/test_datagen.py

def test_grpo_generator_parses_response(monkeypatch):
    """GRPOGenerator 能正确解析 API 返回的 JSON"""
    monkeypatch.setenv("SJTU_API_KEY", "test-key")
    from llm_foundry.stage0_datagen.grpo_gen import GRPOGenerator
    gen = GRPOGenerator()
    fake = '{"problem": "求1+1", "solution": "直接计算得2", "answer": "2"}'
    result = gen._parse(fake)
    assert result["problem"] == "求1+1"
    assert result["answer"] == "2"

def test_grpo_generator_skips_invalid(monkeypatch):
    """解析失败时返回 None"""
    monkeypatch.setenv("SJTU_API_KEY", "test-key")
    from llm_foundry.stage0_datagen.grpo_gen import GRPOGenerator
    gen = GRPOGenerator()
    assert gen._parse("invalid") is None
    assert gen._parse('{"problem": "只有题目"}') is None  # 缺少 solution 和 answer
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_datagen.py::test_grpo_generator_parses_response tests/test_datagen.py::test_grpo_generator_skips_invalid -v
```

Expected: FAIL

- [ ] **Step 3: 实现 `grpo_gen.py`**

```python
# llm_foundry/stage0_datagen/grpo_gen.py
"""
GRPO 数学推理数据生成器。
输出格式（每行）：{"problem": "...", "solution": "...", "answer": "..."}
generate() 使用 while 循环，确保恰好写入 n 条。
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

from tqdm.asyncio import tqdm as atqdm

from .client import DeepSeekClient

SYSTEM_PROMPT = "你是一位数学教师，擅长出题和讲解数学题目。"

USER_PROMPT = """请生成一道中学或大学数学题，并给出详细解题过程和最终答案。
要求：
1. 题目难度适中（中学竞赛到大学微积分水平）
2. 解题过程要详细，逐步推导（不少于50字）
3. 答案要精确（数字或数学表达式）
4. 严格按以下JSON格式输出，不要输出其他内容：
{"problem": "题目内容", "solution": "详细解题过程", "answer": "最终答案"}"""


class GRPOGenerator:
    """批量生成 GRPO 数学推理题数据。"""

    def __init__(self, client: DeepSeekClient | None = None) -> None:
        self.client = client or DeepSeekClient()

    def _parse(self, text: str) -> dict[str, str] | None:
        """解析 API 返回的 JSON，失败返回 None。"""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        try:
            data = json.loads(text[start:end])
            if all(k in data for k in ("problem", "solution", "answer")):
                return {
                    "problem": str(data["problem"]),
                    "solution": str(data["solution"]),
                    "answer": str(data["answer"]),
                }
        except json.JSONDecodeError:
            pass
        return None

    async def _generate_one(self) -> dict[str, str] | None:
        """生成单条数学题，最多尝试 3 次解析，失败返回 None。"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ]
        for _ in range(3):
            try:
                text = await self.client.chat(messages, temperature=0.7, max_tokens=1024)
                result = self._parse(text)
                if result:
                    return result
            except RuntimeError:
                pass
        return None

    async def generate(self, n: int, output_path: str) -> int:
        """
        生成恰好 n 条数学题数据，写入 output_path（JSONL 格式）。
        支持断点续传。使用 while 循环保证最终写入数量恰好为 n。
        返回实际写入的新条数。
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = 0
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                existing = sum(1 for line in f if line.strip())

        remaining = n - existing
        if remaining <= 0:
            print(f"[GRPO] {output_path} 已有 {existing} 条，无需继续生成。")
            return 0

        print(f"[GRPO] 已有 {existing} 条，还需生成 {remaining} 条 → {output_path}")

        written = 0
        max_attempts = remaining * 5
        attempts = 0
        with open(path, "a", encoding="utf-8") as f, \
             atqdm(total=remaining, desc="GRPO生成") as pbar:
            while written < remaining and attempts < max_attempts:
                batch = min(8, remaining - written)
                results = await asyncio.gather(
                    *[self._generate_one() for _ in range(batch)],
                    return_exceptions=False,
                )
                for result in results:
                    if result is not None:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")
                        f.flush()
                        written += 1
                        pbar.update(1)
                        if written >= remaining:
                            break
                attempts += batch

        if written < remaining:
            raise RuntimeError(
                f"[GRPO] 已尝试 {max_attempts} 次，仅成功生成 {written}/{remaining} 条。"
                "请检查 API 返回格式是否符合预期。"
            )
        print(f"[GRPO] 完成，新写入 {written} 条")
        return written
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_datagen.py::test_grpo_generator_parses_response tests/test_datagen.py::test_grpo_generator_skips_invalid -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add llm_foundry/stage0_datagen/grpo_gen.py tests/test_datagen.py
git commit -m "feat(datagen): implement GRPOGenerator for math reasoning data"
```

---

## Task 5: 实现 `datagen.py` + `configs/datagen.yaml` + 填充 `__init__.py`

**Files:**
- Create: `llm_foundry/stage0_datagen/datagen.py`
- Create: `configs/datagen.yaml`
- Modify: `llm_foundry/stage0_datagen/__init__.py`
- Test: `tests/test_datagen.py`（追加测试）

- [ ] **Step 1: 写 DataGenConfig 的失败测试**

```python
# 追加到 tests/test_datagen.py

def test_datagenconfig_loads_from_yaml(tmp_path, monkeypatch):
    """DataGenConfig 能从 YAML 文件正确加载"""
    monkeypatch.setenv("SJTU_API_KEY", "test-key")
    config_content = """
datagen:
  sft_n: 100
  grpo_n: 50
  sft_output: results/test/sft_data.jsonl
  grpo_output: results/test/grpo_data.jsonl
"""
    config_file = tmp_path / "datagen.yaml"
    config_file.write_text(config_content)

    from llm_foundry.stage0_datagen.datagen import DataGenConfig
    cfg = DataGenConfig.from_yaml(str(config_file))
    assert cfg.sft_n == 100
    assert cfg.grpo_n == 50
    assert "sft_data.jsonl" in cfg.sft_output
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_datagen.py::test_datagenconfig_loads_from_yaml -v
```

Expected: FAIL

- [ ] **Step 3: 实现 `datagen.py`**

```python
# llm_foundry/stage0_datagen/datagen.py
"""
数据生成统一入口。
DataGenConfig: 从 YAML 加载配置。
run_datagen(): 串行运行 SFT → GRPO 生成器（共用同一个 client，令牌桶统一限速）。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import yaml

from .client import DeepSeekClient
from .sft_gen import SFTGenerator
from .grpo_gen import GRPOGenerator


@dataclass
class DataGenConfig:
    sft_n: int
    grpo_n: int
    sft_output: str
    grpo_output: str

    @classmethod
    def from_yaml(cls, path: str) -> "DataGenConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        cfg = raw["datagen"]
        return cls(
            sft_n=int(cfg["sft_n"]),
            grpo_n=int(cfg["grpo_n"]),
            sft_output=str(cfg["sft_output"]),
            grpo_output=str(cfg["grpo_output"]),
        )


async def _run_datagen_async(cfg: DataGenConfig) -> dict[str, int]:
    """
    串行运行 SFT → GRPO 生成（共用同一个 client 实例，令牌桶统一限速）。
    串行避免两个生成器同时并发导致总速率翻倍超限。
    """
    client = DeepSeekClient()
    try:
        sft_gen = SFTGenerator(client=client)
        sft_written = await sft_gen.generate(cfg.sft_n, cfg.sft_output)

        grpo_gen = GRPOGenerator(client=client)
        grpo_written = await grpo_gen.generate(cfg.grpo_n, cfg.grpo_output)
    finally:
        await client.close()

    return {"sft": sft_written, "grpo": grpo_written}


def run_datagen(cfg: DataGenConfig) -> dict[str, int]:
    """同步入口，供 run.py CLI 调用。"""
    return asyncio.run(_run_datagen_async(cfg))
```

- [ ] **Step 4: 创建 `configs/datagen.yaml`（各 500 条）**

```yaml
# configs/datagen.yaml
# DeepSeek API 数据生成配置
# 使用前请设置环境变量: export SJTU_API_KEY=your_key

datagen:
  sft_n: 500              # SFT 指令数据条数
  grpo_n: 500             # GRPO 数学推理数据条数
  sft_output: results/datagen/sft_data.jsonl
  grpo_output: results/datagen/grpo_data.jsonl
```

- [ ] **Step 5: 填充 `llm_foundry/stage0_datagen/__init__.py`**

```python
# llm_foundry/stage0_datagen/__init__.py
from .datagen import DataGenConfig, run_datagen
from .sft_gen import SFTGenerator
from .grpo_gen import GRPOGenerator

__all__ = ["DataGenConfig", "run_datagen", "SFTGenerator", "GRPOGenerator"]
```

- [ ] **Step 6: 运行测试，确认通过**

```bash
pytest tests/test_datagen.py::test_datagenconfig_loads_from_yaml -v
```

Expected: 1 passed

- [ ] **Step 7: Commit**

```bash
git add llm_foundry/stage0_datagen/datagen.py llm_foundry/stage0_datagen/__init__.py configs/datagen.yaml tests/test_datagen.py
git commit -m "feat(datagen): implement DataGenConfig, run_datagen, and finalize __init__.py"
```

---

## Task 6: 为 `run.py` 新增 `datagen` 子命令

**Files:**
- Modify: `run.py`（若已存在）或 Create（若 Plan 1 未完成）
- Test: `tests/test_datagen.py`（追加 CLI 测试）

- [ ] **Step 1: 写 CLI 的失败测试（使用绝对路径，跨平台）**

```python
# 追加到 tests/test_datagen.py

def test_datagen_cli_missing_api_key():
    """没有设置 SJTU_API_KEY 时，CLI 应以非零退出码退出"""
    env = os.environ.copy()
    env.pop("SJTU_API_KEY", None)
    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "run.py"),
            "datagen",
            "--config",
            str(PROJECT_ROOT / "configs" / "datagen.yaml"),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode != 0
    assert "SJTU_API_KEY" in result.stderr or "SJTU_API_KEY" in result.stdout
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_datagen.py::test_datagen_cli_missing_api_key -v
```

Expected: FAIL（run.py 没有 datagen 子命令）

- [ ] **Step 3a: 若 `run.py` 已存在，追加 datagen 子命令**

在已有 subparsers 定义区域追加：

```python
# 在 subparsers 区域追加：
parser_datagen = subparsers.add_parser(
    "datagen",
    help="用 DeepSeek API 生成 SFT/GRPO 训练数据",
)
parser_datagen.add_argument(
    "--config",
    default="configs/datagen.yaml",
    help="数据生成配置文件路径（默认: configs/datagen.yaml）",
)

# 在命令分发处追加：
elif args.command == "datagen":
    from llm_foundry.stage0_datagen import DataGenConfig, run_datagen
    try:
        cfg = DataGenConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"[错误] 配置文件不存在: {args.config}", file=sys.stderr)
        sys.exit(1)
    try:
        result = run_datagen(cfg)
        print(f"\n[datagen] 完成！SFT: {result['sft']} 条，GRPO: {result['grpo']} 条")
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        sys.exit(1)
```

- [ ] **Step 3b: 若 `run.py` 不存在，创建最小骨架**

```python
#!/usr/bin/env python3
# run.py — LLM Foundry Simulator CLI 入口（Plan 0 最小骨架，Plan 1 扩充）
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="run.py",
        description="LLM Foundry Simulator",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # datagen 子命令
    parser_datagen = subparsers.add_parser(
        "datagen",
        help="用 DeepSeek API 生成 SFT/GRPO 训练数据",
    )
    parser_datagen.add_argument(
        "--config",
        default="configs/datagen.yaml",
        help="数据生成配置文件路径",
    )

    args = parser.parse_args()

    if args.command == "datagen":
        from llm_foundry.stage0_datagen import DataGenConfig, run_datagen
        try:
            cfg = DataGenConfig.from_yaml(args.config)
        except FileNotFoundError:
            print(f"[错误] 配置文件不存在: {args.config}", file=sys.stderr)
            sys.exit(1)
        try:
            result = run_datagen(cfg)
            print(f"\n[datagen] 完成！SFT: {result['sft']} 条，GRPO: {result['grpo']} 条")
        except ValueError as e:
            print(f"[错误] {e}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试，确认通过**

```bash
pytest tests/test_datagen.py::test_datagen_cli_missing_api_key -v
```

Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add run.py tests/test_datagen.py
git commit -m "feat(datagen): add datagen subcommand to run.py CLI"
```

---

## Task 7: 全量测试 + 冒烟测试 + 正式生成

**Files:**
- Test: `tests/test_datagen.py`

- [ ] **Step 1: 运行全部 datagen 单元测试**

```bash
pytest tests/test_datagen.py -v
```

Expected: 全部 PASS

- [ ] **Step 2: 冒烟测试（各生成 3 条，用临时目录，跨平台）**

```bash
# Windows PowerShell：
$env:SJTU_API_KEY="your_actual_key"

# Windows Bash / Linux：
export SJTU_API_KEY=your_actual_key

python -c "
import asyncio, tempfile, pathlib
from llm_foundry.stage0_datagen.sft_gen import SFTGenerator
from llm_foundry.stage0_datagen.grpo_gen import GRPOGenerator
from llm_foundry.stage0_datagen.client import DeepSeekClient

async def smoke():
    tmp = pathlib.Path(tempfile.gettempdir())
    client = DeepSeekClient()
    try:
        sft = SFTGenerator(client=client)
        grpo = GRPOGenerator(client=client)
        n1 = await sft.generate(3, str(tmp / 'smoke_sft.jsonl'))
        n2 = await grpo.generate(3, str(tmp / 'smoke_grpo.jsonl'))
        print(f'SFT: {n1} 条，GRPO: {n2} 条')
    finally:
        await client.close()

asyncio.run(smoke())
"
```

Expected: 输出 `SFT: 3 条，GRPO: 3 条`，检查临时目录下两个文件内容格式正确

- [ ] **Step 3: 正式生成各 500 条**

```bash
python run.py datagen --config configs/datagen.yaml
```

Expected:
- 进度条显示实时进度（先跑 SFT，再跑 GRPO）
- 完成后输出 `[datagen] 完成！SFT: 500 条，GRPO: 500 条`
- 验证文件行数：

```bash
wc -l results/datagen/sft_data.jsonl results/datagen/grpo_data.jsonl
```

Expected: 各 500 行

```bash
python -c "
import json
with open('results/datagen/sft_data.jsonl') as f:
    first = json.loads(f.readline())
print(list(first.keys()))
"
```

Expected: `['prompt', 'response']`

- [ ] **Step 4: 最终 Commit**

```bash
git add pyproject.toml
git commit -m "chore(datagen): finalize asyncio config in pyproject.toml"
```

---

## 验收标准

| 检查项 | 命令 | 期望结果 |
|--------|------|----------|
| 单元测试全通过 | `pytest tests/test_datagen.py -v` | 全部 PASS |
| API key 保护 | 不设置 `SJTU_API_KEY` 运行 CLI | 非零退出 + 错误提示含 "SJTU_API_KEY" |
| 断点续传 | 中途 Ctrl+C 后重新运行 | 从中断处继续，不重复生成 |
| 数量精确 | 生成完成后统计行数 | sft: 500 行，grpo: 500 行 |
| SFT 格式正确 | 解析第一行 JSON | 含 `prompt`、`response` 字段 |
| GRPO 格式正确 | 解析第一行 JSON | 含 `problem`、`solution`、`answer` 字段 |
| 限速合规 | 观察生成速度（≈8条/6s） | 不超过 80 req/min |
