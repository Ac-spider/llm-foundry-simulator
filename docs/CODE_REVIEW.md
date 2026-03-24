# LLM Foundry Simulator 代码审查报告

审查日期：2026-03-25
审查范围：`llm_foundry/` 全部模块

---

## 总体评价

项目结构清晰，从 CS336 课程代码整合到统一 CLI 管道，工程思路合理，代码风格一致。以下按严重程度分级列出问题。

---

## Critical（须修复）

### 1. `model.py:315` — `forward()` 强假设输入为 2D

```python
def forward(self, x: Int[Tensor, " ... sequence_length"]):
    _, sequence_length = x.size()  # 崩溃：若 x 是 3D 会报 ValueError
```

类型注解标注了 `...`（任意前缀维度），但 `x.size()` 只能解包 2 个值。若传入非 `(batch, seq)` 的形状会直接崩溃。

**修复建议：**
```python
*batch_dims, sequence_length = x.shape
```

---

### 2. `attention.py:163-169` — `lru_cache` 与全局变量双重缓存逻辑矛盾

```python
@lru_cache(maxsize=1)
def get_attention_fn(use_flash_attn: bool = True):
    global _attention_fn_cache
    if _attention_fn_cache is not None:   # 冗余且有 bug
        return _attention_fn_cache
```

`@lru_cache` 已按参数缓存，但全局变量 `_attention_fn_cache` 无视 `use_flash_attn` 参数（永远返回第一次调用的结果）。若先调用 `get_attention_fn(False)` 再调用 `get_attention_fn(True)`，两次都会错误地返回 `sdpa`。

**修复建议：** 删除全局变量 `_attention_fn_cache` 及相关逻辑，只保留 `@lru_cache`。

---

### 3. `grpo.py:317` — Tensor 高级索引维度一致性风险

```python
mb_advantages = advantages[micro_indices]      # micro_indices 是 list[int]
mb_old_log_probs = old_log_probs[micro_indices] # old_log_probs 形状需与 micro_indices 对应
```

`advantages` 形状为 `(N*G,)`，用 list 索引正确。但 `old_log_probs` 形状为 `(N*G, seq_len)`，用相同 `micro_indices` 索引时取的是第 0 维，需确认与 `advantages` 的对应关系完全一致，否则会静默返回错误数据。建议加断言：
```python
assert old_log_probs.shape[0] == advantages.shape[0]
```

---

## Important（建议修复）

### 4. `tokenizer.py:519-525` — `pickle.load()` 反序列化不可信数据存在安全风险

```python
with open(vocab_path, "rb") as f:
    vocab = pickle.load(f)  # 若文件来自外部来源，可执行任意代码
```

Pickle 反序列化任意文件存在代码执行风险。建议在文档中标注此风险，或考虑改用 JSON 格式存储 vocab（bytes 值需 base64 编码）。

---

### 5. `model.py:372`、`sft.py:387`、`grpo.py:499` — `torch.load()` 缺少 `weights_only=True`

```python
state_dict = torch.load(weights_path)  # PyTorch 2.x FutureWarning，2.6+ 行为变更
```

**修复建议：**
```python
state_dict = torch.load(weights_path, weights_only=True)
```

三处文件均需修改。

---

### 6. `trainer.py:70` — DDP 检测逻辑不够明确

```python
is_ddp = dist.is_available() and int(os.environ.get("RANK", -1)) != -1
```

意图是检测是否在 torchrun 环境下运行，但写法不够直观（`RANK=0` 时 `int("0") != -1` 为 True，实际正确）。

**更清晰的写法：**
```python
is_ddp = dist.is_available() and "RANK" in os.environ
```

---

### 7. `sft.py:78-83` — truncate 逻辑有 bug，response 可能未被截断

```python
if len(concat_ids) > self.max_length:
    concat_ids = concat_ids[:self.max_length]
    if len(prompt_ids) >= self.max_length:
        prompt_ids = prompt_ids[:self.max_length - 1]
        concat_ids = prompt_ids + response_ids  # response_ids 未截断，总长可能远超 max_length
```

重新赋值后 `concat_ids` 没有再次截断，实际长度可能仍然超过 `max_length`。

**修复建议：**
```python
if len(prompt_ids) >= self.max_length:
    prompt_ids = prompt_ids[:self.max_length - 1]
concat_ids = (prompt_ids + response_ids)[:self.max_length]
```

---

### 8. `pipeline.py:249-253` — `process_file` 一次性读整个文件到内存

```python
with open(input_path, "r", encoding="utf-8") as f:
    content = f.read()  # 数十 GB 文件会 OOM
```

数据质量过滤通常处理大规模语料，应改为流式处理（逐行/逐块读取），而不是一次全部载入内存。

---

## Minor（优化建议）

### 9. `attention.py:71-75` — `_flash_attention_compiled` 每次调用都重新编译

```python
def _flash_attention_compiled(q, k, v, is_causal=True):
    @torch.compile(mode="max-autotune", fullgraph=False)
    def _compiled_sdpa(q, k, v, is_causal):   # 每次调用都重新定义和编译
        ...
    return _compiled_sdpa(q, k, v, is_causal)
```

`_compiled_sdpa` 在函数体内定义，每次调用都会触发重新编译，丧失 `torch.compile` 缓存优势。

**修复建议：** 将 `_compiled_sdpa` 提升为模块级函数。

---

### 10. `trainer.py:184-186` — warn 在每个训练 step 都打印

```python
if grad_accum_steps > 1:
    print(f"[WARN] DDP mode: gradient_accumulation_steps forced to 1")
grad_accum_steps = 1
```

这段代码在训练循环内，每个 step 都会打印 warn，产生大量无用日志。

**修复建议：** 将此检测移到训练循环开始前，只打印一次。

---

### 11. `backends/inference.py:294` — `import torch` 放在文件末尾

```python
# Import torch for HF backend
import torch  # noqa: E402   ← 文件末尾
```

这是反模式，违反 PEP 8。`torch` 在 `HFInferenceBackend` 方法中使用，但导入语句在文件末尾。应移到文件顶部 imports 区。

---

### 12. `grpo.py` — `rollout()` 串行生成，未利用 batch 接口

```python
for prompt, gt in zip(prompts, ground_truths):
    for _ in range(self.group_size):
        response = self.inference_backend.generate(prompt, config)  # 逐条调用
```

`HFInferenceBackend` 已实现 `generate_batch()`，此处完全没有利用，导致推理吞吐极低。

**修复建议：** 将所有 prompts 汇总后调用 `generate_batch()`。

---

### 13. `tokenizer.py:378-428` — `_encode_chunk` 编码复杂度 O(n²)

每次 BPE merge 扫描整个 `b_list` 找最低 rank，内外层循环嵌套。对长 token 序列性能较差。

**优化建议：** 使用优先队列（`heapq`）维护候选 pair，可将复杂度降至 O(n log n)，这是 HuggingFace tokenizers 的标准做法。

---

## 问题汇总表

| 编号 | 文件 | 行号 | 严重程度 | 问题摘要 |
|------|------|------|----------|----------|
| 1 | `common/model.py` | 315 | Critical | `forward` 假设 2D 输入，3D 会崩 |
| 2 | `backends/attention.py` | 163 | Critical | lru_cache + 全局变量双重缓存逻辑矛盾 |
| 3 | `stage5_align/grpo.py` | 317 | Critical | Tensor 索引维度一致性风险 |
| 4 | `stage1_tokenize/tokenizer.py` | 519 | Important | pickle 不安全反序列化 |
| 5 | `model.py`/`sft.py`/`grpo.py` | 372/387/499 | Important | `torch.load` 缺少 `weights_only=True` |
| 6 | `stage2_train/trainer.py` | 70 | Important | DDP 检测逻辑不够明确 |
| 7 | `stage5_align/sft.py` | 78 | Important | truncate bug：response 未再截断 |
| 8 | `stage4_data/pipeline.py` | 249 | Important | 整个文件一次性读入内存 |
| 9 | `backends/attention.py` | 71 | Minor | torch.compile 每次调用重新编译 |
| 10 | `stage2_train/trainer.py` | 184 | Minor | warn 在每个 step 都打印 |
| 11 | `backends/inference.py` | 294 | Minor | import 放在文件末尾 |
| 12 | `stage5_align/grpo.py` | — | Minor | rollout 串行，未用 batch 接口 |
| 13 | `stage1_tokenize/tokenizer.py` | 378 | Minor | BPE 编码 O(n²)，可用 heapq 优化 |
