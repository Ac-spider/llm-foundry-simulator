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
