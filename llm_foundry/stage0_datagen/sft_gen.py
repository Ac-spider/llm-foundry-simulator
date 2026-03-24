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
