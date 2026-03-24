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
