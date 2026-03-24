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
