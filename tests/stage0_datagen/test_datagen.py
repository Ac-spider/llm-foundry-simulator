"""Tests for stage0_datagen module."""
import os
import pathlib
import subprocess
import sys
import tempfile

import pytest

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent


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
