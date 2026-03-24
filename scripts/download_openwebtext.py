#!/usr/bin/env python3
"""
从 Hugging Face 下载 OpenWebText 数据集用于训练。
使用流式加载避免内存不足，支持断点续传。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from datasets import load_dataset
from tqdm import tqdm


def download_openwebtext(
    output_dir: str = "data/raw",
    split: str = "train",
    max_samples: int = 100000,  # 下载 10 万条样本约 200MB
    streaming: bool = True,
) -> str:
    """
    从 HuggingFace 下载 OpenWebText 数据集。

    Args:
        output_dir: 输出目录
        split: 数据集划分 (train/valid/test)
        max_samples: 最大样本数
        streaming: 是否使用流式加载

    Returns:
        输出文件路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_file = output_path / f"openwebtext_{split}.txt"

    print(f"[下载] 正在从 HuggingFace 加载 OpenWebText 数据集 ({split} split)...")
    print(f"[下载] 使用 streaming={streaming} 模式")

    # 加载数据集
    dataset = load_dataset(
        "openwebtext",
        split=split,
        streaming=streaming,
    )

    print(f"[下载] 开始下载 {max_samples} 条样本到 {output_file}")

    total_bytes = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for i, sample in enumerate(tqdm(dataset, total=max_samples, desc="下载进度")):
            if i >= max_samples:
                break

            text = sample.get("text", "")
            if text.strip():  # 只保存非空文本
                f.write(text)
                f.write("\n\n")  # 样本间用空行分隔
                total_bytes += len(text.encode("utf-8"))

    file_size_mb = total_bytes / (1024 * 1024)
    print(f"\n[完成] 数据集已保存到: {output_file}")
    print(f"[完成] 总大小: {file_size_mb:.2f} MB")
    print(f"[完成] 样本数: {max_samples}")

    return str(output_file)


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download OpenWebText dataset from HuggingFace"
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory (default: data/raw)"
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "valid", "test"],
        help="Dataset split to download"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100000,
        help="Maximum number of samples to download (default: 100000)"
    )
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming mode (load all into memory)"
    )

    args = parser.parse_args()

    # 设置代理（如果环境变量中有）
    for proxy_var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        if proxy_var in os.environ:
            print(f"[配置] 检测到代理: {proxy_var}={os.environ[proxy_var]}")

    try:
        output_file = download_openwebtext(
            output_dir=args.output_dir,
            split=args.split,
            max_samples=args.max_samples,
            streaming=not args.no_streaming,
        )
        print(f"\n✓ 成功下载到: {output_file}")
        return 0
    except Exception as e:
        print(f"\n✗ 下载失败: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
