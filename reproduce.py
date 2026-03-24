#!/usr/bin/env python3
"""
LLM Foundry Reproduce - 一键复现入口

一键运行各 Stage 的复现脚本并验证结果。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence

# Stage 依赖关系图
STAGE_DEPENDENCIES: dict[str, list[str]] = {
    "tokenize": [],
    "train": ["tokenize"],
    "scaling": ["train"],
    "data": [],
    "align": ["train"],
}

# 所有可用的 stages（按依赖顺序排序）
ALL_STAGES: list[str] = ["tokenize", "train", "scaling", "data", "align"]


def get_stages_to_run(stage_arg: str) -> list[str]:
    """根据命令行参数确定要执行的 stages 列表。"""
    if stage_arg == "all":
        return ALL_STAGES.copy()
    return [stage_arg]


def check_dependencies(stages: list[str]) -> bool:
    """检查 stage 依赖关系是否满足。"""
    executed = set()
    for stage in stages:
        deps = STAGE_DEPENDENCIES.get(stage, [])
        for dep in deps:
            if dep not in executed and dep not in stages:
                print(f"[错误] Stage '{stage}' 依赖 '{dep}'，但 '{dep}' 不在执行列表中", file=sys.stderr)
                print(f"       请先运行: python reproduce.py --stage {dep}", file=sys.stderr)
                return False
        executed.add(stage)
    return True


def run_script(script_path: Path, dry_run: bool = False) -> tuple[int, str]:
    """
    执行脚本并返回结果。

    Returns:
        (returncode, output)
    """
    if dry_run:
        return 0, f"[DRY-RUN] 将执行: bash {script_path}"

    try:
        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600,  # 1小时超时
        )
        output = result.stdout + result.stderr
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "[错误] 脚本执行超时（超过1小时）"
    except Exception as e:
        return 1, f"[错误] 执行失败: {e}"


def run_verify(stage: str, dry_run: bool = False) -> tuple[int, str]:
    """
    运行验证脚本。

    Returns:
        (returncode, output)
    """
    verify_script = Path("reproduce/verify.py")
    if not verify_script.exists():
        return 1, f"[错误] 验证脚本不存在: {verify_script}"

    if dry_run:
        return 0, f"[DRY-RUN] 将执行: python {verify_script} --stage {stage}"

    try:
        result = subprocess.run(
            [sys.executable, str(verify_script), "--stage", stage],
            capture_output=True,
            text=True,
            timeout=300,  # 5分钟超时
        )
        output = result.stdout + result.stderr
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "[错误] 验证超时（超过5分钟）"
    except Exception as e:
        return 1, f"[错误] 验证失败: {e}"


def print_stage_header(stage: str) -> None:
    """打印 stage 头部信息。"""
    print(f"\n--- Stage: {stage} ---")


def print_command(cmd: str) -> None:
    """打印要执行的命令。"""
    print(f"$ {cmd}")


def print_output(output: str, max_lines: int = 50) -> None:
    """打印脚本输出（限制行数）。"""
    lines = output.splitlines()
    if len(lines) > max_lines:
        print("\n".join(lines[:max_lines]))
        print(f"... ({len(lines) - max_lines} more lines)")
    else:
        print(output)


def run_stage(stage: str, skip_verify: bool, dry_run: bool) -> dict:
    """
    执行单个 stage 的复现流程。

    Returns:
        包含执行结果的字典
    """
    result = {
        "stage": stage,
        "script_status": "SKIPPED",
        "script_output": "",
        "verify_status": "SKIPPED",
        "verify_output": "",
        "overall": "PASS",
    }

    print_stage_header(stage)

    # 1. 执行 reproduce/{stage}.sh
    script_path = Path(f"reproduce/{stage}.sh")
    if script_path.exists():
        print_command(f"bash {script_path}")
        returncode, output = run_script(script_path, dry_run)
        result["script_output"] = output
        print_output(output)

        if returncode == 0:
            result["script_status"] = "PASS"
        else:
            result["script_status"] = "FAIL"
            result["overall"] = "FAIL"
            print(f"\nStatus: FAIL (script returned {returncode})")
            return result
    else:
        msg = f"[警告] 脚本不存在: {script_path}"
        print(msg)
        result["script_output"] = msg
        if not dry_run:
            result["script_status"] = "FAIL"
            result["overall"] = "FAIL"
            return result

    # 2. 验证（除非 --skip-verify）
    if not skip_verify:
        print_command(f"python reproduce/verify.py --stage {stage}")
        returncode, output = run_verify(stage, dry_run)
        result["verify_output"] = output

        # 提取验证状态
        if dry_run:
            result["verify_status"] = "DRY-RUN"
        elif returncode == 0:
            result["verify_status"] = "PASS"
        else:
            result["verify_status"] = "FAIL"
            result["overall"] = "FAIL"

        # 打印验证结果摘要
        if "Status:" in output:
            for line in output.splitlines():
                if "Status:" in line:
                    print(line)
                    break
        else:
            print(f"Status: {result['verify_status']}")
    else:
        print("[跳过验证] --skip-verify 已设置")

    return result


def print_summary(results: list[dict]) -> None:
    """打印汇总报告。"""
    print("\n=== Summary ===")

    passed = 0
    failed = 0

    for result in results:
        stage = result["stage"]
        overall = result["overall"]

        if overall == "PASS":
            print(f"{stage}: PASS")
            passed += 1
        else:
            print(f"{stage}: FAIL")
            failed += 1

    total = passed + failed
    print(f"\nOverall: {passed}/{total} stages passed")

    if failed > 0:
        print("\n失败的 stages:")
        for result in results:
            if result["overall"] != "PASS":
                print(f"  - {result['stage']}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="reproduce",
        description="LLM Foundry Reproduce - 一键复现各 Stage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python reproduce.py --stage all              # 运行所有 stages
  python reproduce.py --stage tokenize         # 只运行 tokenize stage
  python reproduce.py --stage train --dry-run  # 只打印命令，不实际运行
  python reproduce.py --stage all --skip-verify # 跳过验证步骤
        """,
    )

    parser.add_argument(
        "--stage",
        type=str,
        choices=["all", "tokenize", "train", "scaling", "data", "align"],
        default="all",
        help="指定要复现的 stage (default: all)",
    )

    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="跳过验证步骤",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印要执行的命令，不实际运行",
    )

    args = parser.parse_args()

    # 打印头部信息
    print("=== LLM Foundry Reproduce ===")

    # 确定要执行的 stages
    stages = get_stages_to_run(args.stage)
    print(f"Stages to run: {stages}")

    # 检查依赖关系
    if not check_dependencies(stages):
        return 1

    # 执行各 stage
    results: list[dict] = []
    for stage in stages:
        result = run_stage(stage, args.skip_verify, args.dry_run)
        results.append(result)

        # 如果某个 stage 失败且不是 dry-run，询问是否继续
        if result["overall"] == "FAIL" and not args.dry_run:
            print(f"\n[警告] Stage '{stage}' 失败")
            # 继续执行其他 stages，但记录失败

    # 打印汇总报告
    print_summary(results)

    # 返回状态码
    failed_count = sum(1 for r in results if r["overall"] != "PASS")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
