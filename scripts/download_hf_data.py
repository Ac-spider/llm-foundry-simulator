#!/usr/bin/env python3
"""
从 HuggingFace 下载 SFT 和 DPO 数据集
"""

import json
import os
from pathlib import Path
from datasets import load_dataset

# 创建数据目录
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

def download_sft_alpaca():
    """下载 Alpaca 指令微调数据集 (52k 条)"""
    print("[SFT] 下载 Alpaca 数据集...")
    dataset = load_dataset("tatsu-lab/alpaca", split="train")

    # 转换为 prompt-response 格式
    output_file = data_dir / "sft_train.jsonl"
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            # Alpaca 格式: instruction, input, output
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")

            # 组合 prompt
            if input_text:
                prompt = f"{instruction}\nInput: {input_text}"
            else:
                prompt = instruction

            data = {
                "prompt": prompt,
                "response": output_text
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
            if count >= 1000:  # 限制 1000 条用于快速实验
                break

    print(f"[SFT] Alpaca: 已下载 {count} 条到 {output_file}")
    return count

def download_sft_dolly():
    """下载 Dolly 15k 数据集"""
    print("[SFT] 下载 Dolly 数据集...")
    dataset = load_dataset("databricks/databricks-dolly-15k", split="train")

    output_file = data_dir / "sft_train.jsonl"
    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            instruction = item.get("instruction", "")
            context = item.get("context", "")
            response = item.get("response", "")

            if context:
                prompt = f"{instruction}\nContext: {context}"
            else:
                prompt = instruction

            data = {
                "prompt": prompt,
                "response": response
            }
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1

    print(f"[SFT] Dolly: 已下载 {count} 条到 {output_file}")
    return count

def download_dpo_hh_rlhf():
    """下载 Anthropic HH-RLHF 数据集"""
    print("[DPO] 下载 HH-RLHF 数据集...")

    try:
        dataset = load_dataset("Anthropic/hh-rlhf", split="train")

        output_file = data_dir / "dpo_train.jsonl"
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                # HH-RLHF 格式: chosen, rejected
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")

                # 提取 prompt (通常以 Assistant: 或 Human: 分隔)
                if "Assistant:" in chosen:
                    parts = chosen.split("Assistant:")
                    prompt = parts[0].strip() + "\nAssistant:"
                    chosen_response = "Assistant:".join(parts[1:]).strip()
                else:
                    prompt = ""
                    chosen_response = chosen

                if "Assistant:" in rejected:
                    parts = rejected.split("Assistant:")
                    rejected_response = "Assistant:".join(parts[1:]).strip()
                else:
                    rejected_response = rejected

                data = {
                    "prompt": prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
                if count >= 1000:
                    break

        print(f"[DPO] HH-RLHF: 已下载 {count} 条到 {output_file}")
        return count
    except Exception as e:
        print(f"[DPO] HH-RLHF 下载失败: {e}")
        return 0

def download_dpo_orca():
    """下载 Intel Orca DPO Pairs 数据集"""
    print("[DPO] 下载 Intel Orca DPO Pairs 数据集...")

    try:
        dataset = load_dataset("Intel/orca_dpo_pairs", split="train")

        output_file = data_dir / "dpo_train.jsonl"
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                # Orca DPO 格式: system, question, chat, chosen, rejected
                system = item.get("system", "")
                question = item.get("question", "")
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")

                # 组合 prompt
                if system:
                    prompt = f"{system}\n\n{question}"
                else:
                    prompt = question

                data = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
                if count >= 1000:
                    break

        print(f"[DPO] Orca DPO: 已下载 {count} 条到 {output_file}")
        return count
    except Exception as e:
        print(f"[DPO] Orca DPO 下载失败: {e}")
        return 0

def download_dpo_ultrafeedback():
    """下载 UltraFeedback Binarized 数据集"""
    print("[DPO] 下载 UltraFeedback Binarized 数据集...")

    try:
        dataset = load_dataset("trl-lib/ultrafeedback_binarized", split="train")

        output_file = data_dir / "dpo_train.jsonl"
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for item in dataset:
                # UltraFeedback 格式包含 prompts, chosen, rejected
                prompt = item.get("prompt", "")
                chosen = item.get("chosen", "")
                rejected = item.get("rejected", "")

                # 处理 chosen/rejected (可能是列表或字符串)
                if isinstance(chosen, list):
                    chosen_text = chosen[0].get("content", "") if chosen else ""
                elif isinstance(chosen, dict):
                    chosen_text = chosen.get("content", "")
                else:
                    chosen_text = str(chosen)

                if isinstance(rejected, list):
                    rejected_text = rejected[0].get("content", "") if rejected else ""
                elif isinstance(rejected, dict):
                    rejected_text = rejected.get("content", "")
                else:
                    rejected_text = str(rejected)

                # 处理 prompt
                if isinstance(prompt, list):
                    prompt_text = prompt[0].get("content", "") if prompt else ""
                elif isinstance(prompt, dict):
                    prompt_text = prompt.get("content", "")
                else:
                    prompt_text = str(prompt)

                data = {
                    "prompt": prompt_text,
                    "chosen": chosen_text,
                    "rejected": rejected_text
                }
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                count += 1
                if count >= 1000:
                    break

        print(f"[DPO] UltraFeedback: 已下载 {count} 条到 {output_file}")
        return count
    except Exception as e:
        print(f"[DPO] UltraFeedback 下载失败: {e}")
        return 0

def main():
    print("=" * 60)
    print("从 HuggingFace 下载 SFT 和 DPO 数据集")
    print("=" * 60)

    # 下载 SFT 数据
    print("\n### SFT 数据集 ###")
    try:
        sft_count = download_sft_alpaca()
    except Exception as e:
        print(f"[SFT] Alpaca 下载失败，尝试 Dolly: {e}")
        try:
            sft_count = download_sft_dolly()
        except Exception as e2:
            print(f"[SFT] Dolly 也失败了: {e2}")
            sft_count = 0

    # 下载 DPO 数据
    print("\n### DPO 数据集 ###")
    dpo_count = 0

    # 尝试多个 DPO 数据集
    for download_fn in [download_dpo_ultrafeedback, download_dpo_orca, download_dpo_hh_rlhf]:
        if dpo_count == 0:
            try:
                dpo_count = download_fn()
            except Exception as e:
                print(f"跳过: {e}")
                continue

    # 生成 eval 数据 (从 train 分割一部分)
    if sft_count > 0:
        generate_eval_split("sft_train.jsonl", "sft_eval.jsonl", n_eval=min(100, sft_count // 10))
    if dpo_count > 0:
        generate_eval_split("dpo_train.jsonl", "dpo_eval.jsonl", n_eval=min(100, dpo_count // 10))

    print("\n" + "=" * 60)
    print("下载完成!")
    print(f"  - SFT 训练数据: {sft_count} 条")
    print(f"  - DPO 训练数据: {dpo_count} 条")
    print("=" * 60)

def generate_eval_split(train_file, eval_file, n_eval=100):
    """从训练集生成分割验证集"""
    train_path = data_dir / train_file
    eval_path = data_dir / eval_file

    if not train_path.exists():
        return

    # 读取训练集
    with open(train_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 取前 n_eval 条作为验证集
    eval_lines = lines[:n_eval]
    train_lines = lines[n_eval:]

    # 保存验证集
    with open(eval_path, "w", encoding="utf-8") as f:
        f.writelines(eval_lines)

    # 更新训练集
    with open(train_path, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    print(f"[Split] {train_file}: 训练集 {len(train_lines)} 条, 验证集 {len(eval_lines)} 条")

if __name__ == "__main__":
    main()
