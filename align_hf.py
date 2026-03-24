#!/usr/bin/env python3
"""
Stage 5 Alignment Training - HuggingFace Implementation
单独脚本，不影响前面 Stages 0-4 的代码
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def run_sft_hf(config_path: str, args) -> int:
    """Run SFT training with HuggingFace."""
    print("[align_hf] 使用 HuggingFace 实现运行 SFT...")

    try:
        import yaml
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )
        from datasets import Dataset
        import torch

        # Load config
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        # Get model and tokenizer names
        model_name = cfg.get("model", {}).get("name", "gpt2")
        tokenizer_name = cfg.get("tokenizer", {}).get("name", model_name)
        method_cfg = cfg.get("sft", {})

        print(f"[align_hf] 加载模型: {model_name}")
        print(f"[align_hf] 加载 tokenizer: {tokenizer_name}")

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Load model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[align_hf] 使用设备: {device}")

        model = AutoModelForCausalLM.from_pretrained(model_name)
        model = model.to(device)

        # Load data
        data_path = method_cfg.get("data_path", "data/sft_train.jsonl")
        eval_path = method_cfg.get("eval_data_path", "data/sft_eval.jsonl")
        max_length = method_cfg.get("max_length", 256)

        print(f"[align_hf] 加载训练数据: {data_path}")

        # Read JSONL data
        import json

        train_data = []
        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                # Combine prompt and response
                text = item["prompt"] + " " + item["response"]
                train_data.append({"text": text})

        # Create dataset
        train_dataset = Dataset.from_list(train_data)

        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

        train_dataset = train_dataset.map(tokenize_function, batched=True)

        # Load eval data if exists
        eval_dataset = None
        if os.path.exists(eval_path):
            print(f"[align_hf] 加载验证数据: {eval_path}")
            eval_data = []
            with open(eval_path) as f:
                for line in f:
                    item = json.loads(line.strip())
                    text = item["prompt"] + " " + item["response"]
                    eval_data.append({"text": text})
            eval_dataset = Dataset.from_list(eval_data)
            eval_dataset = eval_dataset.map(tokenize_function, batched=True)

        # Setup training arguments
        output_dir = cfg.get("output", {}).get("base_dir", "./outputs/align_hf")
        os.makedirs(output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=float(method_cfg.get("learning_rate", 1e-4)),
            num_train_epochs=int(method_cfg.get("num_epochs", 3)),
            per_device_train_batch_size=int(method_cfg.get("batch_size", 2)),
            gradient_accumulation_steps=int(method_cfg.get("gradient_accumulation_steps", 2)),
            logging_steps=1,
            save_steps=method_cfg.get("save_steps", 5),
            eval_steps=method_cfg.get("eval_steps", 5),
            eval_strategy="steps" if eval_dataset else "no",
            save_total_limit=cfg.get("output", {}).get("save_total_limit", 3),
            fp16=torch.cuda.is_available(),
            report_to="none",
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )

        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

        # Train
        print("[align_hf] 开始训练...")
        trainer.train()

        # Save final model
        final_path = os.path.join(output_dir, "final_model")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        print(f"[align_hf] 模型已保存至: {final_path}")

        return 0

    except Exception as e:
        print(f"[错误] SFT 训练失败: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description="Stage 5 Alignment - HuggingFace Implementation",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/align_hf.yaml",
        help="Path to alignment config YAML",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sft", "dpo", "grpo"],
        default="sft",
        help="Alignment method",
    )

    args = parser.parse_args()

    if args.method == "sft":
        return run_sft_hf(args.config, args)
    else:
        print(f"[align_hf] 方法 {args.method} 暂未实现")
        return 1


if __name__ == "__main__":
    sys.exit(main())
