#!/usr/bin/env python3
"""
LLM Foundry Simulator - Unified CLI pipeline for LLM training workflows.

This is the main entry point for all stages of the LLM training pipeline:
- env: Environment check
- datagen: Data generation (Plan 0)
- tokenize: Tokenization (Plan 2)
- train: Training (Plan 3)
- scaling: Scaling laws (Plan 4)
- data: Data quality pipeline (Plan 5)
- align: Alignment (SFT, DPO, RLHF) (Plan 6)
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

import torch

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def cmd_env(args: argparse.Namespace) -> int:
    """Environment check command."""
    from llm_foundry.common.env_check import check_environment

    status = check_environment()
    status.print_report()
    return 0


def cmd_datagen(args: argparse.Namespace) -> int:
    """Data generation command (Plan 0)."""
    from llm_foundry.stage0_datagen import DataGenConfig, run_datagen

    try:
        cfg = DataGenConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"[错误] 配置文件不存在: {args.config}", file=sys.stderr)
        return 1
    try:
        result = run_datagen(cfg)
        print(f"\n[datagen] 完成！SFT: {result['sft']} 条，GRPO: {result['grpo']} 条")
    except ValueError as e:
        print(f"[错误] {e}", file=sys.stderr)
        return 1
    return 0


def cmd_tokenize(args: argparse.Namespace) -> int:
    """Tokenization command (Plan 2)."""
    from llm_foundry.stage1_tokenize import BPETokenizer, TokenizerConfig

    if BPETokenizer is None:
        print("[错误] Tokenizer模块尚未完全实现", file=sys.stderr)
        return 1

    try:
        cfg = TokenizerConfig.from_yaml(args.config)
    except FileNotFoundError:
        print(f"[错误] 配置文件不存在: {args.config}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[错误] 配置加载失败: {e}", file=sys.stderr)
        return 1

    try:
        # Train or load tokenizer based on config
        if cfg.data.train_file and not cfg.skip_training:
            print(f"[tokenize] 训练tokenizer，vocab_size={cfg.training.vocab_size}")
            tokenizer = BPETokenizer.train(
                input_path=cfg.data.train_file,
                vocab_size=cfg.training.vocab_size,
                special_tokens=cfg.training.special_tokens,
            )
            # Save the trained tokenizer
            save_path = Path(cfg.output.output_dir) / cfg.output.name
            tokenizer.save(save_path)
            print(f"[tokenize] Tokenizer已保存至: {save_path}")
        else:
            print(f"[tokenize] 加载已有tokenizer: {cfg.output.output_dir}/{cfg.output.name}")
            save_path = Path(cfg.output.output_dir) / cfg.output.name
            tokenizer = BPETokenizer.load(save_path)

        # Validate on test file if provided
        if cfg.data.test_file:
            print(f"[tokenize] 在测试集上验证: {cfg.data.test_file}")
            with open(cfg.data.test_file, "r", encoding="utf-8") as f:
                test_text = f.read()
            encoded = tokenizer.encode(test_text)
            decoded = tokenizer.decode(encoded)
            print(f"[tokenize] 测试通过 - 编码后token数: {len(encoded)}")

        print(f"[tokenize] 完成！输出目录: {cfg.output.output_dir}/{cfg.output.name}")
        return 0

    except Exception as e:
        print(f"[错误] Tokenization失败: {e}", file=sys.stderr)
        return 1


def cmd_train(args: argparse.Namespace) -> int:
    """Training command (Plan 3).

    支持选项（均可在 YAML 或命令行参数中设置，命令行优先级更高）：
    - --config：YAML 配置文件路径（默认 configs/train.yaml）
    - --data：覆盖 training.data_path
    - --output：覆盖 output.base_dir
    - --flash-attn：强制启用 Flash Attention（覆盖 model.use_flash_attn）
    - --set：覆盖任意配置项（如 --set training.lr=1e-4）
    """
    from llm_foundry.stage2_train.trainer import Trainer
    from llm_foundry.common.env_check import check_environment
    from llm_foundry.common.config import load_config, namespace_to_dict, merge_configs

    # 加载配置文件
    config_path = args.config or "configs/train.yaml"
    try:
        cfg = load_config(config_path)
        cfg_dict = namespace_to_dict(cfg)
    except FileNotFoundError:
        print(f"[错误] 配置文件不存在: {config_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[错误] 配置加载失败: {e}", file=sys.stderr)
        return 1

    # 命令行参数覆盖
    if args.data:
        cfg_dict.setdefault("training", {})["data_path"] = args.data
    if args.output:
        cfg_dict.setdefault("output", {})["base_dir"] = args.output
    if args.flash_attn:
        cfg_dict.setdefault("model", {})["use_flash_attn"] = True

    # 处理 --set 参数（如 --set training.lr=1e-4）
    if args.set:
        for set_item in args.set:
            if "=" not in set_item:
                print(f"[错误] --set 参数格式错误: {set_item}，应为 key=value", file=sys.stderr)
                return 1
            key, value = set_item.split("=", 1)
            keys = key.split(".")
            # 尝试将值转换为适当类型
            try:
                # 尝试转换为数字
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # 保持字符串
                pass
            # 设置嵌套字典值
            current = cfg_dict
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            current[keys[-1]] = value

    # 环境检测：若要求 Flash Attention 但无 GPU，给出警告（不 crash）
    env = check_environment()
    if cfg_dict.get("model", {}).get("use_flash_attn", False) and not env.cuda_available:
        print("[WARN] use_flash_attn=True 但无可用 GPU，将自动 fallback 至 CPU SDPA")
        cfg_dict["model"]["use_flash_attn"] = False

    # DDP 检测：若 --ddp 但非 torchrun 环境，给出提示
    if getattr(args, 'ddp', False) and int(os.environ.get("RANK", -1)) == -1:
        print("[INFO] 检测到 --ddp，但当前非 torchrun 环境。")
        print("       多卡训练请使用：torchrun --nproc_per_node=N run.py train --config ...")
        print("       单卡训练继续执行...")

    print(f"[INFO] 开始训练，配置文件: {config_path}")
    trainer = Trainer(cfg_dict)
    print(f"[INFO] 输出目录: {trainer.run_dir}")
    trainer.train()
    print(f"[INFO] 训练完成。结果已写入: {trainer.run_dir}")
    return 0


def cmd_scaling(args: argparse.Namespace) -> int:
    """
    缩放律分析子命令 (Plan 4)。

    用法：
        python run.py scaling --config configs/scaling.yaml
    """
    from llm_foundry.common.config import load_config
    from llm_foundry.stage3_scaling import ScalingAnalyzer

    # 加载配置文件
    config_path = args.config or "configs/scaling.yaml"
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        print(f"[错误] 配置文件不存在: {config_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[错误] 配置加载失败: {e}", file=sys.stderr)
        return 1

    # 加载实验数据
    if hasattr(cfg, "scaling") and hasattr(cfg.scaling, "experiments_file"):
        experiments_file = cfg.scaling.experiments_file
    else:
        experiments_file = "data/scaling_experiments.json"
    try:
        with open(experiments_file, "r", encoding="utf-8") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"[错误] 实验数据文件不存在: {experiments_file}", file=sys.stderr)
        return 1
    except json.JSONDecodeError as e:
        print(f"[错误] JSON解析失败: {e}", file=sys.stderr)
        return 1

    # 执行缩放律分析
    print(f"[scaling] 开始缩放律分析，配置文件: {config_path}")
    analyzer = ScalingAnalyzer(cfg.__dict__)
    result = analyzer.run(input_data)

    print(f"\n[scaling] Chinchilla 拟合参数:")
    ch = result["chinchilla"]
    print(f"  E={ch['E']:.4f}  A={ch['A']:.2f}  alpha={ch['alpha']:.4f}  B={ch['B']:.2f}  beta={ch['beta']:.4f}")

    print(f"\n[scaling] IsoFLOPs 外推结果:")
    for opt in result["isoflops_optimal"]:
        print(f"  C={opt['compute']:.0e}: N={opt['n_params']:.2e}, D={opt['n_tokens']:.2e}")

    print(f"\n[scaling] 结果文件: {result['plots_dir']}/../scaling_params.json")
    print(f"[scaling] 图表目录: {result['plots_dir']}")

    return 0


def cmd_data(args: argparse.Namespace) -> int:
    """Data pipeline command (Plan 5).

    Usage:
        python run.py data --config configs/data.yaml
    """
    from llm_foundry.stage4_data import DataPipeline, DataPipelineConfig
    from llm_foundry.common.config import load_config

    # Load configuration
    config_path = args.config or "configs/data.yaml"
    try:
        cfg = load_config(config_path)
    except FileNotFoundError:
        print(f"[错误] 配置文件不存在: {config_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[错误] 配置加载失败: {e}", file=sys.stderr)
        return 1

    # Get configuration values
    if hasattr(cfg, 'data'):
        input_file = getattr(cfg.data, 'input_file', 'data/raw/openwebtext_train.txt')
        output_file = getattr(cfg.data, 'output_file', 'data/filtered/openwebtext_filtered.txt')
        min_length = getattr(cfg.data, 'min_length', 100)
        max_length = getattr(cfg.data, 'max_length', 100000)
    else:
        input_file = 'data/raw/openwebtext_train.txt'
        output_file = 'data/filtered/openwebtext_filtered.txt'
        min_length = 100
        max_length = 100000

    print(f"[data] 启动数据质量过滤流程")
    print(f"[data] 输入文件: {input_file}")
    print(f"[data] 输出文件: {output_file}")
    print(f"[data] 长度限制: [{min_length}, {max_length}]")

    # Check input file exists
    if not Path(input_file).exists():
        print(f"[错误] 输入文件不存在: {input_file}", file=sys.stderr)
        return 1

    # Create pipeline config
    pipeline_config = DataPipelineConfig(
        min_length=min_length,
        max_length=max_length,
        enable_gopher_filter=True,
        enable_deduplication=True,
    )

    # Create and run pipeline
    try:
        pipeline = DataPipeline(pipeline_config)

        # Process file
        stats = pipeline.process_file(
            input_path=input_file,
            output_path=output_file,
            doc_separator="\n\n",
        )

        # Print statistics
        print(f"\n[data] 处理完成！")
        print(f"[data] 总文档数: {stats['total']}")
        print(f"[data] 保留文档数: {stats['kept']}")
        print(f"[data] 过滤率: {(1 - stats['kept']/stats['total'])*100:.2f}%")
        print(f"[data]   - 长度过滤: {stats['filtered_length']}")
        print(f"[data]   - Gopher质量过滤: {stats['filtered_gopher']}")
        print(f"[data]   - 重复过滤: {stats['filtered_duplicate']}")
        print(f"[data] 输出文件: {output_file}")

        return 0

    except Exception as e:
        print(f"[错误] 数据处理失败: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


def cmd_align(args: argparse.Namespace) -> int:
    """
    Alignment training command (Plan 6).

    Supports three methods:
    - sft: Supervised Fine-Tuning
    - dpo: Direct Preference Optimization
    - grpo: Group Relative Policy Optimization

    Usage:
        python run.py align --config configs/align.yaml --method sft
        python run.py align --config configs/align.yaml --method dpo
        python run.py align --config configs/align.yaml --method grpo
    """
    from llm_foundry.common.config import load_config, namespace_to_dict, merge_configs
    from llm_foundry.common.model import ModelConfig

    # 1. Load configuration
    config_path = args.config or "configs/align.yaml"
    try:
        cfg = load_config(config_path)
        cfg_dict = namespace_to_dict(cfg)
    except FileNotFoundError:
        print(f"[错误] 配置文件不存在: {config_path}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"[错误] 配置加载失败: {e}", file=sys.stderr)
        return 1

    # 2. Command line overrides
    if args.method:
        cfg_dict["method"] = args.method
    if args.model:
        cfg_dict.setdefault("model", {})["name"] = args.model
    if args.data:
        cfg_dict.setdefault("data", {})["path"] = args.data
    if args.output:
        cfg_dict.setdefault("output", {})["dir"] = args.output

    # 3. Get method from config
    method = cfg_dict.get("method", "sft")
    if method not in ("sft", "dpo", "grpo"):
        print(f"[错误] 不支持的方法: {method}，可选: sft, dpo, grpo", file=sys.stderr)
        return 1

    # 4. Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[align] 使用设备: {device}")

    # 5. Load tokenizer (use project's BPE tokenizer)
    tokenizer_path = cfg_dict.get("tokenizer", {}).get("path", "results/tokenizer/bpe_tokenizer")
    print(f"[align] 加载 BPE tokenizer: {tokenizer_path}")
    try:
        from llm_foundry.stage1_tokenize import BPETokenizer

        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"[align] Tokenizer 加载成功，vocab_size={len(tokenizer)}")
    except Exception as e:
        print(f"[错误] Tokenizer加载失败: {e}", file=sys.stderr)
        return 1

    # 6. Load or create model
    print(f"[align] 初始化模型...")
    try:
        model_cfg_dict = cfg_dict.get("model", {}).copy()
        checkpoint_path = model_cfg_dict.pop("name", None)  # Extract checkpoint path if present
        model_config = ModelConfig(**model_cfg_dict)
        from llm_foundry.common.model import create_model

        model = create_model(model_config, use_flash_attn=False)
        model = model.to(device)

        # Load checkpoint if specified
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"[align] 加载 checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            # Handle different checkpoint formats
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("[align] Checkpoint 加载成功")
    except Exception as e:
        print(f"[错误] 模型初始化失败: {e}", file=sys.stderr)
        return 1

    # 7. Create optimizer
    method_cfg = cfg_dict.get(method, {})
    lr = float(method_cfg.get("learning_rate", 1e-5))
    from llm_foundry.common.optimizer import create_optimizer

    optimizer = create_optimizer(model.parameters(), lr=lr)

    # 8. Create output directory
    import datetime

    output_dir = cfg_dict.get("output", {}).get("dir", "./outputs/align")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, method, timestamp)
    os.makedirs(run_dir, exist_ok=True)
    print(f"[align] 输出目录: {run_dir}")

    # 9. Dispatch to appropriate trainer
    if method == "sft":
        print("[align] 开始 SFT 训练...")
        try:
            from llm_foundry.stage5_align import SFTTrainer, SFTDataset, collate_fn

            # Load dataset
            data_path = method_cfg.get("data_path", "data/sft_train.jsonl")
            eval_data_path = method_cfg.get("eval_data_path", "data/sft_eval.jsonl")
            max_length = method_cfg.get("max_length", 256)

            print(f"[align] 加载训练数据: {data_path}")
            pad_token_id = 0  # Use 0 as pad token for BPE tokenizer
            dataset = SFTDataset(data_path, tokenizer, max_length=max_length, pad_token_id=pad_token_id)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=method_cfg.get("batch_size", 16),
                shuffle=True,
                collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id),
            )

            # Load eval dataset if exists
            eval_dataloader = None
            if os.path.exists(eval_data_path):
                print(f"[align] 加载验证数据: {eval_data_path}")
                eval_dataset = SFTDataset(eval_data_path, tokenizer, max_length=max_length, pad_token_id=pad_token_id)
                eval_dataloader = torch.utils.data.DataLoader(
                    eval_dataset,
                    batch_size=method_cfg.get("batch_size", 16),
                    shuffle=False,
                    collate_fn=lambda batch: collate_fn(batch, pad_token_id=pad_token_id),
                )

            # Update method_cfg with pad_token_id
            method_cfg_with_pad = method_cfg.copy()
            method_cfg_with_pad["pad_token_id"] = pad_token_id

            trainer = SFTTrainer(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                device=str(device),
                config=method_cfg_with_pad,
            )

            # Train for num_epochs
            num_epochs = method_cfg.get("num_epochs", 3)
            grad_accum_steps = method_cfg.get("gradient_accumulation_steps", 1)
            save_steps = method_cfg.get("save_steps", 5)
            eval_steps = method_cfg.get("eval_steps", 5)

            for epoch in range(num_epochs):
                print(f"\n[align] Epoch {epoch + 1}/{num_epochs}")
                metrics = trainer.train_epoch(
                    dataloader,
                    gradient_accumulation_steps=grad_accum_steps,
                )
                print(f"[align] Epoch {epoch + 1} 完成，loss: {metrics['loss']:.4f}")

                # Save checkpoint
                if (epoch + 1) % save_steps == 0 or epoch == num_epochs - 1:
                    ckpt_path = os.path.join(run_dir, f"checkpoint_epoch_{epoch + 1:03d}.pt")
                    trainer.save_checkpoint(ckpt_path)
                    print(f"[align] Checkpoint 保存至: {ckpt_path}")

                # Evaluate
                if eval_dataloader and ((epoch + 1) % eval_steps == 0 or epoch == num_epochs - 1):
                    eval_metrics = trainer.evaluate(eval_dataloader)
                    print(f"[align] Eval loss: {eval_metrics['eval_loss']:.4f}")

            # Save final model
            model_path = os.path.join(run_dir, "final_model.pt")
            trainer.save_checkpoint(model_path)
            print(f"[align] SFT 模型已保存至: {model_path}")

        except ImportError as e:
            print(f"[错误] SFTTrainer 未实现: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"[错误] SFT 训练失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    elif method == "dpo":
        print("[align] 开始 DPO 训练...")
        try:
            from llm_foundry.stage5_align import DPOTrainer, DPODataset

            # Load reference model
            ref_model_path = method_cfg.get("ref_model_path", "checkpoints/sft_final")
            print(f"[align] 加载参考模型: {ref_model_path}")
            ref_model = create_model(model_config, use_flash_attn=False)
            ref_model = ref_model.to(device)
            ref_model.eval()

            # Load dataset
            data_path = method_cfg.get("data_path", "data/dpo_train.jsonl")
            print(f"[align] 加载 DPO 数据: {data_path}")
            dataset = DPODataset(data_path, tokenizer)
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=method_cfg.get("batch_size", 8),
                shuffle=True,
            )

            trainer = DPOTrainer(
                model=model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                beta=method_cfg.get("beta", 0.1),
                device=str(device),
                config=method_cfg,
            )

            # Train
            trainer.train(dataloader, output_dir=run_dir)

            # Save model
            model_path = os.path.join(run_dir, "final_model.pt")
            trainer.save_checkpoint(model_path)
            print(f"[align] DPO 模型已保存至: {model_path}")

        except ImportError as e:
            print(f"[错误] DPOTrainer 未实现: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"[错误] DPO 训练失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    elif method == "grpo":
        print("[align] 开始 GRPO 训练...")
        try:
            from llm_foundry.stage5_align import GRPOTrainer
            from llm_foundry.backends.inference import get_inference_backend, GenerationConfig

            # Get model checkpoint path if exists
            model_checkpoint = cfg_dict.get("model", {}).get("checkpoint_path")

            # Create inference backend for rollout
            inference_backend = get_inference_backend(
                backend_type=cfg_dict.get("inference", {}).get("backend", "auto"),
                model_name_or_path=model_checkpoint if model_checkpoint and os.path.exists(model_checkpoint) else None,
                model=model,
                tokenizer=tokenizer,
                device=str(device),
            )

            # Load reference model (frozen, for KL penalty)
            ref_model_path = method_cfg.get("ref_model_path", "checkpoints/sft_final")
            print(f"[align] 加载参考模型: {ref_model_path}")
            ref_model = create_model(model_config, use_flash_attn=False)
            ref_model = ref_model.to(device)
            ref_model.eval()

            # Define reward function (or load from config)
            reward_fn_name = method_cfg.get("reward_fn", "default")
            if reward_fn_name == "default":

                def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
                    """Placeholder reward function."""
                    return {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}
            else:
                # Try to load custom reward function
                try:
                    import importlib

                    module_path, fn_name = reward_fn_name.rsplit(":", 1)
                    module = importlib.import_module(module_path)
                    reward_fn = getattr(module, fn_name)
                except Exception:
                    print(f"[警告] 无法加载奖励函数: {reward_fn_name}，使用默认函数")

                    def reward_fn(response: str, ground_truth: str) -> dict[str, float]:
                        return {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}

            # Load dataset
            data_path = method_cfg.get("data_path", "data/grpo_train.jsonl")
            print(f"[align] 加载 GRPO 数据: {data_path}")
            # GRPO uses simple prompt dataset
            from llm_foundry.stage5_align import PromptDataset

            dataset = PromptDataset(data_path)

            trainer = GRPOTrainer(
                model=model,
                ref_model=ref_model,
                optimizer=optimizer,
                inference_backend=inference_backend,
                reward_fn=reward_fn,
                group_size=method_cfg.get("group_size", 8),
                device=str(device),
                config=method_cfg,
            )

            # Train
            trainer.train(dataset, output_dir=run_dir)

            # Save model
            model_path = os.path.join(run_dir, "final_model.pt")
            trainer.save_checkpoint(model_path)
            print(f"[align] GRPO 模型已保存至: {model_path}")

        except ImportError as e:
            print(f"[错误] GRPOTrainer 未实现: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"[错误] GRPO 训练失败: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    print(f"[align] {method.upper()} 训练完成！")
    print(f"[align] 最终结果保存至: {run_dir}")
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="llm-foundry",
        description="LLM Foundry Simulator - Unified CLI for LLM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py env                    # Check environment
  python run.py datagen --config configs/datagen.yaml
  python run.py tokenize --config configs/tokenize.yaml
  python run.py train --config configs/train.yaml
  python run.py train --config configs/train.yaml --data data/train.npy
  python run.py train --config configs/train.yaml --set training.lr=1e-4
  torchrun --nproc_per_node=2 run.py train --config configs/train.yaml
  python run.py scaling --config configs/scaling.yaml
  python run.py data --config configs/data.yaml
  python run.py align --config configs/align.yaml
        """,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # env command
    env_parser = subparsers.add_parser(
        "env",
        help="Check hardware environment and dependencies",
    )
    env_parser.set_defaults(func=cmd_env)

    # datagen command
    datagen_parser = subparsers.add_parser(
        "datagen",
        help="Generate SFT/GRPO training data (Plan 0)",
    )
    datagen_parser.add_argument(
        "--config",
        default="configs/datagen.yaml",
        help="Data generation config file path",
    )
    datagen_parser.set_defaults(func=cmd_datagen)

    # tokenize command
    tokenize_parser = subparsers.add_parser(
        "tokenize",
        help="Tokenize data (Plan 2)",
    )
    tokenize_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to tokenization config YAML",
    )
    tokenize_parser.set_defaults(func=cmd_tokenize)

    # train command
    train_parser = subparsers.add_parser(
        "train",
        help="Train model (Plan 3)",
    )
    train_parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML (default: configs/train.yaml)",
    )
    train_parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Override training data path",
    )
    train_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Override output base directory",
    )
    train_parser.add_argument(
        "--flash-attn",
        action="store_true",
        help="Force enable Flash Attention",
    )
    train_parser.add_argument(
        "--ddp",
        action="store_true",
        help="Enable DDP mode (use with torchrun)",
    )
    train_parser.add_argument(
        "--set",
        action="append",
        help="Override config values (e.g., --set training.lr=1e-4)",
    )
    train_parser.set_defaults(func=cmd_train)

    # scaling command
    scaling_parser = subparsers.add_parser(
        "scaling",
        help="Run scaling law experiments (Plan 4)",
    )
    scaling_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to scaling config YAML",
    )
    scaling_parser.set_defaults(func=cmd_scaling)

    # data command
    data_parser = subparsers.add_parser(
        "data",
        help="Run data quality pipeline (Plan 5)",
    )
    data_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to data pipeline config YAML",
    )
    data_parser.set_defaults(func=cmd_data)

    # align command
    align_parser = subparsers.add_parser(
        "align",
        help="Run alignment training (SFT, DPO, RLHF) (Plan 6)",
    )
    align_parser.add_argument(
        "--config",
        type=str,
        default="configs/align.yaml",
        help="Path to alignment config YAML",
    )
    align_parser.add_argument(
        "--method",
        type=str,
        choices=["sft", "dpo", "grpo"],
        default=None,
        help="Alignment method (overrides config)",
    )
    align_parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Base model path or name (overrides config)",
    )
    align_parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Data file path (overrides config)",
    )
    align_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    align_parser.add_argument(
        "--hf",
        action="store_true",
        help="Use HuggingFace implementation (for Stage 5)",
    )
    align_parser.set_defaults(func=cmd_align)

    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.command is None:
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
