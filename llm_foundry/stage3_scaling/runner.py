"""
llm_foundry/stage3_scaling/runner.py — Scaling Experiment Runner

负责管理和执行缩放实验，包括：
- 根据FLOPs预算自动计算模型参数量N和数据量D
- 集成trainer进行训练
- 记录训练结果到CSV
- 支持实验断点续跑

核心公式：
    C = 6 * N * D（前向 2ND + 反向 4ND FLOPs）
    N_opt(C) = A_N * C^a  (IsoFLOPs 幂律)
    D_opt(C) = B_D * C^b

接口规格：
    runner = ScalingRunner(cfg)
    runner.run_experiments()  # 批量运行多个实验
"""
from __future__ import annotations

import csv
import json
import os
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from llm_foundry.common.model import ModelConfig
from llm_foundry.stage2_train.trainer import Trainer


# ─────────────────────────────────────────
# 实验配置与状态管理
# ─────────────────────────────────────────

@dataclass
class ScalingExperiment:
    """单个缩放实验的配置和状态。

    Attributes:
        exp_id: 实验唯一标识符
        compute_budget: 计算预算（FLOPs）
        n_params: 模型参数量（非嵌入参数）
        n_tokens: 训练token数
        d_model: 模型维度
        num_layers: Transformer层数
        num_heads: 注意力头数
        d_ff: FFN隐藏层维度
        batch_size: 训练批次大小
        learning_rate: 学习率
        status: 实验状态（pending/running/completed/failed）
        result: 实验结果（验证损失等）
        checkpoint_path: 检查点路径
    """
    exp_id: str
    compute_budget: float
    n_params: float
    n_tokens: float
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    batch_size: int = 32
    learning_rate: float = 3e-4
    status: str = "pending"
    result: dict = field(default_factory=dict)
    checkpoint_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "exp_id": self.exp_id,
            "compute_budget": self.compute_budget,
            "n_params": self.n_params,
            "n_tokens": self.n_tokens,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "status": self.status,
            "result": self.result,
            "checkpoint_path": self.checkpoint_path,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ScalingExperiment":
        """从字典创建。"""
        return cls(
            exp_id=d["exp_id"],
            compute_budget=d["compute_budget"],
            n_params=d["n_params"],
            n_tokens=d["n_tokens"],
            d_model=d.get("d_model", 512),
            num_layers=d.get("num_layers", 8),
            num_heads=d.get("num_heads", 8),
            d_ff=d.get("d_ff", 2048),
            batch_size=d.get("batch_size", 32),
            learning_rate=d.get("learning_rate", 3e-4),
            status=d.get("status", "pending"),
            result=d.get("result", {}),
            checkpoint_path=d.get("checkpoint_path", ""),
        )


# ─────────────────────────────────────────
# FLOPs 计算与模型配置生成
# ─────────────────────────────────────────

def compute_flops(n_params: float, n_tokens: float) -> float:
    """计算训练所需FLOPs。

    公式：C = 6 * N * D（前向 2ND + 反向 4ND）

    Args:
        n_params: 模型参数量
        n_tokens: 训练token数

    Returns:
        计算预算（FLOPs）
    """
    return 6.0 * n_params * n_tokens


def compute_non_embedding_params(d_model: int, num_layers: int) -> int:
    """计算Transformer模型的非嵌入参数量。

    基于 12Ld² 公式：
    - 注意力：4个投影矩阵（Q,K,V,O），每个 d×d = 4d²
    - FFN：两个矩阵，d→4d→d = 8d²
    - 每层合计：12d²
    - L层总计：12 * L * d²

    Args:
        d_model: 模型维度
        num_layers: Transformer层数

    Returns:
        非嵌入参数量
    """
    return 12 * num_layers * (d_model ** 2)


def derive_tokens_from_flops(compute_budget: float, n_params: float) -> float:
    """根据FLOPs预算和参数量反推训练token数。

    公式：D = C / (6 * N)

    Args:
        compute_budget: 计算预算（FLOPs）
        n_params: 模型参数量

    Returns:
        训练token数
    """
    return compute_budget / (6.0 * n_params)


def derive_params_from_flops(compute_budget: float, n_tokens: float) -> float:
    """根据FLOPs预算和token数反推模型参数量。

    公式：N = C / (6 * D)

    Args:
        compute_budget: 计算预算（FLOPs）
        n_tokens: 训练token数

    Returns:
        模型参数量
    """
    return compute_budget / (6.0 * n_tokens)


def generate_model_config_from_params(
    n_params: float,
    vocab_size: int = 50257,
    context_length: int = 1024,
    head_dim: int = 64,
) -> dict:
    """根据目标参数量生成模型配置。

    使用 12 * L * d² 公式反推 d_model 和 num_layers 的组合。
    优先保持 d_model 为 2 的幂次，调整 num_layers。

    Args:
        n_params: 目标非嵌入参数量
        vocab_size: 词表大小
        context_length: 上下文长度
        head_dim: 每个头的维度

    Returns:
        模型配置字典
    """
    # 常见 d_model 值（2的幂次）
    d_model_candidates = [256, 384, 512, 768, 1024, 1280, 1536, 2048]

    best_config = None
    best_error = float("inf")

    for d_model in d_model_candidates:
        # 计算达到目标参数量需要的层数
        # n_params = 12 * L * d²  =>  L = n_params / (12 * d²)
        num_layers = int(round(n_params / (12 * d_model ** 2)))

        # 限制层数在合理范围内
        num_layers = max(2, min(num_layers, 48))

        # 计算实际参数量
        actual_params = compute_non_embedding_params(d_model, num_layers)
        error = abs(actual_params - n_params) / n_params

        if error < best_error:
            best_error = error
            num_heads = max(2, d_model // head_dim)
            d_ff = 4 * d_model  # 标准FFN扩展比

            best_config = {
                "vocab_size": vocab_size,
                "context_length": context_length,
                "d_model": d_model,
                "num_layers": num_layers,
                "num_heads": num_heads,
                "d_ff": d_ff,
            }

    return best_config


def generate_experiment_matrix(
    compute_budgets: list[float],
    params_ratios: list[float] | None = None,
) -> list[dict]:
    """生成实验矩阵。

    对于每个计算预算，生成多个 (N, D) 组合，用于IsoFLOPs分析。

    Args:
        compute_budgets: 计算预算列表（FLOPs）
        params_ratios: 参数量比例列表（相对于最优值）

    Returns:
        实验配置字典列表
    """
    if params_ratios is None:
        # 默认：在 0.25x 到 4x 最优参数范围内采样
        params_ratios = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]

    experiments = []
    exp_id = 0

    for C in compute_budgets:
        for ratio in params_ratios:
            # 假设最优配置时 N ~ C^0.5, D ~ C^0.5
            # 这里使用启发式：N_opt ≈ sqrt(C / 6 / 20)  (假设D/N ≈ 20)
            n_opt = np.sqrt(C / 6 / 20)
            n_params = n_opt * ratio
            n_tokens = derive_tokens_from_flops(C, n_params)

            # 确保token数合理
            if n_tokens < 1000 or n_tokens > 1e13:
                continue

            exp_id += 1
            experiments.append({
                "exp_id": f"exp_{exp_id:03d}",
                "compute_budget": C,
                "n_params": n_params,
                "n_tokens": n_tokens,
                "ratio": ratio,
            })

    return experiments


# ─────────────────────────────────────────
# 实验运行器
# ─────────────────────────────────────────

class ScalingRunner:
    """缩放实验运行器，批量管理和执行多个缩放实验。

    支持：
    - 从配置自动生成实验矩阵
    - 断点续跑（通过状态文件）
    - 结果记录到CSV
    - 集成Trainer进行训练

    Example:
        cfg = {
            "compute_budgets": [1e17, 1e18, 1e19],
            "params_ratios": [0.5, 1.0, 2.0],
            "output_dir": "results/scaling",
            "data_path": "data/train.bin",
        }
        runner = ScalingRunner(cfg)
        results = runner.run_experiments()
    """

    def __init__(self, cfg: dict):
        """初始化运行器。

        Args:
            cfg: 配置字典，包含：
                - compute_budgets: 计算预算列表
                - params_ratios: 参数量比例列表（可选）
                - output_dir: 输出目录
                - data_path: 训练数据路径
                - training: 训练配置（max_steps, batch_size等）
        """
        self.cfg = cfg
        self.compute_budgets = cfg.get("compute_budgets", [1e17, 1e18, 1e19])
        self.params_ratios = cfg.get("params_ratios", [0.5, 1.0, 2.0])
        self.output_dir = cfg.get("output_dir", "results/scaling")
        self.data_path = cfg.get("data_path", "")
        self.training_cfg = cfg.get("training", {})

        # 状态文件路径
        self.state_path = os.path.join(self.output_dir, "runner_state.pkl")
        self.results_path = os.path.join(self.output_dir, "experiments.csv")

        # 实验列表
        self.experiments: list[ScalingExperiment] = []

        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)

    def _load_state(self) -> bool:
        """加载运行器状态（断点续跑）。

        Returns:
            是否成功加载状态
        """
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path, "rb") as f:
                    state = pickle.load(f)
                self.experiments = [ScalingExperiment.from_dict(e) for e in state["experiments"]]
                print(f"[ScalingRunner] 已加载状态，共 {len(self.experiments)} 个实验")
                return True
            except Exception as e:
                print(f"[ScalingRunner] 加载状态失败: {e}")
                return False
        return False

    def _save_state(self) -> None:
        """保存运行器状态。"""
        state = {
            "experiments": [e.to_dict() for e in self.experiments],
        }
        with open(self.state_path, "wb") as f:
            pickle.dump(state, f)

    def _save_results_csv(self) -> None:
        """保存实验结果到CSV。"""
        if not self.experiments:
            return

        fieldnames = [
            "exp_id", "compute_budget", "n_params", "n_tokens",
            "d_model", "num_layers", "num_heads", "d_ff",
            "batch_size", "learning_rate", "status", "loss",
        ]

        with open(self.results_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for exp in self.experiments:
                row = {
                    "exp_id": exp.exp_id,
                    "compute_budget": exp.compute_budget,
                    "n_params": exp.n_params,
                    "n_tokens": exp.n_tokens,
                    "d_model": exp.d_model,
                    "num_layers": exp.num_layers,
                    "num_heads": exp.num_heads,
                    "d_ff": exp.d_ff,
                    "batch_size": exp.batch_size,
                    "learning_rate": exp.learning_rate,
                    "status": exp.status,
                    "loss": exp.result.get("loss", ""),
                }
                writer.writerow(row)

        print(f"[ScalingRunner] 结果已保存到: {self.results_path}")

    def generate_experiments(self, force_regenerate: bool = False) -> list[ScalingExperiment]:
        """生成实验列表。

        Args:
            force_regenerate: 是否强制重新生成（忽略已有状态）

        Returns:
            实验列表
        """
        if not force_regenerate and self._load_state():
            # 检查是否有未完成的实验
            pending = [e for e in self.experiments if e.status == "pending"]
            if pending:
                print(f"[ScalingRunner] 发现 {len(pending)} 个待运行实验")
                return self.experiments
            else:
                print("[ScalingRunner] 所有实验已完成，重新生成...")

        # 生成实验矩阵
        matrix = generate_experiment_matrix(self.compute_budgets, self.params_ratios)

        self.experiments = []
        for item in matrix:
            # 根据参数量生成模型配置
            model_cfg = generate_model_config_from_params(item["n_params"])

            exp = ScalingExperiment(
                exp_id=item["exp_id"],
                compute_budget=item["compute_budget"],
                n_params=item["n_params"],
                n_tokens=item["n_tokens"],
                d_model=model_cfg["d_model"],
                num_layers=model_cfg["num_layers"],
                num_heads=model_cfg["num_heads"],
                d_ff=model_cfg["d_ff"],
                batch_size=self.training_cfg.get("batch_size", 32),
                learning_rate=self.training_cfg.get("learning_rate", 3e-4),
                status="pending",
            )
            self.experiments.append(exp)

        print(f"[ScalingRunner] 生成了 {len(self.experiments)} 个实验")
        self._save_state()
        return self.experiments

    def run_experiment(self, exp: ScalingExperiment) -> dict:
        """运行单个实验。

        Args:
            exp: 实验配置

        Returns:
            实验结果字典
        """
        print(f"\n[ScalingRunner] 开始实验 {exp.exp_id}")
        print(f"  计算预算: {exp.compute_budget:.2e} FLOPs")
        print(f"  参数量: {exp.n_params:.2e}")
        print(f"  Token数: {exp.n_tokens:.2e}")
        print(f"  模型: d_model={exp.d_model}, layers={exp.num_layers}, heads={exp.num_heads}")

        exp.status = "running"
        self._save_state()

        try:
            # 构建训练配置
            max_steps = self._estimate_steps(exp)

            train_cfg = {
                "model": {
                    "vocab_size": 50257,
                    "context_length": 1024,
                    "d_model": exp.d_model,
                    "num_layers": exp.num_layers,
                    "num_heads": exp.num_heads,
                    "d_ff": exp.d_ff,
                    "use_flash_attn": self.training_cfg.get("use_flash_attn", False),
                },
                "training": {
                    "data_path": self.data_path,
                    "max_steps": max_steps,
                    "batch_size": exp.batch_size,
                    "lr": exp.learning_rate,
                    "min_lr": exp.learning_rate * 0.1,
                    "weight_decay": self.training_cfg.get("weight_decay", 0.1),
                    "warmup_steps": max_steps // 10,
                    "grad_clip": 1.0,
                    "save_interval": max_steps,
                    "log_interval": max(1, max_steps // 20),
                    "device": self.training_cfg.get("device", "auto"),
                    "gradient_accumulation_steps": self.training_cfg.get("gradient_accumulation_steps", 1),
                },
                "output": {
                    "base_dir": os.path.join(self.output_dir, exp.exp_id),
                },
            }

            # 运行训练
            trainer = Trainer(train_cfg)
            trainer.train()

            # 读取训练结果（从metrics.jsonl最后一行）
            metrics_path = os.path.join(trainer.run_dir, "metrics.jsonl")
            final_loss = self._read_final_loss(metrics_path)

            exp.result = {
                "loss": final_loss,
                "run_dir": trainer.run_dir,
            }
            exp.status = "completed"
            exp.checkpoint_path = os.path.join(trainer.ckpt_dir, f"step_{max_steps:06d}.pt")

            print(f"[ScalingRunner] 实验 {exp.exp_id} 完成，最终损失: {final_loss:.4f}")

        except Exception as e:
            exp.status = "failed"
            exp.result = {"error": str(e)}
            print(f"[ScalingRunner] 实验 {exp.exp_id} 失败: {e}")

        self._save_state()
        return exp.result

    def _estimate_steps(self, exp: ScalingExperiment) -> int:
        """估算训练步数。

        根据token数、batch size和context length计算。

        Args:
            exp: 实验配置

        Returns:
            训练步数
        """
        context_length = 1024
        tokens_per_step = exp.batch_size * context_length
        steps = int(exp.n_tokens / tokens_per_step)

        # 限制最大步数
        max_steps = self.training_cfg.get("max_steps", 100000)
        return min(steps, max_steps)

    def _read_final_loss(self, metrics_path: str) -> float:
        """从metrics.jsonl读取最终损失。

        Args:
            metrics_path: metrics文件路径

        Returns:
            最终损失值
        """
        if not os.path.exists(metrics_path):
            return float("inf")

        final_loss = float("inf")
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                for line in f:
                    record = json.loads(line.strip())
                    if "loss" in record:
                        final_loss = record["loss"]
        except Exception:
            pass

        return final_loss

    def run_experiments(
        self,
        resume: bool = True,
        max_experiments: int | None = None,
    ) -> list[dict]:
        """批量运行实验。

        Args:
            resume: 是否从上次状态继续（断点续跑）
            max_experiments: 最大运行实验数（None表示运行所有待运行实验）

        Returns:
            实验结果列表
        """
        # 生成或加载实验列表
        if resume:
            self.generate_experiments(force_regenerate=False)
        else:
            self.generate_experiments(force_regenerate=True)

        # 筛选待运行的实验
        pending = [e for e in self.experiments if e.status == "pending"]

        if not pending:
            print("[ScalingRunner] 没有待运行的实验")
            return [e.to_dict() for e in self.experiments]

        if max_experiments:
            pending = pending[:max_experiments]

        print(f"\n[ScalingRunner] 准备运行 {len(pending)} 个实验")

        # 运行实验
        for i, exp in enumerate(pending):
            print(f"\n[ScalingRunner] 进度: {i+1}/{len(pending)}")
            self.run_experiment(exp)
            self._save_results_csv()

        print(f"\n[ScalingRunner] 所有实验完成，结果保存在: {self.output_dir}")
        return [e.to_dict() for e in self.experiments]

    def get_results_for_analysis(self) -> dict:
        """获取用于缩放律分析的结果数据。

        Returns:
            符合ScalingAnalyzer输入格式的数据字典
        """
        experiments = []
        for exp in self.experiments:
            if exp.status == "completed" and "loss" in exp.result:
                experiments.append({
                    "n_params": exp.n_params,
                    "n_tokens": exp.n_tokens,
                    "loss": exp.result["loss"],
                })

        return {"experiments": experiments}


# ─────────────────────────────────────────
# 便捷函数
# ─────────────────────────────────────────

def run_scaling_experiment(
    compute_budget: float,
    n_params: float | None = None,
    n_tokens: float | None = None,
    data_path: str = "",
    output_dir: str = "results/scaling",
    training_cfg: dict | None = None,
) -> dict:
    """运行单个缩放实验（便捷函数）。

    根据FLOPs预算自动计算N和D（如果未提供）。

    Args:
        compute_budget: 计算预算（FLOPs）
        n_params: 模型参数量（可选，默认根据预算估算）
        n_tokens: 训练token数（可选，默认根据预算估算）
        data_path: 训练数据路径
        output_dir: 输出目录
        training_cfg: 训练配置

    Returns:
        实验结果字典
    """
    # 如果未提供N或D，使用启发式估算（假设D/N ≈ 20）
    if n_params is None and n_tokens is None:
        n_params = np.sqrt(compute_budget / 6 / 20)
        n_tokens = derive_tokens_from_flops(compute_budget, n_params)
    elif n_params is None:
        n_params = derive_params_from_flops(compute_budget, n_tokens)
    elif n_tokens is None:
        n_tokens = derive_tokens_from_flops(compute_budget, n_params)

    # 生成模型配置
    model_cfg = generate_model_config_from_params(n_params)

    # 创建实验
    exp = ScalingExperiment(
        exp_id="single_exp",
        compute_budget=compute_budget,
        n_params=n_params,
        n_tokens=n_tokens,
        d_model=model_cfg["d_model"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
    )

    # 创建运行器并运行
    cfg = {
        "output_dir": output_dir,
        "data_path": data_path,
        "training": training_cfg or {},
    }
    runner = ScalingRunner(cfg)
    runner.experiments = [exp]

    result = runner.run_experiment(exp)
    return result


def estimate_optimal_nd(
    compute_budget: float,
    alpha: float = 0.34,
    beta: float = 0.28,
    A: float = 406.4,
    B: float = 410.7,
) -> tuple[float, float]:
    """根据Chinchilla公式估算最优N和D。

    使用Chinchilla损失函数 L(N,D) = E + A/N^α + B/D^β
    在约束 C = 6ND 下最小化损失。

    Args:
        compute_budget: 计算预算（FLOPs）
        alpha: 容量衰减指数
        beta: 数据衰减指数
        A: 容量惩罚系数
        B: 数据惩罚系数

    Returns:
        (n_opt, d_opt) 最优参数量和token数
    """
    # 解析解：在 C = 6ND 约束下，最优比例满足
    # (N^α / A) / α = (D^β / B) / β
    # 近似解：N_opt ∝ C^(β/(α+β)), D_opt ∝ C^(α/(α+β))

    # 使用Chinchilla论文的近似比例
    # 对于典型值 α≈0.34, β≈0.28
    # N_opt ≈ 0.1 * C^0.5, D_opt ≈ 1.67 * C^0.5

    C = compute_budget
    n_opt = 0.1 * (C ** 0.5)
    d_opt = C / (6 * n_opt)

    return n_opt, d_opt
