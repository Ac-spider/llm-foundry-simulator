"""
llm_foundry/stage3_scaling/scaling.py — Scaling Laws Analyzer

整合 CS336 Assignment 3 两个模块的缩放律分析逻辑：
    - chinchilla_isoflops_scaling.py  → analyze_isoflops()（IsoFLOPs 幂律拟合）
    - chinchilla_scaling_laws_fitting.py → fit_chinchilla()（Chinchilla L-BFGS-B 拟合）

核心公式：
    Chinchilla:  L(N, D) = E + A/N^α + B/D^β
    IsoFLOPs:    N_opt(C) = A_N * C^a,  D_opt(C) = B_D * C^b
    计算约束:    C = 6 * N * D（前向 2ND + 反向 4ND FLOPs）

接口规格：
    analyzer = ScalingAnalyzer(cfg)
    result = analyzer.run(input_data)  # input_data 格式见 run() docstring
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from types import SimpleNamespace
from typing import Any

import matplotlib
matplotlib.use("Agg")  # 无显示器环境（CLI / headless Linux），必须在 pyplot import 之前设置
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize

from llm_foundry.common.hashing import compute_config_hash


def _to_dict(obj: Any) -> Any:
    """递归地将 SimpleNamespace 转换为 dict。"""
    if isinstance(obj, SimpleNamespace):
        return {k: _to_dict(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, list):
        return [_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    return obj


# ─────────────────────────────────────────
# IsoFLOPs 幂律拟合辅助函数
# 来源：chinchilla_isoflops_scaling.py
# ─────────────────────────────────────────

def _linear_log_space_model(log_C: np.ndarray, log_factor: float, exponent: float) -> np.ndarray:
    """
    对数空间中的线性方程，用于幂律拟合（scipy.optimize.curve_fit）。

    原始幂律 y = factor * C^exponent 取对数后变为：
        log(y) = exponent * log(C) + log(factor)

    参数：
        log_C (np.ndarray): 计算预算的自然对数，shape: (num_budgets,)
        log_factor (float): 幂律比例系数的自然对数 log(A) 或 log(B)
        exponent (float): 幂律指数 a 或 b

    返回：
        np.ndarray: 预测的 log(y)，shape: (num_budgets,)
    """
    return exponent * log_C + log_factor


def _canonical_power_law(C: np.ndarray, factor: float, exponent: float) -> np.ndarray:
    """
    标准幂律函数 y = factor * C^exponent，用于外推预测和可视化。

    参数：
        C (float 或 np.ndarray): 计算预算（FLOPs）
        factor (float): 幂律比例系数（exp(log_factor) 还原）
        exponent (float): 幂律指数

    返回：
        float 或 np.ndarray: 预测值，shape 与 C 相同
    """
    return factor * (C ** exponent)


# ─────────────────────────────────────────
# Chinchilla 损失函数相关（纯数值计算）
# 来源：chinchilla_scaling_laws_fitting.py（去除 API 客户端部分）
# ─────────────────────────────────────────

def _parametric_loss_hypothesis(params: tuple, N_space: np.ndarray, D_space: np.ndarray) -> np.ndarray:
    """
    Chinchilla 损失函数 L(N, D) = E + A/N^α + B/D^β。

    三项的物理含义：
        E：不可约损失（数据集固有熵），无法通过扩大模型或数据消除
        A/N^α：容量惩罚项，参数量越大惩罚越小
        B/D^β：数据惩罚项，token 越多惩罚越小

    参数：
        params (tuple): (E, A, B, alpha, beta)
        N_space (np.ndarray): 模型参数量，shape: (num_samples,)
        D_space (np.ndarray): 训练 token 数，shape: (num_samples,)

    返回：
        np.ndarray: 预测损失，shape: (num_samples,)
    """
    E, A, B, alpha, beta = params
    epsilon = 1e-12  # 防除零保护
    penalty_capacity = A / (N_space ** alpha + epsilon)
    penalty_data = B / (D_space ** beta + epsilon)
    return E + penalty_capacity + penalty_data


def _l_bfgs_b_objective(params: tuple, N_actual: np.ndarray, D_actual: np.ndarray, L_actual: np.ndarray) -> float:
    """
    L-BFGS-B 优化目标函数：对数空间均方残差（log-space MSE）。

    选用对数空间残差的原因：等权处理各量级样本，避免大损失值主导拟合。
        objective = mean( (log(L_pred) - log(L_actual))^2 )

    参数：
        params (tuple): (E, A, B, alpha, beta)
        N_actual (np.ndarray): 实验参数量，shape: (num_samples,)
        D_actual (np.ndarray): 实验 token 数，shape: (num_samples,)
        L_actual (np.ndarray): 实测验证损失，shape: (num_samples,)

    返回：
        float: 对数空间均方残差（标量），L-BFGS-B 最小化此目标
    """
    L_predicted = _parametric_loss_hypothesis(params, N_actual, D_actual)
    log_diff = np.log(L_predicted) - np.log(L_actual)
    return float(np.mean(log_diff ** 2))


# ─────────────────────────────────────────
# ScalingAnalyzer — 主分析类
# ─────────────────────────────────────────

class ScalingAnalyzer:
    """
    缩放律分析器，整合 IsoFLOPs 幂律拟合与 Chinchilla L(N,D) 曲面拟合。

    使用方式：
        cfg = {
            "scaling": {
                "experiments_file": "data/scaling_experiments.json",
                "compute_budgets": [1e18, 1e19, 1e20, 1e21]
            },
            "output": {
                "base_dir": "results/",
                "save_plots": True,
                "plot_format": "png"
            }
        }
        analyzer = ScalingAnalyzer(cfg)
        result = analyzer.run(input_data)
    """

    def __init__(self, cfg: dict):
        """
        初始化分析器。

        参数：
            cfg (dict): 由 configs/scaling.yaml 解析而来的配置字典，
                        包含 scaling 节和 output 节。
        """
        # 将 SimpleNamespace 递归转换为 dict
        self.cfg = _to_dict(cfg)
        self._scaling_cfg = self.cfg.get("scaling", {})
        self._output_cfg = self.cfg.get("output", {})

    def run(self, input_data: dict) -> dict:
        """
        执行 IsoFLOPs + Chinchilla 拟合分析的完整流水线。

        参数：
            input_data (dict): 实验数据，格式：
                {
                    "experiments": [
                        {"n_params": 1e7, "n_tokens": 1e9, "loss": 3.5},
                        ...
                    ]
                }
                字段含义：
                    n_params: 模型非嵌入参数量（等同于论文中的 N）
                    n_tokens: 训练 token 数（等同于论文中的 D）
                    loss:     验证损失（等同于论文中的 L）
                计算预算 C 由 C = 6 * n_params * n_tokens 自动推算。

        返回：
            dict: {
                "chinchilla": {"alpha": ..., "beta": ..., "A": ..., "B": ..., "E": ...},
                "isoflops": {
                    "optimal_model_size_at_compute": [...],
                    "optimal_tokens_at_compute": [...]
                },
                "plots_dir": "results/{hash}/scaling_plots/"
            }
        """
        experiments = input_data["experiments"] if isinstance(input_data, dict) else input_data

        # 计算每条实验的计算预算 C = 6 * N * D
        for exp in experiments:
            exp["compute"] = 6.0 * exp["n_params"] * exp["n_tokens"]

        # 执行两条分析流水线
        chinchilla_result = self.fit_chinchilla(experiments)
        isoflops_result = self.analyze_isoflops(experiments)

        # 确定输出路径
        base_dir = self._output_cfg.get("base_dir", "results/")
        run_hash = self._make_run_hash(input_data)
        run_dir = os.path.join(base_dir, run_hash)
        plots_dir = os.path.join(run_dir, "scaling_plots")
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # 可视化
        if self._output_cfg.get("save_plots", True):
            fmt = self._output_cfg.get("plot_format", "png")
            self._plot_isoflops(experiments, isoflops_result, plots_dir, fmt)
            self._plot_chinchilla(experiments, chinchilla_result, plots_dir, fmt)

        # 组装输出结果
        compute_budgets_raw = self._scaling_cfg.get("compute_budgets", [1e18, 1e19, 1e20, 1e21])
        # 确保所有预算值都是数值类型（YAML可能解析为字符串）
        compute_budgets = [float(C) for C in compute_budgets_raw]
        isoflops_optimal = [
            {
                "compute": C,
                "n_params": _canonical_power_law(C, isoflops_result["factor_N"], isoflops_result["exponent_N"]),
                "n_tokens": _canonical_power_law(C, isoflops_result["factor_D"], isoflops_result["exponent_D"]),
            }
            for C in compute_budgets
        ]

        result = {
            "chinchilla": chinchilla_result,
            "isoflops": {
                "optimal_model_size_at_compute": [r["n_params"] for r in isoflops_optimal],
                "optimal_tokens_at_compute": [r["n_tokens"] for r in isoflops_optimal],
            },
            "isoflops_optimal": isoflops_optimal,
            "plots_dir": plots_dir,
        }

        # 写入 scaling_params.json
        params_path = os.path.join(run_dir, "scaling_params.json")
        self._write_params_json(result, params_path)

        print(f"[ScalingAnalyzer] 分析完成，结果写入: {params_path}")
        print(f"[ScalingAnalyzer] 图表目录: {plots_dir}")

        return result

    def fit_chinchilla(self, data: list[dict]) -> dict:
        """
        拟合 Chinchilla 损失函数 L(N, D) = E + A/N^α + B/D^β。

        使用 L-BFGS-B 算法在对数空间中最小化预测损失与实测损失的均方差，
        求解五个参数 (E, A, B, alpha, beta)。

        参数：
            data (list[dict]): 实验数据列表，每条记录包含 n_params、n_tokens、loss、compute。

        返回：
            dict: {
                "alpha": float,   # 容量衰减指数（原论文约 0.34）
                "beta": float,    # 数据衰减指数（原论文约 0.28）
                "A": float,       # 容量惩罚系数（原论文约 406.4）
                "B": float,       # 数据惩罚系数（原论文约 410.7）
                "E": float        # 不可约损失（原论文约 1.69）
            }
        """
        N_arr = np.array([d["n_params"] for d in data])
        D_arr = np.array([d["n_tokens"] for d in data])
        L_arr = np.array([d["loss"] for d in data])

        # 初始猜测值：基于 Chinchilla 原论文拟合结果附近初始化
        initial_guess = [1.5, 450.0, 2100.0, 0.35, 0.35]  # [E, A, B, alpha, beta]

        # 参数边界约束（保证物理意义）
        bounds = [
            (0.1, 5.0),       # E：不可约损失，(0, ~5)
            (1.0, 10000.0),   # A：容量惩罚系数，正数
            (1.0, 10000.0),   # B：数据惩罚系数，正数
            (0.01, 1.0),      # alpha：容量衰减指数，(0, 1)
            (0.01, 1.0),      # beta：数据衰减指数，(0, 1)
        ]

        result = minimize(
            _l_bfgs_b_objective,
            initial_guess,
            args=(N_arr, D_arr, L_arr),
            method="L-BFGS-B",
            bounds=bounds,
        )

        E_opt, A_opt, B_opt, alpha_opt, beta_opt = result.x

        return {
            "alpha": float(alpha_opt),
            "beta": float(beta_opt),
            "A": float(A_opt),
            "B": float(B_opt),
            "E": float(E_opt),
        }

    def analyze_isoflops(self, data: list[dict]) -> dict:
        """
        分析 IsoFLOPs 曲线，拟合 N_opt(C) 与 D_opt(C) 的幂律缩放关系。

        IsoFLOPs 核心思想：固定计算预算 C，从多个 (N, D) 组合中选取
        验证损失最小的配置，再对多个预算截面的最优点做幂律拟合。

        拟合方式：对数空间线性化（log-space linear regression），
        避免非线性优化器初值敏感发散。

        参数：
            data (list[dict]): 实验数据列表，每条记录包含 n_params、n_tokens、loss、compute。

        返回：
            dict: {
                "factor_N": float,    # N_opt 幂律比例系数 A_N
                "exponent_N": float,  # N_opt 幂律指数 a
                "factor_D": float,    # D_opt 幂律比例系数 B_D
                "exponent_D": float,  # D_opt 幂律指数 b
                "optimal_points": [   # 各预算截面的最优点（原始观测）
                    {"compute": C, "n_params": N_opt, "n_tokens": D_opt, "loss": L}, ...
                ]
            }
        """
        # 按计算预算分组构建 IsoFLOPs 截面
        # 使用 round 到最近的整数数量级避免浮点精度导致的分组错误
        budget_groups: dict[float, list] = defaultdict(list)
        for d in data:
            # 将 compute 舍入到最近的整数幂次（用于截面分组）
            C = d["compute"]
            # 对数空间取整到最近的整数幂次（避免过度合并，保留更多数据点）
            C_key = round(np.log10(C), 0)  # 精度：1 个数量级
            budget_groups[C_key].append(d)

        # 若只有 1 个截面，IsoFLOPs 拟合退化（至少需要 2 个截面）
        if len(budget_groups) < 2:
            raise ValueError(
                f"IsoFLOPs analysis requires at least 2 compute budget cross-sections, "
                f"but got {len(budget_groups)}. "
                f"Please provide more diverse experimental data covering different compute budgets."
            )
        optimal_C, optimal_N, optimal_D = [], [], []
        optimal_points = []

        for C_key in sorted(budget_groups.keys()):
            cross_section = budget_groups[C_key]
            # 从截面中选取验证损失最小的配置（IsoFLOPs Approach 1）
            best = min(cross_section, key=lambda d: d["loss"])
            C_i = best["compute"]
            N_i = best["n_params"]
            D_i = best["n_tokens"]

            optimal_C.append(C_i)
            optimal_N.append(N_i)
            optimal_D.append(D_i)
            optimal_points.append({
                "compute": C_i,
                "n_params": N_i,
                "n_tokens": D_i,
                "loss": best["loss"],
            })

        C_arr = np.array(optimal_C)
        N_arr = np.array(optimal_N)
        D_arr = np.array(optimal_D)

        log_C = np.log(C_arr)
        log_N = np.log(N_arr)
        log_D = np.log(D_arr)

        # 对数空间幂律拟合：log(N_opt) = a * log(C) + log(A_N)
        popt_N, _ = curve_fit(_linear_log_space_model, log_C, log_N)
        log_A_N, exponent_N = popt_N
        factor_N = float(np.exp(log_A_N))

        # 对数空间幂律拟合：log(D_opt) = b * log(C) + log(B_D)
        popt_D, _ = curve_fit(_linear_log_space_model, log_C, log_D)
        log_B_D, exponent_D = popt_D
        factor_D = float(np.exp(log_B_D))

        print(f"[IsoFLOPs] a + b = {exponent_N + exponent_D:.4f}（理论值应接近 1.0）")

        return {
            "factor_N": factor_N,
            "exponent_N": float(exponent_N),
            "factor_D": factor_D,
            "exponent_D": float(exponent_D),
            "optimal_points": optimal_points,
        }

    # ─────────────────────────────────────────
    # 私有方法
    # ─────────────────────────────────────────

    @staticmethod
    def _make_run_hash(input_data: dict) -> str:
        """根据输入数据生成确定性哈希（8位十六进制），用于命名输出目录。

        使用 common/hashing.py 的 compute_config_hash 保证跨 Plan hash 算法一致。
        """
        return compute_config_hash(input_data)[:8]

    @staticmethod
    def _write_params_json(result: dict, path: str) -> None:
        """将分析结果写入 scaling_params.json，numpy float 转为 Python float。"""
        def _to_python(obj: Any) -> Any:
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def _convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return _to_python(obj)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(_convert(result), f, indent=4, ensure_ascii=False)

    def _plot_isoflops(
        self,
        data: list[dict],
        isoflops_result: dict,
        plots_dir: str,
        fmt: str,
    ) -> None:
        """
        绘制 IsoFLOPs 缩放律图（双子图：N_opt vs C，D_opt vs C），
        保存至 plots_dir/isoflops_scaling.{fmt}。
        """
        optimal_points = isoflops_result["optimal_points"]
        C_obs = np.array([p["compute"] for p in optimal_points])
        N_obs = np.array([p["n_params"] for p in optimal_points])
        D_obs = np.array([p["n_tokens"] for p in optimal_points])

        C_cont = np.logspace(np.log10(max(C_obs.min(), 1.0)), np.log10(C_obs.max() * 100), 300)
        N_cont = _canonical_power_law(C_cont, isoflops_result["factor_N"], isoflops_result["exponent_N"])
        D_cont = _canonical_power_law(C_cont, isoflops_result["factor_D"], isoflops_result["exponent_D"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        axes[0].scatter(C_obs, N_obs, color="#1f77b4", s=60, label="Empirical N_opt")
        axes[0].plot(C_cont, N_cont, color="#d62728", linestyle="--", linewidth=2,
                     label=f"N = {isoflops_result['factor_N']:.2e} * C^{isoflops_result['exponent_N']:.3f}")
        axes[0].set_xscale("log")
        axes[0].set_yscale("log")
        axes[0].set_xlabel("Compute Budget C (FLOPs)")
        axes[0].set_ylabel("Optimal Params N")
        axes[0].set_title("IsoFLOPs: N_opt vs Compute")
        axes[0].legend()
        axes[0].grid(True, which="both", linestyle="--", alpha=0.5)

        axes[1].scatter(C_obs, D_obs, color="#2ca02c", s=60, label="Empirical D_opt")
        axes[1].plot(C_cont, D_cont, color="#ff7f0e", linestyle="--", linewidth=2,
                     label=f"D = {isoflops_result['factor_D']:.2e} * C^{isoflops_result['exponent_D']:.3f}")
        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].set_xlabel("Compute Budget C (FLOPs)")
        axes[1].set_ylabel("Optimal Tokens D")
        axes[1].set_title("IsoFLOPs: D_opt vs Compute")
        axes[1].legend()
        axes[1].grid(True, which="both", linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"isoflops_scaling.{fmt}")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[ScalingAnalyzer] IsoFLOPs 图表已保存: {out_path}")

    def _plot_chinchilla(
        self,
        data: list[dict],
        chinchilla_result: dict,
        plots_dir: str,
        fmt: str,
    ) -> None:
        """
        绘制 Chinchilla 损失曲面拟合图（预测损失 vs 实测损失），
        保存至 plots_dir/chinchilla_fit.{fmt}。
        """
        N_arr = np.array([d["n_params"] for d in data])
        D_arr = np.array([d["n_tokens"] for d in data])
        L_arr = np.array([d["loss"] for d in data])

        params = (
            chinchilla_result["E"],
            chinchilla_result["A"],
            chinchilla_result["B"],
            chinchilla_result["alpha"],
            chinchilla_result["beta"],
        )
        L_pred = _parametric_loss_hypothesis(params, N_arr, D_arr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(L_arr, L_pred, color="#9467bd", s=60, alpha=0.8, label="Experiments")
        lims = [min(L_arr.min(), L_pred.min()) * 0.95, max(L_arr.max(), L_pred.max()) * 1.05]
        ax.plot(lims, lims, "k--", linewidth=1.5, label="Perfect fit")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel("Actual Loss L")
        ax.set_ylabel("Predicted Loss L(N,D)")
        ax.set_title(
            f"Chinchilla Fit\n"
            f"E={chinchilla_result['E']:.3f}  A={chinchilla_result['A']:.1f}  "
            f"α={chinchilla_result['alpha']:.3f}  B={chinchilla_result['B']:.1f}  "
            f"β={chinchilla_result['beta']:.3f}"
        )
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        out_path = os.path.join(plots_dir, f"chinchilla_fit.{fmt}")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[ScalingAnalyzer] Chinchilla 图表已保存: {out_path}")
