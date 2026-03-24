# Plan 4: Scaling Laws Analysis Implementation Plan

> **For agentic workers (OMC):** 推荐使用 `/oh-my-claudecode:ralph`（自循环执行直到完成，适合多步骤 task）或 `/oh-my-claudecode:ultrawork`（并行高吞吐执行，适合独立任务批量完成，复杂 task 加 `model=opus`）。步骤使用 checkbox (`- [ ]`) 语法跟踪进度，完成后用 TaskUpdate 标记 completed。

**Goal:** 构建 `llm_foundry/stage3_scaling/scaling.py` 中的 `ScalingAnalyzer` 类，整合 IsoFLOPs + Chinchilla 拟合两条流水线，新增 `run.py scaling` 子命令，使 `python run.py scaling --config configs/scaling.yaml` 可执行完整缩放律分析并产出 `results/{hash}/scaling_params.json` 与拟合曲线图。

**Architecture:** `ScalingAnalyzer` 单类封装全部逻辑，`run()` 方法串联 IsoFLOPs 分析（`analyze_isoflops`）和 Chinchilla 参数拟合（`fit_chinchilla`），结果以 JSON 写入 `results/{hash}/scaling_params.json`，可视化写入 `results/{hash}/scaling_plots/`。`run.py` 新增 `scaling` 子命令读取 `configs/scaling.yaml` 驱动流水线。此 Stage 无 GPU 依赖，纯 CPU 数值计算（scipy + numpy），可在 Win11 开发机上全量运行。

**Tech Stack:** Python 3.10+, numpy, scipy.optimize（`curve_fit` + `minimize`），matplotlib（Agg 后端），pyyaml，pytest，argparse

---

## 数学公式说明

### Chinchilla 损失函数（Hoffmann et al., 2022）

```
L(N, D) = E + A/N^α + B/D^β
```

参数含义：
- `E` — 不可约损失（数据固有熵），与 N/D 无关，约 1.69
- `A` — 容量惩罚系数（模型容量不足的影响），约 406.4
- `α` — 容量衰减指数（增加参数量时损失改善速率），约 0.34
- `B` — 数据惩罚系数（训练数据不足的影响），约 410.7
- `β` — 数据衰减指数（增加 token 数时损失改善速率），约 0.28

拟合目标函数（对数空间 MSE，等权处理各量级样本）：

```
objective = mean( (log(L_pred) - log(L_actual))^2 )
```

优化算法：L-BFGS-B（拟牛顿有界优化，scipy.optimize.minimize）

### IsoFLOPs 幂律（IsoFLOPs Approach 1）

```
N_opt(C) = A_N * C^a
D_opt(C) = B_D * C^b
```

其中 `a + b ≈ 1`（量纲守恒，由 `C = 6ND` 保证）。

拟合方法：对数线性化（log-space linear regression，scipy.optimize.curve_fit），避免非线性优化器初值敏感发散：

```
log(N_opt) = a * log(C) + log(A_N)   # 线性化幂律
log(D_opt) = b * log(C) + log(B_D)
```

计算约束关系：`C = 6ND`（前向 2ND + 反向 4ND FLOPs）

---

## 文件映射

**新建文件：**
- `llm_foundry/stage3_scaling/scaling.py` — `ScalingAnalyzer` 类，整合两个参考模块
- `data/scaling_experiments.json` — 内置最小化实验数据（10 条，供测试和默认运行使用）
- `reproduce/stage3_scaling.sh` — 复现脚本
- `reproduce/expected/scaling_params.json` — 预期参数文件（供 verify.py 对比，±2% 容差）

**修改文件：**
- `llm_foundry/stage3_scaling/__init__.py` — 导出 `ScalingAnalyzer`
- `configs/scaling.yaml` — 填充真实配置（替换 Plan 1 的占位内容）
- `run.py` — 将 `scaling` 子命令从"尚未实现"占位改为实际调用 `ScalingAnalyzer`
- `tests/test_stages.py` — 新增 `test_stage3_scaling` 测试函数

**参考（只读，不修改）：**
- `reference_resource/Assignment3-scaling/cs336_scaling/chinchilla_isoflops_scaling.py` — IsoFLOPs 拟合逻辑来源
- `reference_resource/Assignment3-scaling/cs336_scaling/chinchilla_scaling_laws_fitting.py` — Chinchilla 拟合逻辑来源

---

## Task 1: 内置实验数据 + 更新 configs/scaling.yaml

**Files:**
- Create: `data/scaling_experiments.json`
- Modify: `configs/scaling.yaml`

- [ ] **Step 1: 创建 `data/scaling_experiments.json`**

这是内置的最小化实验数据集，10 条记录覆盖多个计算预算截面，供测试和默认运行使用。每条记录包含 `n_params`（模型参数量）、`n_tokens`（训练 token 数）、`loss`（验证损失）。

计算预算由 `C = 6 * n_params * n_tokens` 推算，以确保实验数据满足 IsoFLOPs 截面分组逻辑。

```json
{
    "experiments": [
        {"n_params": 1e7, "n_tokens": 1.67e9, "loss": 3.82},
        {"n_params": 3e7, "n_tokens": 5.56e8, "loss": 3.51},
        {"n_params": 1e8, "n_tokens": 1.67e8, "loss": 3.20},
        {"n_params": 1e8, "n_tokens": 5.0e9,  "loss": 2.95},
        {"n_params": 3e8, "n_tokens": 1.67e9, "loss": 2.70},
        {"n_params": 1e9, "n_tokens": 5.0e8,  "loss": 2.50},
        {"n_params": 3e9, "n_tokens": 1.67e8, "loss": 2.38},
        {"n_params": 1e7, "n_tokens": 5.0e10, "loss": 2.85},
        {"n_params": 3e8, "n_tokens": 5.56e7, "loss": 2.91},
        {"n_params": 1e9, "n_tokens": 1.67e10, "loss": 2.20}
    ]
}
```

**注意**：实验数据的 `n_params` 和 `n_tokens` 字段直接给出（不同于参考代码中的 `compute_budget` + `parameters` 格式）。`ScalingAnalyzer` 的 `run()` 方法接受此格式的 `input_data`，内部自行计算 `compute = 6 * n_params * n_tokens`。

- [ ] **Step 2: 更新 `configs/scaling.yaml`**

将 Plan 1 创建的占位内容替换为实际配置：

```yaml
scaling:
  experiments_file: data/scaling_experiments.json
  compute_budgets: [1.0e18, 1.0e19, 1.0e20, 1.0e21]  # IsoFLOPs 外推目标（FLOPs）

output:
  base_dir: results/
  save_plots: true
  plot_format: png  # 或 svg
```

- [ ] **Step 3: Commit**

```bash
git add data/scaling_experiments.json configs/scaling.yaml
git commit -m "feat: add scaling experiment data and update scaling.yaml config"
```

---

## Task 2: `llm_foundry/stage3_scaling/scaling.py` — ScalingAnalyzer 类

**Files:**
- Create: `llm_foundry/stage3_scaling/scaling.py`

此文件整合两个参考模块的逻辑：
- `chinchilla_isoflops_scaling.py` → `analyze_isoflops()` 方法
- `chinchilla_scaling_laws_fitting.py` → `fit_chinchilla()` 方法

**重要说明**：
- 参考文件 `chinchilla_scaling_laws_fitting.py` 包含对 Stanford HyperTuring 远程 API 的调用（`HyperTuringAPIClient`）。**不要复制 API 客户端相关代码**，只提取 `parametric_loss_hypothesis`、`l_bfgs_b_objective_function`、`compute_non_embedding_parameters` 等纯数值计算函数，以及 L-BFGS-B 优化逻辑。
- 参考文件使用 `compute_budget`/`parameters`/`final_loss` 字段，本项目使用 `n_params`/`n_tokens`/`loss` 字段，需在 `run()` 入口处做字段名转换。
- `matplotlib` 必须在导入后立即设置 `Agg` 后端，以支持无显示器环境（Win11 CLI、Linux headless）。

- [ ] **Step 1: 创建 `llm_foundry/stage3_scaling/scaling.py`**

```python
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
from typing import Any

import matplotlib
matplotlib.use("Agg")  # 无显示器环境（CLI / headless Linux），必须在 pyplot import 之前设置
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit, minimize

from llm_foundry.common.config import compute_cfg_hash


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
        self.cfg = cfg
        self._scaling_cfg = cfg.get("scaling", {})
        self._output_cfg = cfg.get("output", {})

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
        experiments = input_data["experiments"]

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
        compute_budgets = self._scaling_cfg.get("compute_budgets", [1e18, 1e19, 1e20, 1e21])
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

        使用 common/config.py 的 compute_cfg_hash 保证跨 Plan hash 算法一致。
        """
        return compute_cfg_hash(input_data)[:8]

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
```

- [ ] **Step 2: 验证文件语法无误**

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
python -c "from llm_foundry.stage3_scaling.scaling import ScalingAnalyzer; print('OK')"
```

Expected: 输出 `OK`，无 ImportError。

- [ ] **Step 3: Commit**

```bash
git add llm_foundry/stage3_scaling/scaling.py
git commit -m "feat: add ScalingAnalyzer with IsoFLOPs and Chinchilla fitting"
```

---

## Task 3: 更新 `llm_foundry/stage3_scaling/__init__.py`

**Files:**
- Modify: `llm_foundry/stage3_scaling/__init__.py`

- [ ] **Step 1: 更新 `__init__.py` 导出 ScalingAnalyzer**

将 `llm_foundry/stage3_scaling/__init__.py` 内容替换为：

```python
from llm_foundry.stage3_scaling.scaling import ScalingAnalyzer

__all__ = ["ScalingAnalyzer"]
```

- [ ] **Step 2: 验证导入**

```bash
python -c "from llm_foundry.stage3_scaling import ScalingAnalyzer; print('OK')"
```

Expected: 输出 `OK`

- [ ] **Step 3: Commit**

```bash
git add llm_foundry/stage3_scaling/__init__.py
git commit -m "feat: export ScalingAnalyzer from stage3_scaling __init__"
```

---

## Task 4: 更新 `run.py` — 实现 `scaling` 子命令

**Files:**
- Modify: `run.py`

Plan 1 已创建 `run.py` 并添加了 `scaling` 子命令的 argparse 定义（占位），以及 `cmd_scaling` 函数（打印"尚未实现"并退出）。本 Task 将 `cmd_scaling` 从占位改为实际调用 `ScalingAnalyzer`。

**重要说明**：Plan 1 的 `scaling` 子命令使用了 `chinchilla` / `isoflops` 子子命令结构。本 Plan 要求接口为 `python run.py scaling --config configs/scaling.yaml`（不使用子子命令）。需要修改 argparse 定义和 `cmd_scaling` 函数。

- [ ] **Step 1: 修改 `run.py` 中的 `cmd_scaling` 函数**

将 `run.py` 中的 `cmd_scaling` 函数替换为以下实现：

```python
def cmd_scaling(args) -> None:
    """
    缩放律分析子命令。

    用法：
        python run.py scaling --config configs/scaling.yaml
    """
    from llm_foundry.common.config import load_config
    from llm_foundry.stage3_scaling import ScalingAnalyzer

    # 使用 load_config 统一加载配置（支持 --set override，生成 config_snapshot.yaml）
    cfg, run_dir = load_config(args)

    experiments_file = cfg.scaling.get("experiments_file", "data/scaling_experiments.json")
    import json
    with open(experiments_file, "r", encoding="utf-8") as f:
        input_data = json.load(f)

    analyzer = ScalingAnalyzer(cfg.__dict__)  # SimpleNamespace 转 dict
    result = analyzer.run(input_data)

    print(f"\n[scaling] Chinchilla 拟合参数:")
    ch = result["chinchilla"]
    print(f"  E={ch['E']:.4f}  A={ch['A']:.2f}  alpha={ch['alpha']:.4f}  B={ch['B']:.2f}  beta={ch['beta']:.4f}")
    print(f"[scaling] 结果文件: {run_dir}/scaling_params.json")
```

- [ ] **Step 2: 修改 `run.py` 中 `scaling` 子命令的 argparse 定义**

在 `build_parser()` 函数中，找到 `scaling` 子命令的定义部分（当前为带 `chinchilla`/`isoflops` 子子命令的结构），替换为简单的 `--config` 参数：

```python
    # scaling（Plan 4 实现）
    scale_p = subparsers.add_parser("scaling", help="缩放规律分析（IsoFLOPs + Chinchilla）")
    scale_p.add_argument("--config", default="configs/scaling.yaml", help="配置文件路径")
```

**注意**：替换时只修改 `scaling` 子命令的定义，不影响其他子命令。

- [ ] **Step 3: 手动冒烟测试**

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
python run.py scaling --config configs/scaling.yaml
```

Expected：
- 控制台输出 Chinchilla 拟合参数（E/A/alpha/B/beta 的具体数值）
- 控制台输出 IsoFLOPs 理论检验 `a + b = ...`
- 控制台输出结果文件路径（`results/{hash}/scaling_params.json`）
- `results/{hash}/scaling_params.json` 文件存在
- `results/{hash}/scaling_plots/isoflops_scaling.png` 和 `chinchilla_fit.png` 存在

- [ ] **Step 4: Commit**

```bash
git add run.py
git commit -m "feat: implement run.py scaling subcommand using ScalingAnalyzer"
```

---

## Task 5: `tests/test_stages.py` — 新增 `test_stage3_scaling`

**Files:**
- Modify: `tests/test_stages.py`（若不存在则新建）

**测试设计原则**：
- 使用内置最小化数据（10 条），不依赖外部文件
- 不验证拟合数值精度（只验证结构完整性）
- Win11 CPU 可运行，耗时 < 10 秒（scipy L-BFGS-B 在此数据量下 < 1 秒）
- 不需要 GPU

- [ ] **Step 1: 写失败测试**

若 `tests/test_stages.py` 不存在，新建；若存在，追加以下测试函数：

```python
# tests/test_stages.py
import json
import os
import pytest

# ─────────────────────────────────────────
# test_stage3_scaling
# ─────────────────────────────────────────

MINIMAL_SCALING_DATA = {
    "experiments": [
        # 计算预算约 1e17 FLOPs（C = 6 * N * D）
        {"n_params": 1e7, "n_tokens": 1.67e9, "loss": 3.82},
        {"n_params": 3e7, "n_tokens": 5.56e8, "loss": 3.51},
        {"n_params": 1e8, "n_tokens": 1.67e8, "loss": 3.20},
        # 计算预算约 3e18 FLOPs
        {"n_params": 1e8, "n_tokens": 5.0e9,  "loss": 2.95},
        {"n_params": 3e8, "n_tokens": 1.67e9, "loss": 2.70},
        {"n_params": 1e9, "n_tokens": 5.0e8,  "loss": 2.50},
        # 计算预算约 1e19 FLOPs
        {"n_params": 3e9, "n_tokens": 5.56e8, "loss": 2.38},
        {"n_params": 1e7, "n_tokens": 5.0e10, "loss": 2.85},
        {"n_params": 3e8, "n_tokens": 5.56e7, "loss": 2.91},
        {"n_params": 1e9, "n_tokens": 1.67e10, "loss": 2.20},
    ]
}

MINIMAL_SCALING_CFG = {
    "scaling": {
        "experiments_file": "data/scaling_experiments.json",
        "compute_budgets": [1e18, 1e19],
    },
    "output": {
        "base_dir": "results/",
        "save_plots": True,
        "plot_format": "png",
    },
}


def test_stage3_scaling(tmp_path):
    """
    集成测试：ScalingAnalyzer.run() 能正常执行并产出 scaling_params.json。

    验证内容：
    1. scaling_params.json 存在且为合法 JSON
    2. 包含 chinchilla 字段，其中有 alpha/beta/A/B/E 五个数值字段
    3. 包含 isoflops_optimal 字段，其中每个条目含 compute/n_params/n_tokens
    4. 不验证拟合数值精度
    """
    from llm_foundry.stage3_scaling import ScalingAnalyzer

    # 将输出重定向到 tmp_path（隔离测试产出，不污染项目 results/ 目录）
    cfg = {
        "scaling": MINIMAL_SCALING_CFG["scaling"],
        "output": {
            "base_dir": str(tmp_path),
            "save_plots": True,
            "plot_format": "png",
        },
    }

    analyzer = ScalingAnalyzer(cfg)
    result = analyzer.run(MINIMAL_SCALING_DATA)

    # 1. 返回值结构验证
    assert "chinchilla" in result, "result 缺少 chinchilla 字段"
    assert "isoflops_optimal" in result, "result 缺少 isoflops_optimal 字段"
    assert "plots_dir" in result, "result 缺少 plots_dir 字段"

    # 2. chinchilla 参数字段验证
    ch = result["chinchilla"]
    for field in ("alpha", "beta", "A", "B", "E"):
        assert field in ch, f"chinchilla 缺少字段 {field}"
        assert isinstance(ch[field], float), f"chinchilla.{field} 应为 float"

    # 3. isoflops_optimal 结构验证
    isoflops = result["isoflops_optimal"]
    assert isinstance(isoflops, list), "isoflops_optimal 应为列表"
    assert len(isoflops) == len(MINIMAL_SCALING_CFG["scaling"]["compute_budgets"])
    for entry in isoflops:
        assert "compute" in entry
        assert "n_params" in entry
        assert "n_tokens" in entry

    # 4. scaling_params.json 文件验证
    plots_dir = result["plots_dir"]
    run_dir = os.path.dirname(plots_dir)
    params_path = os.path.join(run_dir, "scaling_params.json")
    assert os.path.isfile(params_path), f"scaling_params.json 不存在: {params_path}"

    with open(params_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    assert "chinchilla" in loaded, "scaling_params.json 缺少 chinchilla 字段"
    assert "isoflops_optimal" in loaded, "scaling_params.json 缺少 isoflops_optimal 字段"

    # 5. 图表文件验证
    isoflops_plot = os.path.join(plots_dir, "isoflops_scaling.png")
    chinchilla_plot = os.path.join(plots_dir, "chinchilla_fit.png")
    assert os.path.isfile(isoflops_plot), f"IsoFLOPs 图表不存在: {isoflops_plot}"
    assert os.path.isfile(chinchilla_plot), f"Chinchilla 图表不存在: {chinchilla_plot}"
```

- [ ] **Step 2: 运行测试，确认失败（若 scaling.py 尚未创建）或直接通过（若已创建）**

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
pytest tests/test_stages.py::test_stage3_scaling -v
```

Expected: PASSED（< 10 秒）

- [ ] **Step 3: 运行全量测试，确认无回归**

```bash
pytest tests/ -v
```

Expected: 所有既有测试 PASSED，`test_stage3_scaling` 也 PASSED

- [ ] **Step 4: Commit**

```bash
git add tests/test_stages.py
git commit -m "test: add test_stage3_scaling integration test"
```

---

## Task 6: `reproduce/stage3_scaling.sh` + `reproduce/expected/scaling_params.json`

**Files:**
- Create: `reproduce/stage3_scaling.sh`
- Create: `reproduce/expected/scaling_params.json`

- [ ] **Step 1: 创建 `reproduce/` 目录（若不存在）**

```bash
mkdir -p /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator/reproduce/expected
```

- [ ] **Step 2: 创建 `reproduce/stage3_scaling.sh`**

```bash
#!/bin/bash
# reproduce/stage3_scaling.sh — Stage 3 Scaling 分析复现脚本
#
# 用法：
#   bash reproduce/stage3_scaling.sh
#
# 输出：
#   results/{hash}/scaling_params.json
#   results/{hash}/scaling_plots/

set -e  # 任意命令失败则立即退出

echo "[Stage 3] 开始 Scaling Laws 分析..."
python run.py scaling --config configs/scaling.yaml
echo "[Stage 3] 分析完成。"

# 获取最新的 results/ 子目录（按修改时间排序，取最新的）
LATEST_DIR=$(ls -td results/*/ 2>/dev/null | head -1)

if [ -z "$LATEST_DIR" ]; then
    echo "[ERROR] results/ 目录为空，分析可能失败。"
    exit 1
fi

HASH=$(basename "$LATEST_DIR")
echo "[Stage 3] 输出目录: results/$HASH"
echo "[Stage 3] 参数文件: results/$HASH/scaling_params.json"
echo "[Stage 3] 图表目录: results/$HASH/scaling_plots/"

# 若有 verify.py，执行对比验证
if [ -f "reproduce/verify.py" ]; then
    echo "[Stage 3] 执行参数对比验证（±2% 容差）..."
    python reproduce/verify.py \
        --stage 3 \
        --metrics "results/$HASH/scaling_params.json" \
        --expected reproduce/expected/scaling_params.json \
        --tolerance 0.02
fi
```

- [ ] **Step 3: 首次运行，生成真实参数文件，作为 expected 基准**

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
python run.py scaling --config configs/scaling.yaml
```

运行后，找到 `results/{hash}/scaling_params.json`，将其内容复制到 `reproduce/expected/scaling_params.json`。

**注意**：`expected/scaling_params.json` 只需包含 `chinchilla` 和 `isoflops_optimal` 字段，不需要包含 `plots_dir` 等运行时路径字段。将复制的内容精简为：

```json
{
    "chinchilla": {
        "alpha": <从实际运行结果填写>,
        "beta": <从实际运行结果填写>,
        "A": <从实际运行结果填写>,
        "B": <从实际运行结果填写>,
        "E": <从实际运行结果填写>
    },
    "isoflops_optimal": [
        {"compute": 1e18, "n_params": <填写>, "n_tokens": <填写>},
        {"compute": 1e19, "n_params": <填写>, "n_tokens": <填写>},
        {"compute": 1e20, "n_params": <填写>, "n_tokens": <填写>},
        {"compute": 1e21, "n_params": <填写>, "n_tokens": <填写>}
    ]
}
```

- [ ] **Step 4: 给 sh 脚本添加可执行权限（Linux/Mac），Win11 下跳过此步**

```bash
# Linux/Mac 下执行（Win11 CLI 跳过）
chmod +x reproduce/stage3_scaling.sh
```

- [ ] **Step 5: Commit**

```bash
git add reproduce/stage3_scaling.sh reproduce/expected/scaling_params.json
git commit -m "feat: add reproduce/stage3_scaling.sh and expected params baseline"
```

---

## Task 7: 全量测试 + 端到端验收

**Files:**
- 无新增文件

- [ ] **Step 1: 运行全量 pytest**

```bash
cd /mnt/c/Users/liu_j/Desktop/SJTU/AI/LLM/LLM_Foundry_Simulator
pytest tests/ -v
```

Expected: 所有测试 PASSED，无 FAILED 或 ERROR

- [ ] **Step 2: 端到端验收（主验收标准）**

```bash
python run.py scaling --config configs/scaling.yaml
```

验证以下条件全部满足：
1. 命令正常退出（exit code 0）
2. 控制台输出 Chinchilla 参数（E/A/alpha/B/beta 数值，非"尚未实现"）
3. `results/{hash}/scaling_params.json` 存在且为合法 JSON
4. `scaling_params.json` 包含 `chinchilla` 字段（含 alpha/beta/A/B/E）
5. `scaling_params.json` 包含 `isoflops_optimal` 字段（列表，每项含 compute/n_params/n_tokens）
6. `results/{hash}/scaling_plots/isoflops_scaling.png` 存在
7. `results/{hash}/scaling_plots/chinchilla_fit.png` 存在

- [ ] **Step 3: 验证 pytest::test_stage3_scaling 单独可运行**

```bash
pytest tests/test_stages.py::test_stage3_scaling -v --tb=short
```

Expected: PASSED，耗时 < 10 秒

- [ ] **Step 4: 最终 Commit**

```bash
git add -u
git commit -m "feat: plan4 complete - scaling laws analysis with IsoFLOPs + Chinchilla"
```

---

## 前置条件

本 Plan 依赖以下已完成工作：
- Plan 1：`llm_foundry/stage3_scaling/__init__.py`（空文件）已存在；`configs/scaling.yaml` 占位文件已存在；`run.py` CLI 框架已存在，`scaling` 子命令已在 argparse 中注册（占位实现）
- Plan 2：不强依赖（本 Stage 无数据清洗依赖）
- Plan 3：不强依赖（本 Stage 无 GPU/训练依赖）

若 Plan 1 未完成，需先手动创建：
```bash
mkdir -p llm_foundry/stage3_scaling
touch llm_foundry/stage3_scaling/__init__.py
```

---

## 实施顺序建议

按以下顺序执行，每个 Task 完成后立即运行对应验证命令：

1. **Task 1**（数据 + 配置）— 最简单，先准备好输入数据
2. **Task 2**（ScalingAnalyzer）— 核心逻辑，最耗时
3. **Task 3**（__init__.py）— 1 分钟，让 import 生效
4. **Task 4**（run.py）— 接通 CLI 入口
5. **Task 5**（测试）— 验证行为
6. **Task 6**（复现脚本）— 生成 expected 基准文件
7. **Task 7**（全量验收）— 确保无回归

---

## 验收标准

Plan 4 完成后，以下条件应全部满足：

1. `python run.py scaling --config configs/scaling.yaml` 正常退出（exit code 0）
2. `results/{hash}/scaling_params.json` 存在，包含 `chinchilla` 和 `isoflops_optimal` 字段
3. `results/{hash}/scaling_plots/isoflops_scaling.png` 和 `chinchilla_fit.png` 存在
4. `pytest tests/test_stages.py::test_stage3_scaling` PASSED（< 10 秒，无 GPU）
5. `pytest tests/` 全部 PASSED（无回归）
6. `reproduce/stage3_scaling.sh` 存在，`bash reproduce/stage3_scaling.sh` 可运行
7. `reproduce/expected/scaling_params.json` 包含基准 Chinchilla 参数和 isoflops_optimal
