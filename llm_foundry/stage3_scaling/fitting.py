"""
llm_foundry/stage3_scaling/fitting.py — Chinchilla Scaling Laws Fitting

Chinchilla 损失拟合模块，实现 Hoffmann et al., 2022 的缩放定律参数估计。

核心公式：
    L(N, D) = E + A/N^α + B/D^β

其中：
    E    — 不可约损失（数据固有熵），与 N/D 无关
    A    — 容量惩罚系数（模型容量不足的影响）
    α    — 容量衰减指数（增加参数量时损失改善速率）
    B    — 数据惩罚系数（训练数据不足的影响）
    β    — 数据衰减指数（增加 token 数时损失改善速率）

拟合目标函数（对数空间 MSE）：
    objective = mean( (log(L_pred) - log(L_actual))^2 )

优化算法：L-BFGS-B（scipy.optimize.minimize）

计算约束：C = 6 * N * D（前向 2ND + 反向 4ND FLOPs）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


# ─────────────────────────────────────────
# 配置类
# ─────────────────────────────────────────

@dataclass
class FittingConfig:
    """Chinchilla 拟合配置类。

    Attributes:
        initial_guess: 优化器初始猜测值 [E, A, B, alpha, beta]
        bounds: 参数边界约束 [(E_min, E_max), (A_min, A_max), ...]
        epsilon: 数值稳定性保护（防除零）
        method: 优化算法（默认 L-BFGS-B）
    """

    initial_guess: list[float] = field(default_factory=lambda: [1.5, 450.0, 2100.0, 0.35, 0.35])
    bounds: list[tuple[float, float]] = field(default_factory=lambda: [
        (0.1, 5.0),       # E: 不可约损失
        (1.0, 10000.0),   # A: 容量惩罚系数
        (1.0, 10000.0),   # B: 数据惩罚系数
        (0.01, 1.0),      # alpha: 容量衰减指数
        (0.01, 1.0),      # beta: 数据衰减指数
    ])
    epsilon: float = 1e-12
    method: str = "L-BFGS-B"


# ─────────────────────────────────────────
# 核心函数
# ─────────────────────────────────────────

def chinchilla_loss(
    n_params: float | np.ndarray,
    n_tokens: float | np.ndarray,
    E: float,
    A: float,
    B: float,
    alpha: float,
    beta: float,
    epsilon: float = 1e-12,
) -> float | np.ndarray:
    """Chinchilla 损失函数 L(N, D) = E + A/N^α + B/D^β。

    三项的物理含义：
        E：不可约损失（数据集固有熵），无法通过扩大模型或数据消除
        A/N^α：容量惩罚项，参数量越大惩罚越小
        B/D^β：数据惩罚项，token 越多惩罚越小

    Args:
        n_params: 模型非嵌入参数量 N（标量或数组）
        n_tokens: 训练 token 数 D（标量或数组）
        E: 不可约损失
        A: 容量惩罚系数
        B: 数据惩罚系数
        alpha: 容量衰减指数
        beta: 数据衰减指数
        epsilon: 数值稳定性保护（防除零）

    Returns:
        预测损失值（标量或数组，与输入形状一致）

    Examples:
        >>> # 单点预测
        >>> loss = chinchilla_loss(1e9, 1e10, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
        >>> print(f"Predicted loss: {loss:.4f}")

        >>> # 批量预测
        >>> N = np.array([1e7, 1e8, 1e9])
        >>> D = np.array([1e9, 1e10, 1e11])
        >>> losses = chinchilla_loss(N, D, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
    """
    penalty_capacity = A / (np.power(n_params, alpha) + epsilon)
    penalty_data = B / (np.power(n_tokens, beta) + epsilon)
    return E + penalty_capacity + penalty_data


def _objective_function(
    params: tuple[float, ...],
    n_params: np.ndarray,
    n_tokens: np.ndarray,
    losses: np.ndarray,
    epsilon: float = 1e-12,
) -> float:
    """L-BFGS-B 优化目标函数：对数空间均方残差（log-space MSE）。

    选用对数空间残差的原因：等权处理各量级样本，避免大损失值主导拟合。
        objective = mean( (log(L_pred) - log(L_actual))^2 )

    Args:
        params: (E, A, B, alpha, beta)
        n_params: 实验参数量数组，shape: (num_samples,)
        n_tokens: 实验 token 数数组，shape: (num_samples,)
        losses: 实测验证损失数组，shape: (num_samples,)
        epsilon: 数值稳定性保护

    Returns:
        对数空间均方残差（标量），L-BFGS-B 最小化此目标
    """
    E, A, B, alpha, beta = params
    predicted = chinchilla_loss(n_params, n_tokens, E, A, B, alpha, beta, epsilon)
    log_diff = np.log(predicted) - np.log(losses)
    return float(np.mean(log_diff ** 2))


def fit_chinchilla_params(
    n_params: np.ndarray | list[float],
    n_tokens: np.ndarray | list[float],
    losses: np.ndarray | list[float],
    config: FittingConfig | None = None,
) -> dict[str, float]:
    """拟合 Chinchilla 损失函数参数。

    使用 L-BFGS-B 算法在对数空间中最小化预测损失与实测损失的均方差，
    求解五个参数 (E, A, B, alpha, beta)。

    Args:
        n_params: 模型参数量数组，shape: (num_samples,)
        n_tokens: 训练 token 数数组，shape: (num_samples,)
        losses: 验证损失数组，shape: (num_samples,)
        config: 拟合配置（默认使用 FittingConfig()）

    Returns:
        拟合参数字典：
        {
            "E": float,      # 不可约损失（原论文约 1.69）
            "A": float,      # 容量惩罚系数（原论文约 406.4）
            "alpha": float,  # 容量衰减指数（原论文约 0.34）
            "B": float,      # 数据惩罚系数（原论文约 410.7）
            "beta": float,   # 数据衰减指数（原论文约 0.28）
        }

    Raises:
        ValueError: 如果输入数组长度不一致或数据点过少
        RuntimeError: 如果优化失败

    Examples:
        >>> import numpy as np
        >>> N = np.array([1e7, 1e8, 1e9, 1e10])
        >>> D = np.array([1e9, 1e10, 1e11, 1e12])
        >>> L = np.array([3.5, 2.8, 2.3, 2.0])
        >>> params = fit_chinchilla_params(N, D, L)
        >>> print(f"E={params['E']:.4f}, alpha={params['alpha']:.4f}")
    """
    # 转换为 numpy 数组
    N_arr = np.asarray(n_params, dtype=float)
    D_arr = np.asarray(n_tokens, dtype=float)
    L_arr = np.asarray(losses, dtype=float)

    # 验证输入
    if not (len(N_arr) == len(D_arr) == len(L_arr)):
        raise ValueError(
            f"Input arrays must have same length, got "
            f"len(n_params)={len(N_arr)}, len(n_tokens)={len(D_arr)}, len(losses)={len(L_arr)}"
        )
    if len(N_arr) < 5:
        raise ValueError(
            f"Need at least 5 data points to fit 5 parameters, got {len(N_arr)}"
        )

    # 使用默认配置
    cfg = config if config is not None else FittingConfig()

    # 执行 L-BFGS-B 优化
    result = minimize(
        _objective_function,
        cfg.initial_guess,
        args=(N_arr, D_arr, L_arr, cfg.epsilon),
        method=cfg.method,
        bounds=cfg.bounds,
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    E_opt, A_opt, B_opt, alpha_opt, beta_opt = result.x

    return {
        "E": float(E_opt),
        "A": float(A_opt),
        "B": float(B_opt),
        "alpha": float(alpha_opt),
        "beta": float(beta_opt),
    }


def predict_loss(
    n_params: float | np.ndarray,
    n_tokens: float | np.ndarray,
    params: dict[str, float],
) -> float | np.ndarray:
    """使用拟合的 Chinchilla 参数预测损失。

    Args:
        n_params: 模型参数量 N（标量或数组）
        n_tokens: 训练 token 数 D（标量或数组）
        params: 拟合参数字典，必须包含 E, A, B, alpha, beta

    Returns:
        预测损失值（标量或数组，与输入形状一致）

    Raises:
        KeyError: 如果 params 缺少必要字段

    Examples:
        >>> params = {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28}
        >>> loss = predict_loss(1e9, 1e10, params)
        >>> print(f"Predicted loss: {loss:.4f}")
    """
    required = {"E", "A", "B", "alpha", "beta"}
    missing = required - set(params.keys())
    if missing:
        raise KeyError(f"Missing required parameters: {missing}")

    return chinchilla_loss(
        n_params=n_params,
        n_tokens=n_tokens,
        E=params["E"],
        A=params["A"],
        B=params["B"],
        alpha=params["alpha"],
        beta=params["beta"],
    )


def compute_optimal_allocation(
    compute_budget: float,
    params: dict[str, float],
    n_search_points: int = 50000,
    n_range: tuple[float, float] = (1e6, 1e12),
) -> dict[str, float]:
    """在给定计算预算下计算最优的 N 和 D 分配。

    在固定计算预算 C = 6ND 的约束下，寻找使 L(N, D) 最小的最优参数量 N_opt
    和对应的最优 token 数 D_opt = C / (6 * N_opt)。

    使用一维网格搜索在预算超平面上寻找损失最低点。

    Args:
        compute_budget: 计算预算 C（FLOPs）
        params: 拟合的 Chinchilla 参数字典
        n_search_points: 网格搜索点数（默认 50000）
        n_range: 参数量搜索范围 (N_min, N_max)

    Returns:
        最优分配字典：
        {
            "n_params": float,      # 最优参数量 N_opt
            "n_tokens": float,      # 最优 token 数 D_opt
            "loss": float,          # 预测的最小损失
            "compute_budget": float, # 计算预算 C
            "tokens_per_param": float,  # D/N 比率（Chinchilla 建议约 20）
        }

    Raises:
        ValueError: 如果 compute_budget <= 0
        KeyError: 如果 params 缺少必要字段

    Examples:
        >>> params = {"E": 1.69, "A": 406.4, "B": 410.7, "alpha": 0.34, "beta": 0.28}
        >>> result = compute_optimal_allocation(1e19, params)
        >>> print(f"Optimal N: {result['n_params']:.2e}, D: {result['n_tokens']:.2e}")
        >>> print(f"D/N ratio: {result['tokens_per_param']:.2f}")
    """
    if compute_budget <= 0:
        raise ValueError(f"compute_budget must be positive, got {compute_budget}")

    required = {"E", "A", "B", "alpha", "beta"}
    missing = required - set(params.keys())
    if missing:
        raise KeyError(f"Missing required parameters: {missing}")

    # 提取参数
    E = params["E"]
    A = params["A"]
    B = params["B"]
    alpha = params["alpha"]
    beta = params["beta"]

    # 构建搜索空间（对数均匀分布）
    N_search = np.logspace(
        np.log10(n_range[0]),
        np.log10(n_range[1]),
        n_search_points,
    )

    # 计算约束下的 D: D = C / (6N)
    D_derived = compute_budget / (6.0 * N_search)

    # 计算对应损失
    losses = chinchilla_loss(N_search, D_derived, E, A, B, alpha, beta)

    # 找到最小损失对应的索引
    min_idx = np.argmin(losses)
    optimal_N = float(N_search[min_idx])
    optimal_D = float(D_derived[min_idx])
    min_loss = float(losses[min_idx])

    return {
        "n_params": optimal_N,
        "n_tokens": optimal_D,
        "loss": min_loss,
        "compute_budget": compute_budget,
        "tokens_per_param": optimal_D / optimal_N,
    }


def load_experiments_from_csv(csv_path: str | Path) -> dict[str, np.ndarray]:
    """从 CSV 文件加载实验数据。

    期望 CSV 格式：
        n_params,n_tokens,loss
        1e7,1.67e9,3.82
        1e8,1.67e8,3.20
        ...

    Args:
        csv_path: CSV 文件路径

    Returns:
        数据字典：{"n_params": array, "n_tokens": array, "loss": array}

    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果 CSV 格式不正确
    """
    import csv

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    n_params_list = []
    n_tokens_list = []
    loss_list = []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                n_params_list.append(float(row["n_params"]))
                n_tokens_list.append(float(row["n_tokens"]))
                loss_list.append(float(row["loss"]))
            except (KeyError, ValueError) as e:
                raise ValueError(f"Invalid CSV format: {e}")

    return {
        "n_params": np.array(n_params_list),
        "n_tokens": np.array(n_tokens_list),
        "loss": np.array(loss_list),
    }


def compute_flops(n_params: float, n_tokens: float) -> float:
    """计算训练 FLOPs。

    使用 Kaplan et al., 2020 的估算公式：
        C = 6 * N * D

    其中：
        前向传播：约 2ND FLOPs
        反向传播：约 4ND FLOPs（梯度计算 + 激活梯度）
        总计：2ND + 4ND = 6ND

    Args:
        n_params: 模型参数量 N
        n_tokens: 训练 token 数 D

    Returns:
        训练所需总 FLOPs

    Examples:
        >>> C = compute_flops(1e9, 1e10)  # 1B 参数，10B tokens
        >>> print(f"FLOPs: {C:.2e}")  # 6e19 FLOPs
    """
    return 6.0 * n_params * n_tokens


def derive_tokens_from_flops(flops: float, n_params: float) -> float:
    """根据 FLOPs 和参数量反推训练 token 数。

    由 C = 6ND 推导：D = C / (6N)

    Args:
        flops: 训练 FLOPs 预算 C
        n_params: 模型参数量 N

    Returns:
        训练 token 数 D

    Raises:
        ValueError: 如果 n_params <= 0
    """
    if n_params <= 0:
        raise ValueError(f"n_params must be positive, got {n_params}")
    return flops / (6.0 * n_params)


# ─────────────────────────────────────────
# 高级接口：ChinchillaFitter 类
# ─────────────────────────────────────────

class ChinchillaFitter:
    """Chinchilla 拟合器类，提供面向对象的高级接口。

    使用方式：
        >>> fitter = ChinchillaFitter()
        >>> fitter.load_data(n_params, n_tokens, losses)
        >>> params = fitter.fit()
        >>> prediction = fitter.predict(N_new, D_new)
        >>> optimal = fitter.compute_optimal_allocation(1e19)
    """

    def __init__(self, config: FittingConfig | None = None):
        """初始化拟合器。

        Args:
            config: 拟合配置（默认使用 FittingConfig()）
        """
        self.config = config if config is not None else FittingConfig()
        self._n_params: np.ndarray | None = None
        self._n_tokens: np.ndarray | None = None
        self._losses: np.ndarray | None = None
        self._params: dict[str, float] | None = None

    def load_data(
        self,
        n_params: np.ndarray | list[float],
        n_tokens: np.ndarray | list[float],
        losses: np.ndarray | list[float],
    ) -> "ChinchillaFitter":
        """加载实验数据（支持链式调用）。

        Args:
            n_params: 模型参数量数组
            n_tokens: 训练 token 数数组
            losses: 验证损失数组

        Returns:
            self（支持链式调用）
        """
        self._n_params = np.asarray(n_params, dtype=float)
        self._n_tokens = np.asarray(n_tokens, dtype=float)
        self._losses = np.asarray(losses, dtype=float)
        return self

    def load_from_csv(self, csv_path: str | Path) -> "ChinchillaFitter":
        """从 CSV 文件加载数据（支持链式调用）。

        Args:
            csv_path: CSV 文件路径

        Returns:
            self（支持链式调用）
        """
        data = load_experiments_from_csv(csv_path)
        self._n_params = data["n_params"]
        self._n_tokens = data["n_tokens"]
        self._losses = data["loss"]
        return self

    def fit(self) -> dict[str, float]:
        """执行拟合。

        Returns:
            拟合参数字典

        Raises:
            ValueError: 如果没有加载数据
        """
        if self._n_params is None or self._n_tokens is None or self._losses is None:
            raise ValueError("No data loaded. Call load_data() or load_from_csv() first.")

        self._params = fit_chinchilla_params(
            self._n_params,
            self._n_tokens,
            self._losses,
            self.config,
        )
        return self._params

    def predict(
        self,
        n_params: float | np.ndarray,
        n_tokens: float | np.ndarray,
    ) -> float | np.ndarray:
        """预测损失。

        Args:
            n_params: 模型参数量
            n_tokens: 训练 token 数

        Returns:
            预测损失

        Raises:
            RuntimeError: 如果尚未拟合
        """
        if self._params is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return predict_loss(n_params, n_tokens, self._params)

    def compute_optimal_allocation(
        self,
        compute_budget: float,
        n_search_points: int = 50000,
        n_range: tuple[float, float] = (1e6, 1e12),
    ) -> dict[str, float]:
        """计算最优 N, D 分配。

        Args:
            compute_budget: 计算预算（FLOPs）
            n_search_points: 网格搜索点数
            n_range: 参数量搜索范围

        Returns:
            最优分配字典

        Raises:
            RuntimeError: 如果尚未拟合
        """
        if self._params is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        return compute_optimal_allocation(compute_budget, self._params, n_search_points, n_range)

    @property
    def params(self) -> dict[str, float] | None:
        """获取当前拟合参数（未拟合时返回 None）。"""
        return self._params

    def save_params(self, path: str | Path) -> None:
        """保存拟合参数到 JSON 文件。

        Args:
            path: 输出文件路径

        Raises:
            RuntimeError: 如果尚未拟合
        """
        if self._params is None:
            raise RuntimeError("Model not fitted yet. Call fit() first.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._params, f, indent=2)

    def load_params(self, path: str | Path) -> "ChinchillaFitter":
        """从 JSON 文件加载拟合参数（支持链式调用）。

        Args:
            path: 参数文件路径

        Returns:
            self（支持链式调用）
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            self._params = json.load(f)
        return self
