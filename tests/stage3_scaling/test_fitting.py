"""
tests/stage3_scaling/test_fitting.py — Chinchilla Fitting 模块测试

测试覆盖：
    - chinchilla_loss() 函数
    - fit_chinchilla_params() 函数
    - predict_loss() 函数
    - compute_optimal_allocation() 函数
    - FittingConfig 配置类
    - ChinchillaFitter 高级接口类
    - 数据加载工具函数
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from llm_foundry.stage3_scaling.fitting import (
    FittingConfig,
    ChinchillaFitter,
    chinchilla_loss,
    compute_flops,
    compute_optimal_allocation,
    derive_tokens_from_flops,
    fit_chinchilla_params,
    load_experiments_from_csv,
    predict_loss,
)


# ─────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────

@pytest.fixture
def sample_data():
    """提供最小化的测试数据集（10 条记录）。"""
    return {
        "n_params": np.array([1e7, 3e7, 1e8, 1e8, 3e8, 1e9, 3e9, 1e7, 3e8, 1e9]),
        "n_tokens": np.array([
            1.67e9, 5.56e8, 1.67e8, 5.0e9, 1.67e9, 5.0e8, 5.56e8, 5.0e10, 5.56e7, 1.67e10
        ]),
        "loss": np.array([3.82, 3.51, 3.20, 2.95, 2.70, 2.50, 2.38, 2.85, 2.91, 2.20]),
    }


@pytest.fixture
def expected_params():
    """Chinchilla 论文参考参数值。"""
    return {
        "E": 1.69,
        "A": 406.4,
        "B": 410.7,
        "alpha": 0.34,
        "beta": 0.28,
    }


@pytest.fixture
def temp_csv_file(tmp_path):
    """创建临时 CSV 文件用于测试。"""
    csv_path = tmp_path / "test_experiments.csv"
    content = """n_params,n_tokens,loss
10000000.0,1670000000.0,3.82
30000000.0,556000000.0,3.51
100000000.0,167000000.0,3.20
100000000.0,5000000000.0,2.95
300000000.0,1670000000.0,2.70
"""
    csv_path.write_text(content)
    return csv_path


# ─────────────────────────────────────────
# 基础函数测试
# ─────────────────────────────────────────

class TestChinchillaLoss:
    """测试 chinchilla_loss 函数。"""

    def test_scalar_input(self, expected_params):
        """测试标量输入。"""
        loss = chinchilla_loss(
            n_params=1e9,
            n_tokens=1e10,
            **expected_params
        )
        assert isinstance(loss, float)
        assert loss > expected_params["E"]  # 损失应大于不可约损失

    def test_array_input(self, expected_params):
        """测试数组输入。"""
        N = np.array([1e7, 1e8, 1e9])
        D = np.array([1e9, 1e10, 1e11])
        losses = chinchilla_loss(N, D, **expected_params)

        assert isinstance(losses, np.ndarray)
        assert len(losses) == 3
        assert np.all(losses > expected_params["E"])

    def test_monotonicity(self, expected_params):
        """测试单调性：N 或 D 增加时损失应减小。"""
        # 固定 D，增加 N，损失应减小
        N_small, N_large = 1e7, 1e9
        D_fixed = 1e10

        loss_small = chinchilla_loss(N_small, D_fixed, **expected_params)
        loss_large = chinchilla_loss(N_large, D_fixed, **expected_params)

        assert loss_small > loss_large

        # 固定 N，增加 D，损失应减小
        N_fixed = 1e9
        D_small, D_large = 1e8, 1e12

        loss_small = chinchilla_loss(N_fixed, D_small, **expected_params)
        loss_large = chinchilla_loss(N_fixed, D_large, **expected_params)

        assert loss_small > loss_large

    def test_epsilon_protection(self, expected_params):
        """测试 epsilon 防除零保护。"""
        # 极小的 N 和 D 不应导致除零错误
        loss = chinchilla_loss(1e-100, 1e-100, **expected_params, epsilon=1e-12)
        assert np.isfinite(loss)


class TestFitChinchillaParams:
    """测试 fit_chinchilla_params 函数。"""

    def test_basic_fitting(self, sample_data):
        """测试基本拟合功能。"""
        params = fit_chinchilla_params(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        )

        # 验证返回字段
        assert "E" in params
        assert "A" in params
        assert "B" in params
        assert "alpha" in params
        assert "beta" in params

        # 验证数值类型
        for key, value in params.items():
            assert isinstance(value, float)
            assert np.isfinite(value)

    def test_parameter_ranges(self, sample_data):
        """测试拟合参数在合理范围内。"""
        params = fit_chinchilla_params(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        )

        # E 应在合理范围（自然语言熵约 1-3）
        assert 0 < params["E"] < 10

        # A, B 应为正数
        assert params["A"] > 0
        assert params["B"] > 0

        # alpha, beta 应在 (0, 1) 区间
        assert 0 < params["alpha"] < 1
        assert 0 < params["beta"] < 1

    def test_custom_config(self, sample_data):
        """测试自定义配置。"""
        config = FittingConfig(
            initial_guess=[2.0, 500.0, 500.0, 0.3, 0.3],
            epsilon=1e-10,
        )
        params = fit_chinchilla_params(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
            config=config,
        )

        assert "E" in params
        assert params["E"] > 0

    def test_insufficient_data(self):
        """测试数据点过少时抛出异常。"""
        with pytest.raises(ValueError, match="at least 5 data points"):
            fit_chinchilla_params(
                n_params=[1e7, 1e8],
                n_tokens=[1e9, 1e10],
                losses=[3.5, 2.8],
            )

    def test_mismatched_lengths(self):
        """测试输入长度不匹配时抛出异常。"""
        with pytest.raises(ValueError, match="same length"):
            fit_chinchilla_params(
                n_params=[1e7, 1e8, 1e9],
                n_tokens=[1e9, 1e10],  # 长度不匹配
                losses=[3.5, 2.8, 2.3],
            )


class TestPredictLoss:
    """测试 predict_loss 函数。"""

    def test_predict_scalar(self, expected_params):
        """测试标量预测。"""
        loss = predict_loss(1e9, 1e10, expected_params)
        assert isinstance(loss, float)
        assert loss > 0

    def test_predict_array(self, expected_params):
        """测试数组预测。"""
        N = np.array([1e7, 1e8, 1e9])
        D = np.array([1e9, 1e10, 1e11])
        losses = predict_loss(N, D, expected_params)

        assert isinstance(losses, np.ndarray)
        assert len(losses) == 3

    def test_missing_params(self):
        """测试缺少参数时抛出异常。"""
        incomplete_params = {"E": 1.5, "A": 400}  # 缺少 B, alpha, beta

        with pytest.raises(KeyError, match="Missing required parameters"):
            predict_loss(1e9, 1e10, incomplete_params)


class TestComputeOptimalAllocation:
    """测试 compute_optimal_allocation 函数。"""

    def test_basic_allocation(self, expected_params):
        """测试基本最优分配计算。"""
        result = compute_optimal_allocation(
            compute_budget=1e19,
            params=expected_params,
            n_search_points=1000,  # 减少搜索点数以加速测试
        )

        # 验证返回字段
        assert "n_params" in result
        assert "n_tokens" in result
        assert "loss" in result
        assert "compute_budget" in result
        assert "tokens_per_param" in result

        # 验证数值
        assert result["n_params"] > 0
        assert result["n_tokens"] > 0
        assert result["loss"] > 0
        assert result["compute_budget"] == 1e19
        assert result["tokens_per_param"] > 0

    def test_compute_constraint(self, expected_params):
        """测试计算预算约束 C = 6ND 被满足。"""
        C = 1e19
        result = compute_optimal_allocation(
            compute_budget=C,
            params=expected_params,
            n_search_points=1000,
        )

        # 验证 C ≈ 6 * N * D
        computed_C = 6 * result["n_params"] * result["n_tokens"]
        assert np.isclose(computed_C, C, rtol=1e-10)

    def test_different_budgets(self, expected_params):
        """测试不同预算下的最优分配。"""
        budgets = [1e18, 1e19, 1e20]

        for C in budgets:
            result = compute_optimal_allocation(
                compute_budget=C,
                params=expected_params,
                n_search_points=1000,
            )

            # 预算越大，最优 N 和 D 应越大
            assert result["n_params"] > 0
            assert result["n_tokens"] > 0

    def test_invalid_budget(self, expected_params):
        """测试无效预算时抛出异常。"""
        with pytest.raises(ValueError, match="must be positive"):
            compute_optimal_allocation(0, expected_params)

        with pytest.raises(ValueError, match="must be positive"):
            compute_optimal_allocation(-1e19, expected_params)

    def test_missing_params(self):
        """测试缺少参数时抛出异常。"""
        incomplete_params = {"E": 1.5}  # 缺少其他参数

        with pytest.raises(KeyError, match="Missing required parameters"):
            compute_optimal_allocation(1e19, incomplete_params)


# ─────────────────────────────────────────
# 工具函数测试
# ─────────────────────────────────────────

class TestComputeFlops:
    """测试 compute_flops 函数。"""

    def test_basic_computation(self):
        """测试基本 FLOPs 计算。"""
        # 1B 参数，10B tokens
        flops = compute_flops(1e9, 1e10)
        expected = 6 * 1e9 * 1e10  # 6e19
        assert flops == expected

    def test_array_input(self):
        """测试数组输入。"""
        N = np.array([1e7, 1e8, 1e9])
        D = np.array([1e9, 1e10, 1e11])
        flops = compute_flops(N, D)
        expected = 6 * N * D
        np.testing.assert_array_equal(flops, expected)


class TestDeriveTokensFromFlops:
    """测试 derive_tokens_from_flops 函数。"""

    def test_basic_derivation(self):
        """测试基本 token 数推导。"""
        C = 6e19  # FLOPs
        N = 1e9   # 参数
        D = derive_tokens_from_flops(C, N)
        expected = 1e10  # D = C / (6N)
        assert D == expected

    def test_round_trip(self):
        """测试往返一致性：C -> D -> C。"""
        N = 1e9
        D_original = 1e10
        C = compute_flops(N, D_original)
        D_derived = derive_tokens_from_flops(C, N)
        assert D_derived == D_original

    def test_invalid_params(self):
        """测试无效参数时抛出异常。"""
        with pytest.raises(ValueError, match="must be positive"):
            derive_tokens_from_flops(1e19, 0)

        with pytest.raises(ValueError, match="must be positive"):
            derive_tokens_from_flops(1e19, -1e9)


class TestLoadExperimentsFromCsv:
    """测试 load_experiments_from_csv 函数。"""

    def test_load_valid_csv(self, temp_csv_file):
        """测试加载有效 CSV 文件。"""
        data = load_experiments_from_csv(temp_csv_file)

        assert "n_params" in data
        assert "n_tokens" in data
        assert "loss" in data

        assert len(data["n_params"]) == 5
        assert len(data["n_tokens"]) == 5
        assert len(data["loss"]) == 5

        # 验证数值
        np.testing.assert_array_equal(
            data["n_params"],
            np.array([1e7, 3e7, 1e8, 1e8, 3e8])
        )

    def test_file_not_found(self):
        """测试文件不存在时抛出异常。"""
        with pytest.raises(FileNotFoundError):
            load_experiments_from_csv("/nonexistent/path.csv")


# ─────────────────────────────────────────
# 配置类测试
# ─────────────────────────────────────────

class TestFittingConfig:
    """测试 FittingConfig 配置类。"""

    def test_default_values(self):
        """测试默认值。"""
        config = FittingConfig()

        assert config.initial_guess == [1.5, 450.0, 2100.0, 0.35, 0.35]
        assert len(config.bounds) == 5
        assert config.epsilon == 1e-12
        assert config.method == "L-BFGS-B"

    def test_custom_values(self):
        """测试自定义值。"""
        config = FittingConfig(
            initial_guess=[2.0, 100.0, 100.0, 0.2, 0.2],
            epsilon=1e-10,
            method="SLSQP",
        )

        assert config.initial_guess == [2.0, 100.0, 100.0, 0.2, 0.2]
        assert config.epsilon == 1e-10
        assert config.method == "SLSQP"


# ─────────────────────────────────────────
# ChinchillaFitter 类测试
# ─────────────────────────────────────────

class TestChinchillaFitter:
    """测试 ChinchillaFitter 高级接口类。"""

    def test_init_default(self):
        """测试默认初始化。"""
        fitter = ChinchillaFitter()
        assert fitter.config is not None
        assert fitter.params is None

    def test_init_custom_config(self):
        """测试自定义配置初始化。"""
        config = FittingConfig(epsilon=1e-10)
        fitter = ChinchillaFitter(config=config)
        assert fitter.config.epsilon == 1e-10

    def test_load_data(self, sample_data):
        """测试数据加载。"""
        fitter = ChinchillaFitter()
        result = fitter.load_data(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        )

        # 验证链式调用返回 self
        assert result is fitter

    def test_load_from_csv(self, temp_csv_file):
        """测试从 CSV 加载。"""
        fitter = ChinchillaFitter()
        result = fitter.load_from_csv(temp_csv_file)

        assert result is fitter
        # 内部数据应已加载
        with pytest.raises(RuntimeError, match="Model not fitted yet"):
            fitter.predict(1e9, 1e10)  # 尚未拟合，应抛出异常

    def test_fit(self, sample_data):
        """测试拟合流程。"""
        fitter = ChinchillaFitter()
        fitter.load_data(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        )

        params = fitter.fit()

        assert "E" in params
        assert "A" in params
        assert "B" in params
        assert "alpha" in params
        assert "beta" in params

        # 拟合后 params 属性应可用
        assert fitter.params is params

    def test_predict_without_fit(self):
        """测试未拟合时预测抛出异常。"""
        fitter = ChinchillaFitter()

        with pytest.raises(RuntimeError, match="Model not fitted yet"):
            fitter.predict(1e9, 1e10)

    def test_predict_after_fit(self, sample_data):
        """测试拟合后预测。"""
        fitter = ChinchillaFitter()
        fitter.load_data(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        )
        fitter.fit()

        loss = fitter.predict(1e9, 1e10)
        assert isinstance(loss, float)
        assert loss > 0

    def test_compute_optimal_allocation_after_fit(self, sample_data):
        """测试拟合后计算最优分配。"""
        fitter = ChinchillaFitter()
        fitter.load_data(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        )
        fitter.fit()

        result = fitter.compute_optimal_allocation(1e19, n_search_points=1000)

        assert "n_params" in result
        assert "n_tokens" in result
        assert "loss" in result

    def test_compute_optimal_allocation_without_fit(self):
        """测试未拟合时计算最优分配抛出异常。"""
        fitter = ChinchillaFitter()

        with pytest.raises(RuntimeError, match="Model not fitted yet"):
            fitter.compute_optimal_allocation(1e19)

    def test_save_and_load_params(self, sample_data, tmp_path):
        """测试参数保存和加载。"""
        fitter = ChinchillaFitter()
        fitter.load_data(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        )
        original_params = fitter.fit()

        # 保存参数
        params_path = tmp_path / "params.json"
        fitter.save_params(params_path)

        assert params_path.exists()

        # 加载参数到新 fitter
        new_fitter = ChinchillaFitter()
        new_fitter.load_params(params_path)

        # 验证参数一致
        for key in original_params:
            assert new_fitter.params[key] == pytest.approx(original_params[key])

    def test_save_params_without_fit(self):
        """测试未拟合时保存参数抛出异常。"""
        fitter = ChinchillaFitter()

        with pytest.raises(RuntimeError, match="Model not fitted yet"):
            fitter.save_params("/tmp/params.json")

    def test_end_to_end_workflow(self, sample_data):
        """测试端到端工作流。"""
        # 链式调用
        fitter = ChinchillaFitter()
        params = fitter.load_data(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        ).fit()

        # 预测
        loss = fitter.predict(1e9, 1e10)
        assert loss > 0

        # 最优分配
        allocation = fitter.compute_optimal_allocation(1e19, n_search_points=1000)
        assert allocation["n_params"] > 0
        assert allocation["n_tokens"] > 0


# ─────────────────────────────────────────
# 集成测试
# ─────────────────────────────────────────

class TestIntegration:
    """集成测试：验证整个拟合流程。"""

    def test_full_pipeline(self, sample_data):
        """测试完整流水线：拟合 -> 预测 -> 最优分配。"""
        # 1. 拟合参数
        params = fit_chinchilla_params(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        )

        # 2. 使用拟合参数预测训练数据上的损失
        predicted_losses = predict_loss(
            sample_data["n_params"],
            sample_data["n_tokens"],
            params,
        )

        # 3. 计算预测与实际损失的相关系数（应较高）
        correlation = np.corrcoef(predicted_losses, sample_data["loss"])[0, 1]
        assert correlation > 0.5  # 至少中等正相关

        # 4. 计算最优分配
        for C in [1e18, 1e19, 1e20]:
            allocation = compute_optimal_allocation(C, params, n_search_points=1000)

            # 验证约束满足
            computed_C = 6 * allocation["n_params"] * allocation["n_tokens"]
            assert np.isclose(computed_C, C, rtol=1e-10)

    def test_consistency_with_paper_params(self, sample_data, expected_params):
        """测试使用论文参数预测与拟合结果的一致性。"""
        # 使用论文参数预测
        paper_predictions = predict_loss(
            sample_data["n_params"],
            sample_data["n_tokens"],
            expected_params,
        )

        # 拟合参数
        fitted_params = fit_chinchilla_params(
            sample_data["n_params"],
            sample_data["n_tokens"],
            sample_data["loss"],
        )

        # 使用拟合参数预测
        fitted_predictions = predict_loss(
            sample_data["n_params"],
            sample_data["n_tokens"],
            fitted_params,
        )

        # 两种预测应与实际损失有相似的相关性
        paper_corr = np.corrcoef(paper_predictions, sample_data["loss"])[0, 1]
        fitted_corr = np.corrcoef(fitted_predictions, sample_data["loss"])[0, 1]

        # 拟合结果不应比论文参数差太多（允许 0.2 的容差）
        assert fitted_corr > paper_corr - 0.2
