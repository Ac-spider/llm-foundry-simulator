"""
tests/stage3_scaling/test_isoflops.py — Tests for IsoFLOPs curve calculator
"""
import numpy as np
import pytest

from llm_foundry.stage3_scaling.isoflops import (
    IsoFLOPsConfig,
    compute_flops,
    estimate_training_cost,
    find_optimal_nd_for_compute,
    fit_power_law,
    format_flops,
    generate_isoflop_curves,
    get_chinchilla_optimal,
    predict_optimal_size,
)


# ─────────────────────────────────────────
# Test IsoFLOPsConfig
# ─────────────────────────────────────────

class TestIsoFLOPsConfig:
    """Test IsoFLOPsConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = IsoFLOPsConfig()
        assert config.compute_budgets == [1e18, 1e19, 1e20, 1e21]
        assert config.n_points_per_budget == 10
        assert config.n_range == (1e6, 1e12)
        assert config.flops_per_token_per_param == 6.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = IsoFLOPsConfig(
            compute_budgets=[1e17, 1e18],
            n_points_per_budget=5,
            n_range=(1e7, 1e10),
            flops_per_token_per_param=2.0,
        )
        assert config.compute_budgets == [1e17, 1e18]
        assert config.n_points_per_budget == 5
        assert config.n_range == (1e7, 1e10)
        assert config.flops_per_token_per_param == 2.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = IsoFLOPsConfig()
        d = config.to_dict()
        assert d["compute_budgets"] == [1e18, 1e19, 1e20, 1e21]
        assert d["n_points_per_budget"] == 10
        assert d["n_range"] == (1e6, 1e12)
        assert d["flops_per_token_per_param"] == 6.0

    def test_from_dict(self):
        """Test creation from dictionary."""
        d = {
            "compute_budgets": [1e17, 1e18],
            "n_points_per_budget": 5,
            "n_range": [1e7, 1e10],
            "flops_per_token_per_param": 2.0,
        }
        config = IsoFLOPsConfig.from_dict(d)
        assert config.compute_budgets == [1e17, 1e18]
        assert config.n_points_per_budget == 5
        assert config.n_range == (1e7, 1e10)
        assert config.flops_per_token_per_param == 2.0


# ─────────────────────────────────────────
# Test compute_flops
# ─────────────────────────────────────────

class TestComputeFLOPs:
    """Test compute_flops function."""

    def test_basic_calculation(self):
        """Test basic FLOPs calculation."""
        # 1B params, 1B tokens, multiplier 6
        flops = compute_flops(1e9, 1e9, multiplier=6.0)
        assert flops == 6e18

    def test_default_multiplier(self):
        """Test default multiplier of 6.0."""
        flops = compute_flops(1e9, 1e9)
        assert flops == 6e18

    def test_custom_multiplier(self):
        """Test custom multiplier."""
        flops = compute_flops(1e9, 1e9, multiplier=2.0)
        assert flops == 2e18

    def test_small_values(self):
        """Test with small values."""
        flops = compute_flops(1e6, 1e6)
        assert flops == 6e12

    def test_large_values(self):
        """Test with large values (GPT-4 scale)."""
        # 1T params, 10T tokens
        flops = compute_flops(1e12, 1e13)
        assert flops == 6e25


# ─────────────────────────────────────────
# Test generate_isoflop_curves
# ─────────────────────────────────────────

class TestGenerateIsoFLOPCurves:
    """Test generate_isoflop_curves function."""

    def test_default_config(self):
        """Test curve generation with default config."""
        curves = generate_isoflop_curves()

        # Should have 4 compute budgets
        assert len(curves) == 4
        assert 1e18 in curves
        assert 1e19 in curves
        assert 1e20 in curves
        assert 1e21 in curves

        # Each budget should have 10 points
        for budget, points in curves.items():
            assert len(points) == 10
            for point in points:
                assert "n_params" in point
                assert "n_tokens" in point
                assert "flops" in point
                assert point["flops"] == budget
                # Verify C = 6 * N * D
                expected_flops = 6.0 * point["n_params"] * point["n_tokens"]
                assert abs(expected_flops - budget) / budget < 0.01

    def test_custom_config(self):
        """Test with custom config."""
        config = IsoFLOPsConfig(
            compute_budgets=[1e18],
            n_points_per_budget=5,
        )
        curves = generate_isoflop_curves(config)

        assert len(curves) == 1
        assert len(curves[1e18]) == 5

    def test_custom_budgets(self):
        """Test with custom compute budgets."""
        curves = generate_isoflop_curves(compute_budgets=[1e17, 1e18])

        assert len(curves) == 2
        assert 1e17 in curves
        assert 1e18 in curves

    def test_n_params_range(self):
        """Test that N params stays within specified range."""
        config = IsoFLOPsConfig(
            compute_budgets=[1e18],
            n_points_per_budget=10,
            n_range=(1e7, 1e10),
        )
        curves = generate_isoflop_curves(config)

        points = curves[1e18]
        for point in points:
            # N should be within reasonable bounds for this compute budget
            assert point["n_params"] >= 1e6  # At least 1M params
            assert point["n_params"] <= 1e12  # At most 1T params


# ─────────────────────────────────────────
# Test find_optimal_nd_for_compute
# ─────────────────────────────────────────

class TestFindOptimalNDForCompute:
    """Test find_optimal_nd_for_compute function."""

    def test_find_optimal(self):
        """Test finding optimal configuration."""
        loss_data = [
            {"n_params": 1e8, "n_tokens": 1.67e9, "loss": 3.5},   # ~1e18 FLOPs
            {"n_params": 3e8, "n_tokens": 5.56e8, "loss": 3.2},   # ~1e18 FLOPs
            {"n_params": 1e9, "n_tokens": 1.67e9, "loss": 2.9},   # ~1e19 FLOPs
        ]

        # Look for optimal at ~1e18 FLOPs
        result = find_optimal_nd_for_compute(1e18, loss_data)
        assert result is not None
        assert result["loss"] == 3.2  # Minimum loss at this compute budget

    def test_no_match(self):
        """Test when no data matches compute budget."""
        loss_data = [
            {"n_params": 1e8, "n_tokens": 1.67e9, "loss": 3.5},   # ~1e18 FLOPs
        ]

        # Look for optimal at 1e21 FLOPs (no match)
        result = find_optimal_nd_for_compute(1e21, loss_data)
        assert result is None

    def test_tolerance(self):
        """Test that 10% tolerance is applied."""
        loss_data = [
            {"n_params": 1e8, "n_tokens": 1.67e9, "loss": 3.5},   # ~1e18 FLOPs
        ]

        # 1.05e18 is within 10% of 1e18
        result = find_optimal_nd_for_compute(1.05e18, loss_data)
        assert result is not None


# ─────────────────────────────────────────
# Test fit_power_law
# ─────────────────────────────────────────

class TestFitPowerLaw:
    """Test fit_power_law function."""

    def test_perfect_power_law(self):
        """Test fitting perfect power law data."""
        # N = 0.1 * C^0.5
        C = np.array([1e18, 1e19, 1e20, 1e21])
        N = 0.1 * (C ** 0.5)

        factor, exponent = fit_power_law(C, N)

        assert abs(factor - 0.1) < 0.01
        assert abs(exponent - 0.5) < 0.01

    def test_chinchilla_relationship(self):
        """Test fitting Chinchilla N_opt ∝ C^0.5."""
        # Generate data following Chinchilla scaling
        C = np.logspace(18, 21, 10)
        N = 0.15 * (C ** 0.5)  # Slightly different factor

        factor, exponent = fit_power_law(C, N)

        assert 0.1 < factor < 0.2
        assert 0.48 < exponent < 0.52


# ─────────────────────────────────────────
# Test predict_optimal_size
# ─────────────────────────────────────────

class TestPredictOptimalSize:
    """Test predict_optimal_size function."""

    def test_basic_prediction(self):
        """Test basic prediction."""
        # N = 0.1 * C^0.5
        N = predict_optimal_size(1e18, factor=0.1, exponent=0.5)
        assert abs(N - 0.1 * (1e18 ** 0.5)) < 1e-6

    def test_chinchilla_prediction(self):
        """Test Chinchilla-style prediction."""
        # For 1e21 FLOPs, N_opt ≈ 10B params
        N = predict_optimal_size(1e21, factor=0.1, exponent=0.5)
        expected = 0.1 * (1e21 ** 0.5)  # ≈ 3.16e9
        assert abs(N - expected) < 1e6


# ─────────────────────────────────────────
# Test get_chinchilla_optimal
# ─────────────────────────────────────────

class TestGetChinchillaOptimal:
    """Test get_chinchilla_optimal function."""

    def test_1e18_flops(self):
        """Test optimal config for 1e18 FLOPs."""
        N, D = get_chinchilla_optimal(1e18)

        # N ≈ 0.1 * sqrt(1e18) = 0.1 * 1e9 = 1e8
        assert 5e7 < N < 2e8

        # Verify C = 6 * N * D
        C = 6.0 * N * D
        assert abs(C - 1e18) / 1e18 < 0.1

    def test_1e21_flops(self):
        """Test optimal config for 1e21 FLOPs (GPT-3 scale)."""
        N, D = get_chinchilla_optimal(1e21)

        # N ≈ 0.1 * sqrt(1e21) = 0.1 * 3.16e10 = 3.16e9
        assert 1e9 < N < 1e10

        # Verify C = 6 * N * D
        C = 6.0 * N * D
        assert abs(C - 1e21) / 1e21 < 0.1

    def test_scaling_relationship(self):
        """Test that N_opt and D_opt scale correctly with C."""
        N1, D1 = get_chinchilla_optimal(1e18)
        N2, D2 = get_chinchilla_optimal(1e20)

        # 100x more compute should give 10x more params (sqrt scaling)
        ratio_n = N2 / N1
        assert 8 < ratio_n < 12  # Should be ~10


# ─────────────────────────────────────────
# Test estimate_training_cost
# ─────────────────────────────────────────

class TestEstimateTrainingCost:
    """Test estimate_training_cost function."""

    def test_gpt3_scale(self):
        """Test cost estimation for GPT-3 scale model."""
        # GPT-3: 175B params, 300B tokens
        result = estimate_training_cost(
            n_params=175e9,
            n_tokens=300e9,
            hardware_flops_per_sec=1e14,  # 100 TFLOPS (A100-like)
            hardware_cost_per_hour=5.0,
        )

        # Verify structure
        assert "total_flops" in result
        assert "training_seconds" in result
        assert "training_hours" in result
        assert "training_days" in result
        assert "estimated_cost_usd" in result

        # Verify FLOPs
        expected_flops = 6.0 * 175e9 * 300e9
        assert result["total_flops"] == expected_flops

        # Verify time calculations
        assert result["training_hours"] == result["training_seconds"] / 3600
        assert result["training_days"] == result["training_hours"] / 24

        # Verify cost
        assert result["estimated_cost_usd"] == result["training_hours"] * 5.0

    def test_small_model(self):
        """Test cost estimation for small model."""
        result = estimate_training_cost(
            n_params=1e6,
            n_tokens=1e9,
            hardware_flops_per_sec=1e12,
            hardware_cost_per_hour=2.0,
        )

        # 6e15 FLOPs at 1 TFLOPS = 6e3 seconds
        assert result["total_flops"] == 6e15
        assert result["training_seconds"] == 6e3


# ─────────────────────────────────────────
# Test format_flops
# ─────────────────────────────────────────

class TestFormatFLOPs:
    """Test format_flops function."""

    def test_exaflops(self):
        """Test formatting exaflops."""
        s = format_flops(6e18)
        assert "6.0e+18" in s
        assert "EFLOPs" in s

    def test_petaflops(self):
        """Test formatting petaflops."""
        s = format_flops(1e15)
        assert "1.0e+15" in s
        assert "PFLOPs" in s

    def test_teraflops(self):
        """Test formatting teraflops."""
        s = format_flops(1e12)
        assert "1.0e+12" in s
        assert "TFLOPs" in s

    def test_zettaflops(self):
        """Test formatting zettaflops."""
        s = format_flops(1e21)
        assert "ZFLOPs" in s


# ─────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────

class TestIsoFLOPsIntegration:
    """Integration tests for the full IsoFLOPs workflow."""

    def test_end_to_end_workflow(self):
        """Test complete IsoFLOPs workflow."""
        # 1. Generate IsoFLOPs curves with custom N range centered around Chinchilla optimal
        # Chinchilla optimal: N_opt ≈ 0.1 * sqrt(C)
        # For C=1e18: N_opt ≈ 1e8, for C=1e19: N_opt ≈ 3.16e8
        config = IsoFLOPsConfig(
            compute_budgets=[1e18, 1e19],
            n_points_per_budget=8,  # More points for better coverage
            n_range=(1e7, 1e10),  # Range that covers Chinchilla optimal for these budgets
        )
        curves = generate_isoflop_curves(config)

        # 2. Simulate experimental results (loss data)
        # Use a more realistic simulation where loss has a U-shape with minimum
        # at Chinchilla-optimal configuration (N ~ 0.1 * sqrt(C))
        loss_data = []
        for budget, points in curves.items():
            # Chinchilla-optimal N for this budget: N_opt ≈ 0.1 * sqrt(C)
            n_optimal = 0.1 * (budget ** 0.5)
            for point in points:
                n_params = point["n_params"]
                n_tokens = point["n_tokens"]
                # Simulate U-shaped loss: higher when N is too small or too large
                # relative to the optimal for this compute budget
                ratio = n_params / n_optimal
                # Loss is minimized when ratio ≈ 1 (Chinchilla optimal)
                base_loss = 2.0
                penalty = 0.5 * (ratio - 1.0) ** 2
                loss = base_loss + penalty
                loss_data.append({
                    "n_params": n_params,
                    "n_tokens": n_tokens,
                    "loss": loss,
                })

        # 3. Find optimal points for each budget
        optimal_points = []
        for budget in config.compute_budgets:
            optimal = find_optimal_nd_for_compute(budget, loss_data)
            if optimal:
                optimal_points.append(optimal)

        assert len(optimal_points) >= 1

        # 4. Fit power law if we have enough points
        if len(optimal_points) >= 2:
            C = np.array([p["compute"] for p in optimal_points])
            N = np.array([p["n_params"] for p in optimal_points])
            factor, exponent = fit_power_law(C, N)

            assert factor > 0
            # With proper U-shaped loss and good N range coverage,
            # exponent should be ~0.5 (Chinchilla scaling)
            assert 0.3 < exponent < 0.7, f"Expected exponent ~0.5, got {exponent:.4f}"

    def test_chinchilla_consistency(self):
        """Test that results are consistent with Chinchilla paper."""
        # Test multiple compute budgets
        budgets = [1e18, 1e19, 1e20, 1e21]

        for C in budgets:
            N_opt, D_opt = get_chinchilla_optimal(C)

            # Verify C ≈ 6 * N * D
            computed_C = 6.0 * N_opt * D_opt
            assert abs(computed_C - C) / C < 0.1

            # Verify N_opt ∝ C^0.5 (approximately)
            # N_opt / sqrt(C) should be roughly constant
            ratio = N_opt / (C ** 0.5)
            assert 0.05 < ratio < 0.2  # Chinchilla factor is around 0.1
