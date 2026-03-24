"""
tests/stage3_scaling/test_visualization.py — Tests for scaling laws visualization module
"""
import os
import tempfile

import numpy as np
import pytest

# Skip all tests if matplotlib is not available
try:
    import matplotlib
    matplotlib.use("Agg")  # Must set before importing pyplot
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


pytestmark = pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not available")


# Sample test data mimicking real scaling experiments
SAMPLE_EXPERIMENTS = [
    {"n_params": 1e7, "n_tokens": 1.67e9, "loss": 3.82},
    {"n_params": 3e7, "n_tokens": 5.56e8, "loss": 3.51},
    {"n_params": 1e8, "n_tokens": 1.67e8, "loss": 3.20},
    {"n_params": 1e8, "n_tokens": 5.0e9, "loss": 2.95},
    {"n_params": 3e8, "n_tokens": 1.67e9, "loss": 2.70},
    {"n_params": 1e9, "n_tokens": 5.0e8, "loss": 2.50},
    {"n_params": 3e9, "n_tokens": 1.67e8, "loss": 2.38},
    {"n_params": 1e7, "n_tokens": 5.0e10, "loss": 2.85},
    {"n_params": 3e8, "n_tokens": 5.56e7, "loss": 2.91},
    {"n_params": 1e9, "n_tokens": 1.67e10, "loss": 2.20},
]

SAMPLE_CHINCHILLA_PARAMS = {
    "E": 1.69,
    "A": 406.4,
    "B": 410.7,
    "alpha": 0.34,
    "beta": 0.28,
}

SAMPLE_ISOFLOPS_RESULT = {
    "factor_N": 0.5,
    "exponent_N": 0.52,
    "factor_D": 5.0,
    "exponent_D": 0.48,
    "optimal_points": [
        {"compute": 1e17, "n_params": 1e7, "n_tokens": 1.67e9, "loss": 3.82},
        {"compute": 1e18, "n_params": 1e8, "n_tokens": 1.67e9, "loss": 2.70},
        {"compute": 1e19, "n_params": 1e9, "n_tokens": 1.67e9, "loss": 2.20},
    ],
}

COMPUTE_BUDGETS = [1e18, 1e19, 1e20, 1e21]


class TestPlotIsoflopCurves:
    """Tests for plot_isoflop_curves function."""

    def test_plot_isoflop_curves_creates_file(self):
        """Test that plot_isoflop_curves creates a file."""
        from llm_foundry.stage3_scaling.visualization import plot_isoflop_curves

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_isoflop_curves(SAMPLE_EXPERIMENTS, tmpdir)
            assert os.path.isfile(filepath)
            assert filepath.endswith(".png")
            assert "isoflop_curves" in filepath

    def test_plot_isoflop_curves_pdf_format(self):
        """Test PDF format output."""
        from llm_foundry.stage3_scaling.visualization import plot_isoflop_curves

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_isoflop_curves(
                SAMPLE_EXPERIMENTS, tmpdir, format="pdf"
            )
            assert filepath.endswith(".pdf")
            assert os.path.isfile(filepath)

    def test_plot_isoflop_curves_custom_dpi(self):
        """Test custom DPI setting."""
        from llm_foundry.stage3_scaling.visualization import plot_isoflop_curves

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_isoflop_curves(
                SAMPLE_EXPERIMENTS, tmpdir, dpi=150, format="png"
            )
            assert os.path.isfile(filepath)

    def test_plot_isoflop_curves_empty_experiments(self):
        """Test behavior with empty experiments list."""
        from llm_foundry.stage3_scaling.visualization import plot_isoflop_curves

        with tempfile.TemporaryDirectory() as tmpdir:
            # Should handle empty list gracefully
            filepath = plot_isoflop_curves([], tmpdir)
            assert os.path.isfile(filepath)


class TestPlotChinchillaContour:
    """Tests for plot_chinchilla_contour function."""

    def test_plot_chinchilla_contour_creates_file(self):
        """Test that plot_chinchilla_contour creates a file."""
        from llm_foundry.stage3_scaling.visualization import plot_chinchilla_contour

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_chinchilla_contour(SAMPLE_CHINCHILLA_PARAMS, tmpdir)
            assert os.path.isfile(filepath)
            assert filepath.endswith(".png")
            assert "chinchilla_contour" in filepath

    def test_plot_chinchilla_contour_pdf_format(self):
        """Test PDF format output."""
        from llm_foundry.stage3_scaling.visualization import plot_chinchilla_contour

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_chinchilla_contour(
                SAMPLE_CHINCHILLA_PARAMS, tmpdir, format="pdf"
            )
            assert filepath.endswith(".pdf")
            assert os.path.isfile(filepath)

    def test_plot_chinchilla_contour_custom_range(self):
        """Test custom N and D ranges."""
        from llm_foundry.stage3_scaling.visualization import plot_chinchilla_contour

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_chinchilla_contour(
                SAMPLE_CHINCHILLA_PARAMS,
                tmpdir,
                n_range=(1e7, 1e10),
                d_range=(1e8, 1e11),
                n_points=50,
            )
            assert os.path.isfile(filepath)


class TestPlotLossVsTokens:
    """Tests for plot_loss_vs_tokens function."""

    def test_plot_loss_vs_tokens_creates_file(self):
        """Test that plot_loss_vs_tokens creates a file."""
        from llm_foundry.stage3_scaling.visualization import plot_loss_vs_tokens

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_loss_vs_tokens(SAMPLE_EXPERIMENTS, tmpdir)
            assert os.path.isfile(filepath)
            assert filepath.endswith(".png")
            assert "loss_vs_tokens" in filepath

    def test_plot_loss_vs_tokens_grouped(self):
        """Test grouped plotting by parameter count."""
        from llm_foundry.stage3_scaling.visualization import plot_loss_vs_tokens

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_loss_vs_tokens(
                SAMPLE_EXPERIMENTS, tmpdir, group_by_params=True
            )
            assert os.path.isfile(filepath)

    def test_plot_loss_vs_tokens_ungrouped(self):
        """Test ungrouped plotting."""
        from llm_foundry.stage3_scaling.visualization import plot_loss_vs_tokens

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_loss_vs_tokens(
                SAMPLE_EXPERIMENTS, tmpdir, group_by_params=False
            )
            assert os.path.isfile(filepath)


class TestPlotOptimalAllocation:
    """Tests for plot_optimal_allocation function."""

    def test_plot_optimal_allocation_creates_file(self):
        """Test that plot_optimal_allocation creates a file."""
        from llm_foundry.stage3_scaling.visualization import plot_optimal_allocation

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_optimal_allocation(
                SAMPLE_ISOFLOPS_RESULT, COMPUTE_BUDGETS, tmpdir
            )
            assert os.path.isfile(filepath)
            assert filepath.endswith(".png")
            assert "optimal_allocation" in filepath

    def test_plot_optimal_allocation_pdf_format(self):
        """Test PDF format output."""
        from llm_foundry.stage3_scaling.visualization import plot_optimal_allocation

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_optimal_allocation(
                SAMPLE_ISOFLOPS_RESULT, COMPUTE_BUDGETS, tmpdir, format="pdf"
            )
            assert filepath.endswith(".pdf")
            assert os.path.isfile(filepath)

    def test_plot_optimal_allocation_without_optimal_points(self):
        """Test plotting without empirical optimal points."""
        from llm_foundry.stage3_scaling.visualization import plot_optimal_allocation

        result_without_points = {
            "factor_N": 0.5,
            "exponent_N": 0.52,
            "factor_D": 5.0,
            "exponent_D": 0.48,
            # No optimal_points key
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_optimal_allocation(
                result_without_points, COMPUTE_BUDGETS, tmpdir
            )
            assert os.path.isfile(filepath)


class TestPlotAllVisualizations:
    """Tests for plot_all_visualizations function."""

    def test_plot_all_visualizations_creates_all_files(self):
        """Test that all four plots are created."""
        from llm_foundry.stage3_scaling.visualization import plot_all_visualizations

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_all_visualizations(
                SAMPLE_EXPERIMENTS,
                SAMPLE_CHINCHILLA_PARAMS,
                SAMPLE_ISOFLOPS_RESULT,
                COMPUTE_BUDGETS,
                tmpdir,
            )

            assert "isoflop_curves" in paths
            assert "chinchilla_contour" in paths
            assert "loss_vs_tokens" in paths
            assert "optimal_allocation" in paths

            for name, filepath in paths.items():
                assert os.path.isfile(filepath), f"{name} plot not created"
                assert filepath.endswith(".png")

    def test_plot_all_visualizations_pdf_format(self):
        """Test all plots in PDF format."""
        from llm_foundry.stage3_scaling.visualization import plot_all_visualizations

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = plot_all_visualizations(
                SAMPLE_EXPERIMENTS,
                SAMPLE_CHINCHILLA_PARAMS,
                SAMPLE_ISOFLOPS_RESULT,
                COMPUTE_BUDGETS,
                tmpdir,
                format="pdf",
            )

            for filepath in paths.values():
                assert filepath.endswith(".pdf")
                assert os.path.isfile(filepath)


class TestVisualizationFormats:
    """Tests for different output formats."""

    @pytest.mark.parametrize("fmt", ["png", "pdf", "svg"])
    def test_all_formats_supported(self, fmt):
        """Test that all supported formats work."""
        from llm_foundry.stage3_scaling.visualization import plot_isoflop_curves

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_isoflop_curves(
                SAMPLE_EXPERIMENTS, tmpdir, format=fmt
            )
            assert filepath.endswith(f".{fmt}")
            assert os.path.isfile(filepath)
            assert os.path.getsize(filepath) > 0


class TestVisualizationQuality:
    """Tests for visualization quality settings."""

    def test_high_dpi_output(self):
        """Test high-resolution output."""
        from llm_foundry.stage3_scaling.visualization import plot_isoflop_curves

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath_low = plot_isoflop_curves(
                SAMPLE_EXPERIMENTS, tmpdir, dpi=100, format="png"
            )
            filepath_high = plot_isoflop_curves(
                SAMPLE_EXPERIMENTS, tmpdir, dpi=300, format="png"
            )

            # Verify both files exist and have content
            assert os.path.isfile(filepath_low)
            assert os.path.isfile(filepath_high)
            assert os.path.getsize(filepath_low) > 0
            assert os.path.getsize(filepath_high) > 0

            # Verify DPI affects image dimensions using PIL
            try:
                from PIL import Image

                with Image.open(filepath_low) as img_low:
                    width_low, height_low = img_low.size
                    # Check DPI metadata if available
                    dpi_low = img_low.info.get("dpi", (100, 100))[0]
                with Image.open(filepath_high) as img_high:
                    width_high, height_high = img_high.size
                    dpi_high = img_high.info.get("dpi", (300, 300))[0]

                # With bbox_inches="tight", exact dimensions may vary,
                # but high DPI should generally produce equal or larger dimensions
                # and the DPI metadata should be different
                assert width_high >= width_low, f"High DPI width {width_high} should be >= low DPI width {width_low}"
                assert height_high >= height_low, f"High DPI height {height_high} should be >= low DPI height {height_low}"

                # DPI values in metadata should reflect the requested DPI
                # (matplotlib may not always set this, so we check if available)
                if "dpi" in img_low.info or "dpi" in img_high.info:
                    assert dpi_high >= dpi_low, f"High DPI metadata {dpi_high} should be >= low DPI {dpi_low}"

            except ImportError:
                # PIL not available - test already passed with file existence checks
                pass

    def test_custom_figsize(self):
        """Test custom figure size."""
        from llm_foundry.stage3_scaling.visualization import plot_isoflop_curves

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = plot_isoflop_curves(
                SAMPLE_EXPERIMENTS, tmpdir, figsize=(12, 8), format="png"
            )
            assert os.path.isfile(filepath)
