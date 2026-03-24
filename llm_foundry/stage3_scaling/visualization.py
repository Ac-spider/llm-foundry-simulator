"""
llm_foundry/stage3_scaling/visualization.py — Scaling Laws Visualization Module

Provides paper-quality plotting functions for scaling laws analysis:
    - IsoFLOPs curves (loss vs params at fixed compute)
    - Chinchilla loss contours (L(N, D) heatmap)
    - Loss vs tokens curves
    - Optimal N-D allocation lines

All plots support log-scale axes and publication-quality output (PNG/PDF).
"""
from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib
matplotlib.use("Agg")  # Headless backend for CLI/server environments
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator

if TYPE_CHECKING:
    from typing import Sequence


def plot_isoflop_curves(
    experiments: list[dict],
    save_path: str,
    figsize: tuple[int, int] = (10, 7),
    dpi: int = 300,
    format: str = "png",
) -> str:
    """
    Plot IsoFLOPs curves showing loss vs model parameters at fixed compute budgets.

    Each curve represents experiments with similar compute budgets (C = 6 * N * D),
    showing how loss varies with model size when compute is held approximately constant.

    Args:
        experiments: List of experiment dicts with keys:
            - n_params (float): Model parameter count
            - n_tokens (float): Training token count
            - loss (float): Validation loss
        save_path: Directory to save the plot
        figsize: Figure size in inches (width, height)
        dpi: Resolution for raster formats
        format: Output format ('png', 'pdf', 'svg', 'jpg')

    Returns:
        str: Full path to saved plot file
    """
    # Group experiments by compute budget (rounded to nearest order of magnitude)
    from collections import defaultdict
    budget_groups: dict[float, list[dict]] = defaultdict(list)

    for exp in experiments:
        compute = 6.0 * exp["n_params"] * exp["n_tokens"]
        # Round to nearest integer order of magnitude for grouping
        log_c = round(np.log10(compute), 0)
        budget_groups[log_c].append(exp)

    fig, ax = plt.subplots(figsize=figsize)

    # Color map for different compute budgets
    colors = plt.cm.viridis(np.linspace(0, 1, len(budget_groups)))

    for idx, (log_c, group) in enumerate(sorted(budget_groups.items())):
        compute_label = 10 ** log_c
        params = [g["n_params"] for g in group]
        losses = [g["loss"] for g in group]

        # Sort by params for smooth curves
        sorted_data = sorted(zip(params, losses))
        params_sorted, losses_sorted = zip(*sorted_data) if sorted_data else ([], [])

        ax.scatter(
            params_sorted,
            losses_sorted,
            color=colors[idx],
            s=80,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.plot(
            params_sorted,
            losses_sorted,
            color=colors[idx],
            linestyle="--",
            linewidth=1.5,
            alpha=0.8,
            label=f"C ≈ {compute_label:.0e} FLOPs",
            zorder=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Model Parameters N", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Loss", fontsize=12, fontweight="bold")
    ax.set_title("IsoFLOPs Curves: Loss vs Model Size", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Add annotation
    ax.text(
        0.02, 0.98,
        "Each curve: fixed compute budget\nOptimal: valley of U-shaped curves",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
    )

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    filename = f"isoflop_curves.{format}"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight", format=format)
    plt.close(fig)

    return filepath


def plot_chinchilla_contour(
    chinchilla_params: dict,
    save_path: str,
    n_range: tuple[float, float] = (1e6, 1e12),
    d_range: tuple[float, float] = (1e6, 1e13),
    figsize: tuple[int, int] = (11, 9),
    dpi: int = 300,
    format: str = "png",
    n_points: int = 200,
) -> str:
    """
    Plot Chinchilla loss contour map L(N, D) = E + A/N^α + B/D^β.

    Creates a heatmap showing how loss varies across the (N, D) parameter space,
    with contour lines indicating regions of equal loss.

    Args:
        chinchilla_params: Dict with fitted parameters:
            - E (float): Irreducible loss
            - A (float): Capacity penalty coefficient
            - B (float): Data penalty coefficient
            - alpha (float): Capacity exponent
            - beta (float): Data exponent
        save_path: Directory to save the plot
        n_range: (min, max) for parameter count axis
        d_range: (min, max) for token count axis
        figsize: Figure size in inches
        dpi: Resolution for raster formats
        format: Output format ('png', 'pdf', 'svg')
        n_points: Number of grid points per axis

    Returns:
        str: Full path to saved plot file
    """
    E = chinchilla_params["E"]
    A = chinchilla_params["A"]
    B = chinchilla_params["B"]
    alpha = chinchilla_params["alpha"]
    beta = chinchilla_params["beta"]

    # Create log-spaced grid
    N = np.logspace(np.log10(n_range[0]), np.log10(n_range[1]), n_points)
    D = np.logspace(np.log10(d_range[0]), np.log10(d_range[1]), n_points)
    N_grid, D_grid = np.meshgrid(N, D)

    # Compute loss surface: L(N, D) = E + A/N^α + B/D^β
    epsilon = 1e-12
    L_grid = E + A / (N_grid ** alpha + epsilon) + B / (D_grid ** beta + epsilon)

    fig, ax = plt.subplots(figsize=figsize)

    # Use log norm for better visualization across orders of magnitude
    vmin, vmax = np.percentile(L_grid, [1, 99])
    contour = ax.contourf(
        N_grid, D_grid, L_grid,
        levels=50,
        cmap="viridis_r",
        norm=LogNorm(vmin=max(vmin, 1.0), vmax=vmax),
    )

    # Add contour lines
    contour_lines = ax.contour(
        N_grid, D_grid, L_grid,
        levels=10,
        colors="white",
        linewidths=0.5,
        alpha=0.6,
    )
    ax.clabel(contour_lines, inline=True, fontsize=8, fmt="%.2f")

    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label("Loss L(N, D)", fontsize=11, fontweight="bold")

    # Add compute constraint lines (C = 6ND)
    compute_budgets = [1e18, 1e19, 1e20, 1e21]
    for C in compute_budgets:
        # D = C / (6N)
        D_constraint = C / (6 * N)
        # Only plot within visible range
        mask = (D_constraint >= d_range[0]) & (D_constraint <= d_range[1])
        if mask.any():
            ax.plot(
                N[mask],
                D_constraint[mask],
                "w--",
                linewidth=1.5,
                alpha=0.8,
                label=f"C={C:.0e}" if C == compute_budgets[0] else "",
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Model Parameters N", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Tokens D", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Chinchilla Loss Contours\n"
        f"E={E:.3f}, A={A:.1f}, α={alpha:.3f}, B={B:.1f}, β={beta:.3f}",
        fontsize=13,
        fontweight="bold",
    )

    # Add annotation box with parameters
    ax.text(
        0.02, 0.98,
        f"L(N,D) = E + A/N^α + B/D^β\n"
        f"White dashed lines: compute constraints\n"
        f"C = 6ND (FLOPs)",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    filename = f"chinchilla_contour.{format}"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight", format=format)
    plt.close(fig)

    return filepath


def plot_loss_vs_tokens(
    experiments: list[dict],
    save_path: str,
    group_by_params: bool = True,
    figsize: tuple[int, int] = (10, 7),
    dpi: int = 300,
    format: str = "png",
) -> str:
    """
    Plot loss as a function of training tokens for different model sizes.

    Shows how validation loss decreases with more training data,
    with separate curves for different model sizes.

    Args:
        experiments: List of experiment dicts with keys:
            - n_params (float): Model parameter count
            - n_tokens (float): Training token count
            - loss (float): Validation loss
        save_path: Directory to save the plot
        group_by_params: If True, group by parameter count; else plot all together
        figsize: Figure size in inches
        dpi: Resolution for raster formats
        format: Output format ('png', 'pdf', 'svg')

    Returns:
        str: Full path to saved plot file
    """
    fig, ax = plt.subplots(figsize=figsize)

    if group_by_params:
        # Group by rounded parameter count (nearest order of magnitude)
        from collections import defaultdict
        param_groups: dict[float, list[dict]] = defaultdict(list)

        for exp in experiments:
            log_n = round(np.log10(exp["n_params"]), 0)
            param_groups[log_n].append(exp)

        colors = plt.cm.plasma(np.linspace(0, 1, len(param_groups)))

        for idx, (log_n, group) in enumerate(sorted(param_groups.items())):
            n_label = 10 ** log_n
            tokens = [g["n_tokens"] for g in group]
            losses = [g["loss"] for g in group]

            sorted_data = sorted(zip(tokens, losses))
            tokens_sorted, losses_sorted = zip(*sorted_data) if sorted_data else ([], [])

            ax.scatter(
                tokens_sorted,
                losses_sorted,
                color=colors[idx],
                s=80,
                alpha=0.7,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            ax.plot(
                tokens_sorted,
                losses_sorted,
                color=colors[idx],
                linestyle="-",
                linewidth=2,
                alpha=0.8,
                label=f"N ≈ {n_label:.0e} params",
                zorder=2,
            )

        ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    else:
        tokens = [e["n_tokens"] for e in experiments]
        losses = [e["loss"] for e in experiments]
        ax.scatter(tokens, losses, s=80, alpha=0.7, c="#1f77b4", edgecolors="white", linewidths=0.5)

    ax.set_xscale("log")
    ax.set_xlabel("Training Tokens D", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Loss", fontsize=12, fontweight="bold")
    ax.set_title("Loss vs Training Tokens", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    filename = f"loss_vs_tokens.{format}"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight", format=format)
    plt.close(fig)

    return filepath


def plot_optimal_allocation(
    isoflops_result: dict,
    compute_budgets: Sequence[float],
    save_path: str,
    figsize: tuple[int, int] = (12, 5),
    dpi: int = 300,
    format: str = "png",
) -> str:
    """
    Plot optimal N-D allocation lines derived from IsoFLOPs analysis.

    Shows how optimal model size and training tokens scale with compute budget,
    following the power laws: N_opt = A_N * C^a, D_opt = B_D * C^b.

    Args:
        isoflops_result: Dict with fitted power law parameters:
            - factor_N (float): A_N coefficient for N_opt
            - exponent_N (float): Exponent a for N_opt
            - factor_D (float): B_D coefficient for D_opt
            - exponent_D (float): Exponent b for D_opt
            - optimal_points (list): Observed optimal points
        compute_budgets: Sequence of compute budgets for prediction
        save_path: Directory to save the plot
        figsize: Figure size in inches
        dpi: Resolution for raster formats
        format: Output format ('png', 'pdf', 'svg')

    Returns:
        str: Full path to saved plot file
    """
    factor_N = isoflops_result["factor_N"]
    exponent_N = isoflops_result["exponent_N"]
    factor_D = isoflops_result["factor_D"]
    exponent_D = isoflops_result["exponent_D"]
    optimal_points = isoflops_result.get("optimal_points", [])

    # Generate smooth curves for prediction
    C_smooth = np.logspace(
        np.log10(min(compute_budgets) * 0.5),
        np.log10(max(compute_budgets) * 2),
        500,
    )
    N_pred = factor_N * (C_smooth ** exponent_N)
    D_pred = factor_D * (C_smooth ** exponent_D)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot: N_opt vs C
    ax = axes[0]
    if optimal_points:
        C_obs = [p["compute"] for p in optimal_points]
        N_obs = [p["n_params"] for p in optimal_points]
        ax.scatter(
            C_obs, N_obs,
            color="#1f77b4",
            s=100,
            alpha=0.8,
            edgecolors="white",
            linewidths=1,
            label="Empirical Optima",
            zorder=3,
        )

    ax.plot(
        C_smooth, N_pred,
        color="#d62728",
        linestyle="--",
        linewidth=2.5,
        label=f"N = {factor_N:.2e} × C^{exponent_N:.3f}",
        zorder=2,
    )

    # Mark specific compute budgets
    N_at_budgets = [factor_N * (C ** exponent_N) for C in compute_budgets]
    ax.scatter(
        compute_budgets, N_at_budgets,
        color="#ff7f0e",
        s=150,
        marker="*",
        edgecolors="black",
        linewidths=1,
        label="Target Budgets",
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget C (FLOPs)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Optimal Parameters N", fontsize=11, fontweight="bold")
    ax.set_title("Optimal Model Size vs Compute", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Right plot: D_opt vs C
    ax = axes[1]
    if optimal_points:
        C_obs = [p["compute"] for p in optimal_points]
        D_obs = [p["n_tokens"] for p in optimal_points]
        ax.scatter(
            C_obs, D_obs,
            color="#2ca02c",
            s=100,
            alpha=0.8,
            edgecolors="white",
            linewidths=1,
            label="Empirical Optima",
            zorder=3,
        )

    ax.plot(
        C_smooth, D_pred,
        color="#ff7f0e",
        linestyle="--",
        linewidth=2.5,
        label=f"D = {factor_D:.2e} × C^{exponent_D:.3f}",
        zorder=2,
    )

    # Mark specific compute budgets
    D_at_budgets = [factor_D * (C ** exponent_D) for C in compute_budgets]
    ax.scatter(
        compute_budgets, D_at_budgets,
        color="#d62728",
        s=150,
        marker="*",
        edgecolors="black",
        linewidths=1,
        label="Target Budgets",
        zorder=4,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Compute Budget C (FLOPs)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Optimal Tokens D", fontsize=11, fontweight="bold")
    ax.set_title("Optimal Data Size vs Compute", fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)

    # Add overall title
    fig.suptitle(
        f"IsoFLOPs Optimal Allocation (a + b = {exponent_N + exponent_D:.3f})",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()

    os.makedirs(save_path, exist_ok=True)
    filename = f"optimal_allocation.{format}"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches="tight", format=format)
    plt.close(fig)

    return filepath


def plot_all_visualizations(
    experiments: list[dict],
    chinchilla_params: dict,
    isoflops_result: dict,
    compute_budgets: Sequence[float],
    save_path: str,
    dpi: int = 300,
    format: str = "png",
) -> dict[str, str]:
    """
    Generate all scaling laws visualizations in one call.

    Args:
        experiments: List of experiment dicts
        chinchilla_params: Fitted Chinchilla parameters
        isoflops_result: Fitted IsoFLOPs parameters
        compute_budgets: Target compute budgets for prediction
        save_path: Directory to save plots
        dpi: Resolution for raster formats
        format: Output format ('png', 'pdf', 'svg')

    Returns:
        Dict mapping plot name to saved file path
    """
    paths = {}

    paths["isoflop_curves"] = plot_isoflop_curves(
        experiments, save_path, dpi=dpi, format=format
    )
    paths["chinchilla_contour"] = plot_chinchilla_contour(
        chinchilla_params, save_path, dpi=dpi, format=format
    )
    paths["loss_vs_tokens"] = plot_loss_vs_tokens(
        experiments, save_path, dpi=dpi, format=format
    )
    paths["optimal_allocation"] = plot_optimal_allocation(
        isoflops_result, compute_budgets, save_path, dpi=dpi, format=format
    )

    return paths
