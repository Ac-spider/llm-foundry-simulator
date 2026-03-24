"""Stage 3: Scaling Laws Module.

This module implements Chinchilla scaling laws analysis for LLM training optimization,
including IsoFLOPs curve fitting and Chinchilla loss surface modeling.

Key Features:
- Chinchilla loss fitting: L(N, D) = E + A/N^α + B/D^β
- IsoFLOPs analysis: Find optimal (N, D) pairs at fixed compute budgets
- Power law fitting: N_opt(C) = A_N * C^a, D_opt(C) = B_D * C^b
- Visualization: Publication-quality plots for scaling analysis

Core Classes:
    ScalingAnalyzer: Main analysis class combining IsoFLOPs and Chinchilla fitting
    ScalingRunner: Experiment runner for scaling experiments
    ScalingExperiment: Configuration and state for a single scaling experiment
    ChinchillaFitter: Object-oriented interface for Chinchilla parameter fitting
    FittingConfig: Configuration for Chinchilla fitting
    IsoFLOPsConfig: Configuration for IsoFLOPs calculations

Key Functions:
    chinchilla_loss: Compute loss using Chinchilla formula
    fit_chinchilla_params: Fit Chinchilla parameters from experimental data
    predict_loss: Predict loss for given (N, D) using fitted parameters
    compute_optimal_allocation: Find optimal N and D for a compute budget
    compute_flops: Calculate training FLOPs (C = 6 * N * D)
    derive_tokens_from_flops: Derive token count from FLOPs and parameters
    generate_isoflop_curves: Generate (N, D) configurations for fixed FLOPs
    fit_power_law: Fit power law y = A * C^b using log-space regression
    get_chinchilla_optimal: Get Chinchilla-optimal N and D for compute budget

Example:
    >>> from llm_foundry.stage3_scaling import ScalingAnalyzer
    >>> analyzer = ScalingAnalyzer(config)
    >>> result = analyzer.run(experiments_data)
    >>> print(f"Chinchilla alpha: {result['chinchilla']['alpha']}")
    >>> print(f"Optimal N at 1e21 FLOPs: {result['isoflops']['optimal_model_size_at_compute'][3]}")

Reference:
    Hoffmann et al., "Training Compute-Optimal Large Language Models", 2022 (Chinchilla)
"""

from llm_foundry.stage3_scaling.fitting import (
    ChinchillaFitter,
    FittingConfig,
    chinchilla_loss,
    compute_flops,
    compute_optimal_allocation,
    derive_tokens_from_flops,
    fit_chinchilla_params,
    load_experiments_from_csv,
    predict_loss,
)
from llm_foundry.stage3_scaling.runner import (
    ScalingExperiment,
    ScalingRunner,
    compute_flops,
    compute_non_embedding_params,
    derive_tokens_from_flops,
    estimate_optimal_nd,
    generate_experiment_matrix,
    generate_model_config_from_params,
    run_scaling_experiment,
)
from llm_foundry.stage3_scaling.scaling import ScalingAnalyzer

__all__ = [
    # ScalingAnalyzer
    "ScalingAnalyzer",
    # Runner module
    "ScalingExperiment",
    "ScalingRunner",
    "compute_flops",
    "compute_non_embedding_params",
    "derive_tokens_from_flops",
    "estimate_optimal_nd",
    "generate_experiment_matrix",
    "generate_model_config_from_params",
    "run_scaling_experiment",
    # Fitting module
    "ChinchillaFitter",
    "FittingConfig",
    "chinchilla_loss",
    "compute_flops",
    "compute_optimal_allocation",
    "derive_tokens_from_flops",
    "fit_chinchilla_params",
    "load_experiments_from_csv",
    "predict_loss",
]
