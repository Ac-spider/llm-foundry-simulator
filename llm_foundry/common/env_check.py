"""
Hardware environment detection and reporting.

Provides utilities to check GPU availability, CUDA version, and other
hardware capabilities needed for LLM training.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from typing import Any

import torch

logger = logging.getLogger(__name__)


@dataclass
class EnvStatus:
    """Environment status report.

    Attributes:
        cuda_available: Whether CUDA is available
        cuda_version: CUDA version string (e.g., "12.1") or None
        gpu_count: Number of GPUs detected
        gpu_names: List of GPU names
        gpu_memory_gb: List of GPU memory in GB per device
        torch_version: PyTorch version string
        python_version: Python version string
        triton_available: Whether Triton is installed
        flash_attn_available: Whether flash-attn is installed
        distributed_available: Whether torch.distributed is available
        recommended_backend: Recommended attention backend based on hardware
        warnings: List of warning messages about the environment
    """

    cuda_available: bool = False
    cuda_version: str | None = None
    gpu_count: int = 0
    gpu_names: list[str] = field(default_factory=list)
    gpu_memory_gb: list[float] = field(default_factory=list)
    torch_version: str = ""
    python_version: str = ""
    triton_available: bool = False
    flash_attn_available: bool = False
    distributed_available: bool = False
    recommended_backend: str = "sdpa"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "cuda_available": self.cuda_available,
            "cuda_version": self.cuda_version,
            "gpu_count": self.gpu_count,
            "gpu_names": self.gpu_names,
            "gpu_memory_gb": self.gpu_memory_gb,
            "torch_version": self.torch_version,
            "python_version": self.python_version,
            "triton_available": self.triton_available,
            "flash_attn_available": self.flash_attn_available,
            "distributed_available": self.distributed_available,
            "recommended_backend": self.recommended_backend,
            "warnings": self.warnings,
        }

    def print_report(self) -> None:
        """Print a formatted environment report."""
        print("=" * 60)
        print("LLM Foundry Simulator - Environment Check")
        print("=" * 60)
        print(f"Python: {self.python_version}")
        print(f"PyTorch: {self.torch_version}")
        print(f"CUDA Available: {self.cuda_available}")

        if self.cuda_available:
            print(f"CUDA Version: {self.cuda_version}")
            print(f"GPU Count: {self.gpu_count}")
            for i, (name, memory) in enumerate(zip(self.gpu_names, self.gpu_memory_gb)):
                print(f"  GPU {i}: {name} ({memory:.1f} GB)")

        print("-" * 60)
        print("Optional Dependencies:")
        print(f"  Triton: {'✓' if self.triton_available else '✗'}")
        print(f"  Flash Attention: {'✓' if self.flash_attn_available else '✗'}")
        print(f"  Distributed: {'✓' if self.distributed_available else '✗'}")
        print("-" * 60)
        print(f"Recommended Attention Backend: {self.recommended_backend}")

        if self.warnings:
            print("-" * 60)
            print("Warnings:")
            for warning in self.warnings:
                print(f"  ⚠ {warning}")

        print("=" * 60)


def check_environment() -> EnvStatus:
    """
    Check the current hardware and software environment.

    Returns:
        EnvStatus dataclass with detailed environment information
    """
    status = EnvStatus()

    # Python version
    status.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # PyTorch version
    status.torch_version = torch.__version__

    # CUDA availability
    status.cuda_available = torch.cuda.is_available()

    if status.cuda_available:
        # CUDA version
        status.cuda_version = torch.version.cuda

        # GPU information
        status.gpu_count = torch.cuda.device_count()
        for i in range(status.gpu_count):
            props = torch.cuda.get_device_properties(i)
            status.gpu_names.append(props.name)
            status.gpu_memory_gb.append(props.total_memory / (1024**3))

    # Check optional dependencies
    try:
        import triton

        status.triton_available = True
    except ImportError:
        pass

    try:
        import flash_attn

        status.flash_attn_available = True
    except ImportError:
        pass

    # Check distributed availability
    status.distributed_available = torch.distributed.is_available()

    # Determine recommended backend
    if status.triton_available and status.cuda_available:
        status.recommended_backend = "triton"
    elif status.cuda_available:
        status.recommended_backend = "torch.compile"
    else:
        status.recommended_backend = "sdpa"

    # Generate warnings
    if not status.cuda_available:
        status.warnings.append("CUDA not available. Training will be very slow on CPU.")

    if status.cuda_available and not status.triton_available:
        status.warnings.append(
            "Triton not installed. Attention will use torch.compile fallback. "
            "Install triton for better performance: pip install triton"
        )

    if status.gpu_count > 1 and not status.distributed_available:
        status.warnings.append(
            "Multiple GPUs detected but torch.distributed not available. "
            "Multi-GPU training will not work."
        )

    return status


def check_cuda() -> bool:
    """Quick check if CUDA is available.

    Returns:
        True if CUDA is available
    """
    return torch.cuda.is_available()


def get_gpu_memory_info() -> dict[int, dict[str, float]]:
    """Get memory information for all GPUs.

    Returns:
        Dictionary mapping GPU index to memory info dict with keys:
        - total_gb: Total memory in GB
        - allocated_gb: Allocated memory in GB
        - reserved_gb: Reserved memory in GB
        - free_gb: Free memory in GB
    """
    if not torch.cuda.is_available():
        return {}

    info = {}
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        total = props.total_memory / (1024**3)
        allocated = torch.cuda.memory_allocated(i) / (1024**3)
        reserved = torch.cuda.memory_reserved(i) / (1024**3)

        info[i] = {
            "total_gb": total,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "free_gb": total - allocated,
        }

    return info


def print_gpu_memory_summary() -> None:
    """Print a summary of GPU memory usage."""
    if not torch.cuda.is_available():
        print("CUDA not available")
        return

    print("GPU Memory Summary:")
    for i, mem in get_gpu_memory_info().items():
        print(f"  GPU {i}: {mem['allocated_gb']:.2f} / {mem['total_gb']:.2f} GB allocated")
