"""Tests for environment check module."""

import sys

from llm_foundry.common.env_check import (
    EnvStatus,
    check_cuda,
    check_environment,
    get_gpu_memory_info,
)


def test_env_status_creation():
    """Test EnvStatus dataclass creation."""
    status = EnvStatus(
        cuda_available=True,
        cuda_version="12.1",
        gpu_count=1,
        gpu_names=["Test GPU"],
        gpu_memory_gb=[8.0],
    )

    assert status.cuda_available is True
    assert status.cuda_version == "12.1"
    assert status.gpu_count == 1


def test_env_status_to_dict():
    """Test EnvStatus conversion to dict."""
    status = EnvStatus(
        cuda_available=True,
        gpu_count=1,
        gpu_names=["Test GPU"],
    )

    d = status.to_dict()
    assert d["cuda_available"] is True
    assert d["gpu_count"] == 1


def test_check_environment():
    """Test environment check."""
    status = check_environment()

    assert isinstance(status, EnvStatus)
    assert status.python_version.startswith(str(sys.version_info.major))
    assert status.torch_version != ""


def test_check_cuda():
    """Test CUDA check."""
    result = check_cuda()
    assert isinstance(result, bool)


def test_get_gpu_memory_info():
    """Test GPU memory info retrieval."""
    info = get_gpu_memory_info()

    # Should return dict (empty if no CUDA)
    assert isinstance(info, dict)

    if check_cuda():
        # If CUDA available, should have entries
        assert len(info) > 0
        for gpu_id, mem_info in info.items():
            assert "total_gb" in mem_info
            assert "allocated_gb" in mem_info
