"""
分布式训练模块测试。

测试策略：
    - 单进程测试：验证 API 正确性和单进程回退行为
    - 多进程测试：使用 torch.multiprocessing 启动子进程验证实际 DDP 功能

注意：
    多进程测试需要 GPU 环境，如果没有 GPU 会被跳过。
"""

import os
import sys
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from llm_foundry.stage2_train.distributed import (
    setup_distributed,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    all_reduce,
    all_gather,
    broadcast,
    barrier,
    DistributedSampler,
    DDPIndividualParameters,
    DDPBucketed,
    reduce_dict,
)


class TestSingleProcessMode:
    """单进程模式测试（无 DDP 初始化时的回退行为）。"""

    def test_get_rank_single_process(self):
        """测试单进程模式下 get_rank 返回 0。"""
        assert get_rank() == 0

    def test_get_world_size_single_process(self):
        """测试单进程模式下 get_world_size 返回 1。"""
        assert get_world_size() == 1

    def test_is_main_process_single_process(self):
        """测试单进程模式下 is_main_process 返回 True。"""
        assert is_main_process() is True

    def test_all_reduce_single_process(self):
        """测试单进程模式下 all_reduce 正常工作（原地修改）。"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = all_reduce(tensor, op="sum")
        assert result is None  # 同步操作返回 None
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_all_reduce_mean_single_process(self):
        """测试单进程模式下 all_reduce mean 正常工作。"""
        tensor = torch.tensor([2.0, 4.0, 6.0])
        all_reduce(tensor, op="mean")
        # 单进程下 mean 就是除以 world_size=1
        assert torch.allclose(tensor, torch.tensor([2.0, 4.0, 6.0]))

    def test_all_gather_single_process(self):
        """测试单进程模式下 all_gather 正常工作。"""
        tensor = torch.tensor([1.0, 2.0])
        tensor_list = [torch.zeros(2) for _ in range(1)]
        result = all_gather(tensor_list, tensor)
        assert result is None
        assert torch.allclose(tensor_list[0], tensor)

    def test_broadcast_single_process(self):
        """测试单进程模式下 broadcast 正常工作。"""
        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = broadcast(tensor, src=0)
        assert result is None
        assert torch.allclose(tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_barrier_single_process(self):
        """测试单进程模式下 barrier 正常工作（无操作）。"""
        barrier()  # 不应该抛出异常


class TestDistributedSampler:
    """分布式采样器测试。"""

    def test_sampler_length_single_process(self):
        """测试单进程模式下采样器长度。"""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0)
        assert len(sampler) == 100

    def test_sampler_length_with_drop_last(self):
        """测试 drop_last=True 时的采样器长度。"""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = DistributedSampler(dataset, num_replicas=3, rank=0, drop_last=True)
        # 100 // 3 = 33
        assert len(sampler) == 33

    def test_sampler_length_without_drop_last(self):
        """测试 drop_last=False 时的采样器长度。"""
        dataset = TensorDataset(torch.randn(100, 10))
        sampler = DistributedSampler(dataset, num_replicas=3, rank=0, drop_last=False)
        # ceil(100 / 3) = 34
        assert len(sampler) == 34

    def test_sampler_indices_no_shuffle(self):
        """测试无打乱时的采样器索引。"""
        dataset = TensorDataset(torch.randn(10, 5))
        sampler = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)
        indices = list(sampler)
        # rank 0 应该获得索引 0, 2, 4, 6, 8
        assert indices == [0, 2, 4, 6, 8]

    def test_sampler_indices_with_padding(self):
        """测试需要填充时的采样器行为。"""
        dataset = TensorDataset(torch.randn(10, 5))
        sampler = DistributedSampler(dataset, num_replicas=3, rank=0, shuffle=False, drop_last=False)
        indices = list(sampler)
        # ceil(10/3) = 4, total_size = 12, 需要填充 2 个
        # indices = [0, 3, 6, 9] (rank 0 的采样)
        assert len(indices) == 4

    def test_sampler_set_epoch(self):
        """测试设置 epoch 改变随机种子。"""
        dataset = TensorDataset(torch.randn(100, 5))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42)

        sampler.set_epoch(0)
        indices_epoch_0 = list(sampler)

        sampler.set_epoch(1)
        indices_epoch_1 = list(sampler)

        # 不同 epoch 应该产生不同的顺序
        assert indices_epoch_0 != indices_epoch_1

    def test_sampler_determinism(self):
        """测试相同时 epoch 产生相同的顺序（确定性）。"""
        dataset = TensorDataset(torch.randn(100, 5))
        sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42)

        sampler.set_epoch(0)
        indices_1 = list(sampler)

        sampler.set_epoch(0)
        indices_2 = list(sampler)

        assert indices_1 == indices_2


class TestReduceDict:
    """字典归约测试。"""

    def test_reduce_dict_single_process(self):
        """测试单进程模式下 reduce_dict 正常工作。"""
        input_dict = {
            "loss": torch.tensor(2.5),
            "accuracy": torch.tensor(0.8),
        }
        result = reduce_dict(input_dict, average=True)
        assert torch.allclose(result["loss"], torch.tensor(2.5))
        assert torch.allclose(result["accuracy"], torch.tensor(0.8))


# =============================================================================
# 多进程测试（需要 GPU）
# =============================================================================

def _test_setup_distributed_worker(rank, world_size, return_dict):
    """测试 setup_distributed 的工作进程。"""
    try:
        device = setup_distributed(rank, world_size, backend="gloo")
        return_dict[rank] = {
            "rank": get_rank(),
            "world_size": get_world_size(),
            "is_main": is_main_process(),
            "device": device,
        }
        cleanup_distributed()
    except Exception as e:
        return_dict[rank] = {"error": str(e)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_setup_distributed_multiprocess():
    """测试多进程 DDP 初始化。"""
    world_size = 2
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        _test_setup_distributed_worker,
        args=(world_size, return_dict),
        nprocs=world_size,
        join=True,
    )

    # 验证每个进程的结果
    for rank in range(world_size):
        result = return_dict[rank]
        assert "error" not in result, f"Rank {rank} failed: {result.get('error')}"
        assert result["rank"] == rank
        assert result["world_size"] == world_size
        assert result["is_main"] == (rank == 0)


def _test_all_reduce_worker(rank, world_size, return_dict):
    """测试 all_reduce 的工作进程。"""
    try:
        setup_distributed(rank, world_size, backend="gloo")

        # 每个进程有不同的值
        tensor = torch.tensor([float(rank + 1)])
        all_reduce(tensor, op="sum")

        # sum: 1 + 2 = 3
        expected = sum(range(1, world_size + 1))
        return_dict[rank] = {"result": tensor.item(), "expected": expected}

        cleanup_distributed()
    except Exception as e:
        return_dict[rank] = {"error": str(e)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_all_reduce_multiprocess():
    """测试多进程 all_reduce。"""
    world_size = 2
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        _test_all_reduce_worker,
        args=(world_size, return_dict),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        result = return_dict[rank]
        assert "error" not in result
        assert abs(result["result"] - result["expected"]) < 1e-6


def _test_all_gather_worker(rank, world_size, return_dict):
    """测试 all_gather 的工作进程。"""
    try:
        setup_distributed(rank, world_size, backend="gloo")

        tensor = torch.tensor([float(rank)])
        tensor_list = [torch.zeros(1) for _ in range(world_size)]
        all_gather(tensor_list, tensor)

        return_dict[rank] = {"result": [t.item() for t in tensor_list]}

        cleanup_distributed()
    except Exception as e:
        return_dict[rank] = {"error": str(e)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_all_gather_multiprocess():
    """测试多进程 all_gather。"""
    world_size = 2
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        _test_all_gather_worker,
        args=(world_size, return_dict),
        nprocs=world_size,
        join=True,
    )

    expected = [0.0, 1.0]
    for rank in range(world_size):
        result = return_dict[rank]
        assert "error" not in result
        assert result["result"] == expected


def _test_broadcast_worker(rank, world_size, return_dict):
    """测试 broadcast 的工作进程。"""
    try:
        setup_distributed(rank, world_size, backend="gloo")

        if rank == 0:
            tensor = torch.tensor([100.0, 200.0])
        else:
            tensor = torch.tensor([0.0, 0.0])

        broadcast(tensor, src=0)

        return_dict[rank] = {"result": tensor.tolist()}

        cleanup_distributed()
    except Exception as e:
        return_dict[rank] = {"error": str(e)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_broadcast_multiprocess():
    """测试多进程 broadcast。"""
    world_size = 2
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        _test_broadcast_worker,
        args=(world_size, return_dict),
        nprocs=world_size,
        join=True,
    )

    expected = [100.0, 200.0]
    for rank in range(world_size):
        result = return_dict[rank]
        assert "error" not in result
        assert result["result"] == expected


def _test_barrier_worker(rank, world_size, return_dict):
    """测试 barrier 的工作进程。"""
    try:
        setup_distributed(rank, world_size, backend="gloo")

        import time
        if rank == 0:
            time.sleep(0.1)

        barrier()

        return_dict[rank] = {"passed": True}

        cleanup_distributed()
    except Exception as e:
        return_dict[rank] = {"error": str(e)}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_barrier_multiprocess():
    """测试多进程 barrier。"""
    world_size = 2
    manager = mp.Manager()
    return_dict = manager.dict()

    mp.spawn(
        _test_barrier_worker,
        args=(world_size, return_dict),
        nprocs=world_size,
        join=True,
    )

    for rank in range(world_size):
        result = return_dict[rank]
        assert "error" not in result
        assert result["passed"] is True


class TestDDPWrappers:
    """DDP 包装器测试。"""

    def test_ddp_individual_parameters_init(self):
        """测试 DDPIndividualParameters 初始化。"""
        model = torch.nn.Linear(10, 5)
        ddp_model = DDPIndividualParameters(model)

        assert ddp_model.module is model
        assert hasattr(ddp_model, "communication_handle")
        assert hasattr(ddp_model, "world_size")

    def test_ddp_individual_parameters_forward(self):
        """测试 DDPIndividualParameters 前向传播。"""
        model = torch.nn.Linear(10, 5)
        ddp_model = DDPIndividualParameters(model)

        x = torch.randn(2, 10)
        output = ddp_model(x)

        assert output.shape == (2, 5)

    def test_ddp_bucketed_init(self):
        """测试 DDPBucketed 初始化。"""
        model = torch.nn.Sequential(
            torch.nn.Linear(10, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 5),
        )
        ddp_model = DDPBucketed(model, bucket_size_mb=1.0)

        assert ddp_model.model is model
        assert hasattr(ddp_model, "buckets")
        assert hasattr(ddp_model, "handles")

    def test_ddp_bucketed_forward(self):
        """测试 DDPBucketed 前向传播。"""
        model = torch.nn.Linear(10, 5)
        ddp_model = DDPBucketed(model, bucket_size_mb=1.0)

        x = torch.randn(2, 10)
        output = ddp_model(x)

        assert output.shape == (2, 5)

    def test_ddp_bucketed_reset_buckets(self):
        """测试 DDPBucketed reset_buckets。"""
        model = torch.nn.Linear(10, 5)
        ddp_model = DDPBucketed(model, bucket_size_mb=1.0)

        # 手动修改就绪计数
        ddp_model.ready_buckets = [1, 2, 3]
        ddp_model.reset_buckets()

        assert all(r == 0 for r in ddp_model.ready_buckets)


class TestErrorHandling:
    """错误处理测试。"""

    def test_all_reduce_invalid_op(self):
        """测试 all_reduce 无效操作。"""
        tensor = torch.tensor([1.0])
        with pytest.raises(ValueError, match="Unknown reduce op"):
            all_reduce(tensor, op="invalid_op")

    def test_all_gather_wrong_list_length(self):
        """测试 all_gather 列表长度错误。"""
        tensor = torch.tensor([1.0])
        # world_size=1 时列表长度应为 1
        tensor_list = [torch.zeros(1), torch.zeros(1)]
        with pytest.raises(ValueError, match="tensor_list length"):
            all_gather(tensor_list, tensor)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
