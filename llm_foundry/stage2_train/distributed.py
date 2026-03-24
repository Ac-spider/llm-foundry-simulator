"""
分布式数据并行（DDP）工具模块。

提供进程组管理、集合通信操作、分布式采样器等工具函数，
用于多 GPU / 多机训练场景。
"""

import os
from typing import Any, List, Optional, Callable

import torch
import torch.distributed as dist
from torch.utils.data import Sampler, Dataset


def setup_distributed(
    rank: int,
    world_size: int,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: str = "12390",
) -> str:
    """初始化分布式进程组，并为当前进程分配 CUDA 设备。

    本函数应在每个参与训练的进程中调用一次。
    它设置 MASTER 地址/端口环境变量，根据 rank 选择本地 GPU，
    并通过 dist.init_process_group 完成进程间通信握手。

    参数:
        rank (int): 当前进程的全局排名，范围 [0, world_size)。
            rank=0 通常作为主进程负责日志、保存等操作。
        world_size (int): 参与训练的总进程数（= 总 GPU 数，单机多卡时）。
        backend (str): PyTorch 分布式通信后端，多 GPU 训练推荐 "nccl"，
            CPU 训练可使用 "gloo"。默认为 "nccl"。
        master_addr (str): 主节点地址，默认为 "localhost"。
        master_port (str): 主节点端口，默认为 "12390"。

    返回:
        str: 当前进程绑定的设备字符串，如 "cuda:0"、"cuda:1" 或 "cpu"。

    异常:
        ValueError: 当 CUDA 可用但检测不到任何 GPU 设备时抛出。

    注意:
        local_rank = rank % device_count 支持多机多卡场景：
        每台机器上的进程按顺序绑定该机器上的 GPU。
    """
    # 设置主节点地址与端口（所有进程必须一致）
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = master_port

    # https://discuss.pytorch.org/t/should-local-rank-be-equal-to-torch-cuda-current-device/150873/2
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        local_rank = None
        if device_count > 0:
            # 取模运算：在多机场景下将全局 rank 映射到本机 GPU 索引
            local_rank = rank % device_count
            torch.cuda.set_device(local_rank)  # 将当前 CUDA 上下文绑定到 local_rank 对应的 GPU
        else:
            raise ValueError("Unable to find CUDA devices.")
        device = f"cuda:{local_rank}"
    else:
        # 无 CUDA 环境时退回 CPU（通常用于调试）
        device = "cpu"

    # 初始化进程组：所有进程在此处阻塞，直到全部进程完成握手
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    return device


def cleanup_distributed() -> None:
    """安全地销毁分布式进程组，释放通信资源。

    在所有训练逻辑结束后调用，确保：
    1. 所有进程都已完成工作（通过 barrier 同步）；
    2. 进程组被正确销毁，避免资源泄漏或进程僵死。

    注意:
        dist.barrier() 确保没有进程提前销毁进程组，
        防止其他进程在通信时因某进程过早退出而报错。
    """
    # 全局屏障同步：等待所有进程都到达此处后才继续
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def get_rank() -> int:
    """获取当前进程的 rank。

    返回:
        int: 当前进程的全局 rank。如果分布式未初始化，返回 0。
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_world_size() -> int:
    """获取总进程数（world size）。

    返回:
        int: 参与训练的总进程数。如果分布式未初始化，返回 1。
    """
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def is_main_process() -> bool:
    """判断当前进程是否为主进程（rank 0）。

    返回:
        bool: 如果当前进程是 rank 0 或分布式未初始化，返回 True。
    """
    return get_rank() == 0


def all_reduce(tensor: torch.Tensor, op: str = "sum", async_op: bool = False) -> Optional[Any]:
    """对张量执行 AllReduce 操作。

    参数:
        tensor (torch.Tensor): 输入/输出张量，操作结果会原地写回。
        op (str): 归约操作类型，可选 "sum"、"mean"、"max"、"min"、"product"。
            默认为 "sum"。注意："mean" 通过先 sum 再除以 world_size 实现。
        async_op (bool): 是否异步执行。如果为 True，返回 handle；否则返回 None。

    返回:
        Optional[Any]: 如果 async_op=True，返回工作句柄；否则返回 None。

    异常:
        ValueError: 当 op 不是有效的归约操作时抛出。
    """
    # 映射操作字符串到 dist.ReduceOp
    op_map = {
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
        "product": dist.ReduceOp.PRODUCT,
    }

    if op not in op_map and op != "mean":
        raise ValueError(f"Unknown reduce op: {op}. Supported: sum, mean, max, min, product")

    if not dist.is_available() or not dist.is_initialized():
        # 单进程模式，无需操作
        if op == "mean":
            tensor.div_(get_world_size())
        return None

    # mean 操作通过 sum 后再除以 world_size 实现
    if op == "mean":
        handle = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=async_op)
        if async_op:
            # 如果是异步操作，需要返回一个包装器来处理除法
            class MeanHandle:
                def __init__(self, handle, world_size):
                    self.handle = handle
                    self.world_size = world_size

                def wait(self):
                    self.handle.wait()
                    tensor.div_(self.world_size)

            return MeanHandle(handle, get_world_size())
        else:
            tensor.div_(get_world_size())
            return None
    else:
        return dist.all_reduce(tensor, op=op_map[op], async_op=async_op)


def all_gather(tensor_list: List[torch.Tensor], tensor: torch.Tensor, async_op: bool = False) -> Optional[Any]:
    """从所有进程收集张量。

    参数:
        tensor_list (List[torch.Tensor]): 输出列表，用于存储从所有进程收集的张量。
            列表长度必须等于 world_size，每个张量的形状必须与输入 tensor 相同。
        tensor (torch.Tensor): 当前进程要发送的张量。
        async_op (bool): 是否异步执行。如果为 True，返回 handle；否则返回 None。

    返回:
        Optional[Any]: 如果 async_op=True，返回工作句柄；否则返回 None。

    异常:
        ValueError: 当 tensor_list 长度不等于 world_size 时抛出。
    """
    world_size = get_world_size()

    if len(tensor_list) != world_size:
        raise ValueError(f"tensor_list length ({len(tensor_list)}) must equal world_size ({world_size})")

    if not dist.is_available() or not dist.is_initialized():
        # 单进程模式，直接复制
        tensor_list[0] = tensor.clone()
        return None

    return dist.all_gather(tensor_list, tensor, async_op=async_op)


def broadcast(tensor: torch.Tensor, src: int = 0, async_op: bool = False) -> Optional[Any]:
    """从指定源进程广播张量到所有进程。

    参数:
        tensor (torch.Tensor): 要广播的张量。在 src 进程上为输入，在其他进程上为输出。
        src (int): 源进程 rank，默认为 0。
        async_op (bool): 是否异步执行。如果为 True，返回 handle；否则返回 None。

    返回:
        Optional[Any]: 如果 async_op=True，返回工作句柄；否则返回 None。
    """
    if not dist.is_available() or not dist.is_initialized():
        # 单进程模式，无需操作
        return None

    return dist.broadcast(tensor, src=src, async_op=async_op)


def barrier() -> None:
    """同步屏障：等待所有进程到达此处。

    在所有进程都调用 barrier() 之前，没有进程可以继续执行。
    """
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


class DistributedSampler(Sampler):
    """分布式数据采样器。

    将数据集划分为多个子集，每个进程只处理自己的子集。
    支持 shuffle 和确定性采样，确保每个 epoch 的数据顺序可复现。

    参数:
        dataset (Dataset): 要采样的数据集。
        num_replicas (int, optional): 进程总数。默认为 world_size。
        rank (int, optional): 当前进程 rank。默认为当前 rank。
        shuffle (bool): 是否在每个 epoch 打乱数据顺序。默认为 True。
        seed (int): 用于打乱数据的随机种子。默认为 0。
        drop_last (bool): 如果为 True，丢弃最后一个不完整的批次。
            如果为 False，最后一个批次可能较小。默认为 False。

    示例:
        >>> sampler = DistributedSampler(dataset, shuffle=True)
        >>> loader = DataLoader(dataset, batch_size=32, sampler=sampler)
        >>> for epoch in range(num_epochs):
        ...     sampler.set_epoch(epoch)
        ...     for batch in loader:
        ...         # 训练代码
        ...         pass
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.num_replicas = num_replicas if num_replicas is not None else get_world_size()
        self.rank = rank if rank is not None else get_rank()
        self.epoch = 0
        self.drop_last = drop_last

        # 计算每个进程的样本数
        if self.drop_last:
            self.num_samples = len(dataset) // self.num_replicas
        else:
            self.num_samples = (len(dataset) + self.num_replicas - 1) // self.num_replicas

        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        """返回当前进程的样本索引迭代器。"""
        if self.shuffle:
            # 基于 epoch 和 seed 确定性地打乱
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        if not self.drop_last:
            # 添加额外样本使总数能被 num_replicas 整除
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * ((padding_size // len(indices)) + 1))[:padding_size]
        else:
            # 移除尾部无法整除的样本
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size

        # 将索引分配给各个进程
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        """返回当前进程的样本数。"""
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，用于确定性打乱。

        必须在每个 epoch 开始时调用，以确保不同 epoch 的数据顺序不同。

        参数:
            epoch (int): 当前 epoch 数。
        """
        self.epoch = epoch


class DDPIndividualParameters(torch.nn.Module):
    """逐参数异步 AllReduce 的 DDP 封装。

    工作原理：
        1. 初始化时从 rank 0 广播所有参数，确保各进程初始状态一致。
        2. 为每个 requires_grad=True 的参数注册梯度钩子：
           当该参数的梯度计算完成后（post_accumulate_grad），立即：
             a. 将梯度除以 world_size（实现求均值的第一步）
             b. 发起异步 AllReduce SUM（返回 handle，不阻塞）
        3. 在优化器 step() 前调用 finish_gradient_synchronization() 等待所有 handle 完成。

    通信-计算 overlap：
        梯度钩子在反向传播过程中逐参数触发，后层（靠近输出）的参数梯度先完成，
        其 AllReduce 通信与前层（靠近输入）的反向传播计算并行执行，
        有效隐藏通信延迟。

    缺点：
        每个参数触发一次通信 kernel launch，小参数数量多时 launch 开销显著，
        建议改用 DDPBucketed。

    参数:
        module (torch.nn.Module): 需要用 DDP 包装的模型。
    """

    def __init__(self, module: torch.nn.Module):
        super().__init__()

        self.module = module
        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.communication_handle = []  # 存储未完成的异步通信句柄
        self.world_size = dist.get_world_size() if self.is_initialized else 1

        if self.is_initialized:
            for params in self.module.parameters():
                # 从 rank 0 广播参数：确保所有 rank 从相同初始权重开始训练
                dist.broadcast(params.data, src=0)

                if params.requires_grad:
                    # 注册梯度钩子：该参数梯度累积完毕后立即触发异步 AllReduce
                    params.register_post_accumulate_grad_hook(self._make_hook(params))

    def _make_hook(self, params):
        """钩子工厂函数：为每个参数创建独立的梯度同步钩子。"""
        def hook(params):
            if not self.is_initialized:
                return

            # 先除以 world_size，再 SUM AllReduce，等价于对梯度求均值
            params.grad.data.div_(self.world_size)

            # 异步发起 AllReduce（SUM），立即返回 handle（不阻塞后续反向计算）
            handle = dist.all_reduce(params.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            self.communication_handle.append(handle)

        return hook

    def forward(self, *args, **kwargs):
        """前向传播：直接委托给被包装的模块。"""
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """等待所有异步 AllReduce 操作完成。

        必须在 optimizer.step() 之前调用，确保各参数梯度已完成跨进程同步，
        否则优化器将使用未同步的（仅本 rank 的）梯度进行更新。
        """
        for handle in self.communication_handle:
            handle.wait()

        self.communication_handle.clear()


class DDPBucketed(torch.nn.Module):
    """桶式分组 AllReduce 的 DDP 封装。

    工作原理：
        1. 初始化时将模型参数按顺序贪心分配到若干 bucket（每桶不超过 bucket_size_mb）。
        2. 为每个参数注册梯度钩子，维护每个 bucket 的"就绪参数计数"。
        3. 当某个 bucket 内所有参数梯度均就绪时：
             a. 将该桶内所有梯度拼平为一个连续 flat tensor
             b. 对 flat tensor 发起一次异步 AllReduce
        4. finish_gradient_synchronization() 等待所有桶的 AllReduce 完成，
           并将 flat tensor 拆回原始形状写回各参数梯度。

    参数:
        model (torch.nn.Module): 需要包装的模型。
        bucket_size_mb (float): 每个桶的最大大小（MB），控制通信粒度。
    """

    def __init__(self, model: torch.nn.Module, bucket_size_mb: float = 25.0):
        super().__init__()

        self.model = model
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024  # MB 转字节
        self.is_initialized = dist.is_available() and dist.is_initialized()
        self.world_size = dist.get_world_size() if self.is_initialized else 1

        self.handles = []           # 异步通信句柄列表
        self.buckets = []           # 每个 bucket 包含的参数列表
        self.ready_buckets = []     # 每个 bucket 中已就绪的参数计数
        self.total_buckets = []     # 每个 bucket 的总参数数

        if self.is_initialized:
            # 广播参数：确保所有 rank 初始状态相同
            for p in self.model.parameters():
                dist.broadcast(p.data, src=0)

            self._build_buckets()

    def _build_buckets(self):
        """贪心分桶策略：将模型参数按顺序分配到若干固定大小的桶。"""
        current_buckets = []
        current_size = 0

        params = [p for p in self.model.parameters() if p.requires_grad]

        for param in params:
            # 计算该参数占用的字节数
            param_size = param.numel() * param.element_size()

            # 当前桶已满且非空时，保存当前桶并开新桶
            if current_size + param_size > self.bucket_size_bytes and len(current_buckets) > 0:
                self.buckets.append(current_buckets)
                current_buckets = []
                current_size = 0

            current_buckets.append(param)
            current_size += param_size

        # 保存最后一个桶
        if len(current_buckets) > 0:
            self.buckets.append(current_buckets)

        # 初始化每个桶的就绪计数和总参数数
        self.ready_buckets = [0] * len(self.buckets)
        self.total_buckets = [len(b) for b in self.buckets]

        # 为每个桶中的每个参数注册梯度钩子
        for bucket_id, bucket in enumerate(self.buckets):
            for param in bucket:
                param.register_post_accumulate_grad_hook(self._make_hook(param, bucket_id))

    def _make_hook(self, param, bucket_id):
        """钩子工厂函数：为指定参数创建桶计数和触发 AllReduce 的钩子。"""
        def hook(param):
            # 增加该桶的就绪计数
            self.ready_buckets[bucket_id] += 1

            # 只有桶内所有参数梯度都就绪时，才发起 AllReduce
            if self.ready_buckets[bucket_id] == self.total_buckets[bucket_id]:
                grad = [p.grad for p in self.buckets[bucket_id]]

                # 将多个梯度 tensor 拼为一个连续 flat tensor
                flat_grad = torch._utils._flatten_dense_tensors(grad)

                flat_grad.div_(self.world_size)  # 梯度除以 world_size 实现求均值

                # 对整个桶的 flat_grad 发起一次异步 AllReduce
                handle = dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append((handle, bucket_id, flat_grad))

        return hook

    def forward(self, *args, **kwargs):
        """前向传播：直接委托给被包装的模型。"""
        return self.model(*args, **kwargs)

    def finish_gradient_synchronization(self):
        """等待所有桶的 AllReduce 完成，并将同步后的梯度写回各参数。"""
        for handle, bucket_id, flat_grad in self.handles:
            handle.wait()

            grad = [p.grad for p in self.buckets[bucket_id]]

            # 将 flat tensor 拆回与原始梯度相同形状的列表
            unflat_grad = torch._utils._unflatten_dense_tensors(flat_grad, grad)

            # 将同步后的梯度原地写回各参数
            for orig_grad, new_grad in zip(grad, unflat_grad):
                orig_grad.copy_(new_grad)

        self.handles.clear()

    def reset_buckets(self):
        """重置所有桶的就绪计数，为下一次反向传播做准备。

        必须在每个训练 step 结束（optimizer.step() 之后、下一次 backward() 之前）调用。
        """
        for i in range(len(self.ready_buckets)):
            self.ready_buckets[i] = 0


def reduce_dict(input_dict: dict, average: bool = True) -> dict:
    """对字典中的所有张量进行 AllReduce。

    用于聚合各进程的指标（如 loss、accuracy 等）。

    参数:
        input_dict (dict): 包含张量的字典。
        average (bool): 是否对结果取平均。默认为 True。

    返回:
        dict: 聚合后的字典（原地修改）。
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        # 获取所有键，确保所有进程顺序一致
        keys = sorted(input_dict.keys())
        values = [input_dict[k] for k in keys]

        # 将所有值拼接成一个张量进行通信
        # 要求所有值都是标量张量
        try:
            stacked = torch.stack(values)
        except RuntimeError:
            # 如果不是标量，逐个处理
            for k in keys:
                if isinstance(input_dict[k], torch.Tensor):
                    all_reduce(input_dict[k], op="sum")
                    if average:
                        input_dict[k].div_(world_size)
            return input_dict

        all_reduce(stacked, op="sum")
        if average:
            stacked.div_(world_size)

        # 将结果写回字典
        for k, v in zip(keys, stacked):
            input_dict[k] = v

    return input_dict
