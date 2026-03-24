"""Stage 2: Training module.

导出:
    Trainer: 训练器类，支持单卡和 DDP+ZeRO-1 多卡训练
"""

from llm_foundry.stage2_train.trainer import Trainer

__all__ = ["Trainer"]
