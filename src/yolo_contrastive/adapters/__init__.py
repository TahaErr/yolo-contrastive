"""Adapter modules for parameter-efficient domain adaptation."""

from .conv_lora import ConvLoRA
from .freq_gate import FreqGate
from .freq_gated_lora import FreqGatedConvLoRA
from .task_routed_lora import TaskRoutedConvLoRA, TaskRouter, LoRABranch
from .inject import inject_lora, remove_lora
from .merge import compute_merge_alphas, merge_task_routed_model

__all__ = [
    "ConvLoRA", "FreqGate", "FreqGatedConvLoRA",
    "TaskRoutedConvLoRA", "TaskRouter", "LoRABranch",
    "inject_lora", "remove_lora",
    "compute_merge_alphas", "merge_task_routed_model",
]
