"""Adapter modules for parameter-efficient domain adaptation."""

from .conv_lora import ConvLoRA
from .freq_gate import FreqGate
from .freq_gated_lora import FreqGatedConvLoRA
from .inject import inject_lora, remove_lora

__all__ = [
    "ConvLoRA", "FreqGate", "FreqGatedConvLoRA",
    "inject_lora", "remove_lora",
]
