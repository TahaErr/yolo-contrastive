"""inject/remove — YOLO backbone'a LoRA adapter enjeksiyonu."""

from __future__ import annotations
from typing import Dict, List, Optional
import torch.nn as nn

from .conv_lora import ConvLoRA
from .freq_gated_lora import FreqGatedConvLoRA
from .task_routed_lora import TaskRoutedConvLoRA


def inject_lora(
    model: nn.Module,
    rank: int = 4,
    scale: float = 1.0,
    dropout: float = 0.0,
    adapter_type: str = "freq_gated",
    target_layers: Optional[List[int]] = None,
    backbone_layers: int = 10,
    min_channels: int = 32,
    gate_hidden: int = 16,
    gate_init_bias: float = 1.0,
    num_tasks: int = 3,
    use_gate: bool = True,
    verbose: bool = True,
) -> Dict[str, int]:
    """YOLO backbone Conv2d katmanlarına LoRA adapter enjekte et."""

    injected = 0
    total_frozen = 0
    total_lora = 0

    backbone = _get_backbone_convs(model, backbone_layers, target_layers)

    for layer_idx, full_name, parent, attr_name, conv in backbone:
        if conv.in_channels < min_channels and conv.out_channels < min_channels:
            continue

        device = conv.weight.device

        if adapter_type == "freq_gated":
            adapter = FreqGatedConvLoRA(
                conv, rank=rank, scale=scale, dropout=dropout,
                gate_hidden=gate_hidden, gate_init_bias=gate_init_bias,
            ).to(device)
            lora_params = adapter.num_trainable_params
            frozen_params = adapter.num_frozen_params
        elif adapter_type == "task_routed":
            adapter = TaskRoutedConvLoRA(
                conv, num_tasks=num_tasks, rank=rank, scale=scale,
                dropout=dropout, use_gate=use_gate, gate_hidden=gate_hidden,
            ).to(device)
            lora_params = adapter.num_trainable_params
            frozen_params = adapter.num_frozen_params
        elif adapter_type == "plain":
            adapter = ConvLoRA(conv, rank=rank, scale=scale, dropout=dropout).to(device)
            lora_params = adapter.num_lora_params
            frozen_params = adapter.num_frozen_params
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}")

        setattr(parent, attr_name, adapter)
        total_frozen += frozen_params
        total_lora += lora_params
        injected += 1

        if verbose:
            print(f"  [lora] {full_name}: {conv.in_channels}->{conv.out_channels} "
                  f"k={conv.kernel_size} rank={rank} (+{lora_params})")

    _freeze_backbone_non_lora(model, backbone_layers)

    info = {
        "injected": injected,
        "frozen_params": total_frozen,
        "lora_params": total_lora,
        "total_trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }

    if verbose:
        ratio = total_lora / max(1, total_frozen) * 100
        print(f"\n  [lora] Injected: {injected} adapters")
        print(f"  [lora] LoRA params: {total_lora:,} ({ratio:.2f}% of backbone)")
        print(f"  [lora] Total trainable: {info['total_trainable']:,}")

    return info


def remove_lora(model: nn.Module, merge: bool = True, verbose: bool = True) -> int:
    """LoRA adapter'ları kaldır, opsiyonel merge."""
    removed = 0
    replacements = []

    # Tüm adapter'ları bul (sadece en dıştaki — iç içe ConvLoRA'yı atla)
    adapter_names = set()
    for name, module in list(model.named_modules()):
        if isinstance(module, (FreqGatedConvLoRA, TaskRoutedConvLoRA)):
            adapter_names.add(name)
        elif isinstance(module, ConvLoRA):
            # FreqGatedConvLoRA/TaskRoutedConvLoRA'nın içindeki ConvLoRA'yı atla
            if any(name.startswith(p + ".") for p in adapter_names):
                continue
            adapter_names.add(name)

    for name in adapter_names:
        module = dict(model.named_modules())[name]
        if isinstance(module, (FreqGatedConvLoRA, TaskRoutedConvLoRA, ConvLoRA)):
            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                attr_name = parts[1]
            else:
                parent = model
                attr_name = name

            if merge:
                new_conv = module.merge_weights()
            else:
                if isinstance(module, TaskRoutedConvLoRA):
                    new_conv = module.conv
                elif isinstance(module, FreqGatedConvLoRA):
                    new_conv = module.lora.conv
                else:
                    new_conv = module.conv
                for p in new_conv.parameters():
                    p.requires_grad = True

            replacements.append((parent, attr_name, new_conv))
            removed += 1

    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    # Unfreeze everything after removal
    for p in model.parameters():
        p.requires_grad = True

    if verbose:
        action = "merged" if merge else "restored"
        print(f"[lora] {removed} adapters {action}")

    return removed


def _get_backbone_convs(model, backbone_layers, target_layers):
    """YOLO backbone Conv2d modüllerini bul."""
    results = []

    # YOLO: model.model[0..N]
    model_seq = None
    if hasattr(model, "model") and isinstance(model.model, nn.Sequential):
        model_seq = model.model
    elif hasattr(model, "model") and hasattr(model.model, "__len__"):
        model_seq = model.model

    if model_seq is None:
        return results

    for layer_idx in range(min(backbone_layers, len(model_seq))):
        if target_layers is not None and layer_idx not in target_layers:
            continue

        layer = model_seq[layer_idx]
        for sub_name, sub_module in layer.named_modules():
            if isinstance(sub_module, nn.Conv2d):
                full_name = f"model.{layer_idx}.{sub_name}" if sub_name else f"model.{layer_idx}"

                if sub_name:
                    parts = sub_name.rsplit(".", 1)
                    if len(parts) == 2:
                        parent = dict(layer.named_modules())[parts[0]]
                        attr = parts[1]
                    else:
                        parent = layer
                        attr = sub_name
                else:
                    parent = model_seq
                    attr = str(layer_idx)

                results.append((layer_idx, full_name, parent, attr, sub_module))

    return results


def _freeze_backbone_non_lora(model, backbone_layers):
    """Backbone'daki LoRA olmayan parametreleri dondur."""
    lora_keywords = {"lora_down", "lora_up", "lora_dropout", "gate", "mlp", "branches"}

    for name, param in model.named_parameters():
        # LoRA/gate parametreleri → trainable
        if any(kw in name for kw in lora_keywords):
            param.requires_grad = True
            continue

        # Backbone parametreleri → freeze
        parts = name.split(".")
        if len(parts) >= 2 and parts[0] == "model":
            try:
                layer_idx = int(parts[1])
                if layer_idx < backbone_layers:
                    param.requires_grad = False
            except ValueError:
                pass
