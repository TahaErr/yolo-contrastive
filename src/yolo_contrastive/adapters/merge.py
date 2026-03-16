"""Merge stratejileri — Task-Routed LoRA branch'lerini birleştirme.

Stratejiler:
    equal:      ΔW = (1/N) × Σ(branch_i)
    weighted:   ΔW = Σ(αᵢ × branch_i), α önceden belirlenir
    task_loss:  α_i ∝ 1/loss_i (zor task → daha fazla katkı)
    learned:    α'lar validation loss ile optimize edilir (gelecek)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .task_routed_lora import TaskRoutedConvLoRA
from .freq_gated_lora import FreqGatedConvLoRA
from .conv_lora import ConvLoRA


def compute_merge_alphas(
    strategy: str = "equal",
    num_tasks: int = 3,
    task_losses: Optional[Dict[str, float]] = None,
    custom_alphas: Optional[List[float]] = None,
    task_names: Optional[List[str]] = None,
) -> List[float]:
    """Merge alpha değerlerini hesapla.

    Args:
        strategy: "equal", "weighted", "task_loss"
        num_tasks: Task sayısı
        task_losses: {task_name: final_loss} — task_loss stratejisi için
        custom_alphas: Önceden belirlenmiş alpha'lar — weighted stratejisi için
        task_names: Task isimleri (sıralama için)

    Returns:
        [α₁, α₂, ..., αₙ] — normalize edilmiş ağırlıklar
    """
    if strategy == "equal":
        return [1.0 / num_tasks] * num_tasks

    elif strategy == "weighted":
        if custom_alphas is None:
            raise ValueError("weighted stratejisi için custom_alphas gerekli")
        if len(custom_alphas) != num_tasks:
            raise ValueError(f"custom_alphas ({len(custom_alphas)}) != num_tasks ({num_tasks})")
        total = sum(custom_alphas)
        return [a / total for a in custom_alphas]

    elif strategy == "task_loss":
        if task_losses is None:
            raise ValueError("task_loss stratejisi için task_losses gerekli")
        if task_names is None:
            task_names = list(task_losses.keys())

        # Inverse loss weighting: düşük loss → düşük ağırlık (zaten iyi)
        # Yüksek loss → yüksek ağırlık (daha fazla adaptasyon gerekli)
        losses = [max(task_losses.get(n, 1.0), 1e-6) for n in task_names]
        inv = [1.0 / l for l in losses]
        total = sum(inv)
        return [i / total for i in inv]

    else:
        raise ValueError(f"Unknown merge strategy: {strategy}. "
                         f"Available: equal, weighted, task_loss")


def merge_task_routed_model(
    model: nn.Module,
    strategy: str = "equal",
    alphas: Optional[List[float]] = None,
    task_losses: Optional[Dict[str, float]] = None,
    task_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> int:
    """Model'deki tüm TaskRoutedConvLoRA'ları merge et.

    Args:
        model: LoRA enjekte edilmiş model
        strategy: Merge stratejisi
        alphas: weighted stratejisi için alpha'lar
        task_losses: task_loss stratejisi için son epoch loss'ları
        task_names: Task isimleri (sıralama için)
        verbose: Loglama

    Returns:
        Merge edilen adapter sayısı
    """
    merged = 0
    replacements = []
    merge_alphas = None

    for name, module in list(model.named_modules()):
        if isinstance(module, TaskRoutedConvLoRA):
            if merge_alphas is None:
                merge_alphas = compute_merge_alphas(
                    strategy=strategy,
                    num_tasks=module.num_tasks,
                    task_losses=task_losses,
                    custom_alphas=alphas,
                    task_names=task_names,
                )
                if verbose:
                    print(f"[merge] Strategy: {strategy}")
                    print(f"[merge] Alphas: {[f'{a:.3f}' for a in merge_alphas]}")

            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                attr_name = parts[1]
            else:
                parent = model
                attr_name = name

            merged_conv = module.merge_weights(alphas=merge_alphas)
            replacements.append((parent, attr_name, merged_conv))
            merged += 1

    for parent, attr_name, new_module in replacements:
        setattr(parent, attr_name, new_module)

    # Unfreeze all
    for p in model.parameters():
        p.requires_grad = True

    if verbose:
        print(f"[merge] {merged} adapters merged")

    return merged
