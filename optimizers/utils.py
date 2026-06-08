# [NEW]

from __future__ import annotations

import torch


def get_global_norm(tensors: list[torch.Tensor]) -> torch.Tensor:
    if len(tensors) == 0:
        return torch.tensor(0.0)

    device = tensors[0].device
    norms = [
        torch.norm(t.detach(), p=2, dtype=torch.float32).to(device)
        for t in tensors
    ]
    return torch.norm(torch.stack(norms), p=2)