# [NEW]

from __future__ import annotations

from typing import Callable, Optional

import torch

from optimizers.utils import get_global_norm


class RegularizedOGDASAM(torch.optim.Optimizer):
    """
    Regularized OGDA-SAM optimizer.

    Steps:
    1. Compute clean gradient.
    2. Compute d = grad - lambda_val * eps.
    3. Update perturbation using OGDA-style eps update.
    4. Project perturbation to global radius rho.
    5. Perturb parameters.
    6. Recompute adversarial gradient.
    7. Restore parameters.
    8. Apply base optimizer step using adversarial gradient.
    """

    def __init__(
        self,
        params,
        base_optimizer,
        rho: float = 0.05,
        lambda_val: float = 0.1,
        eta_eps: float = 1.0,
        **kwargs,
    ) -> None:
        if rho < 0.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if eta_eps < 0.0:
            raise ValueError(f"Invalid eta_eps value: {eta_eps}")
        if lambda_val < 0.0:
            raise ValueError(f"Invalid lambda_val value: {lambda_val}")

        defaults = dict(
            rho=rho,
            lambda_val=lambda_val,
            eta_eps=eta_eps,
            **kwargs,
        )
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

        self.rho = rho
        self.lambda_val = lambda_val
        self.eta_eps = eta_eps

    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None):
        if closure is None:
            raise ValueError("RegularizedOGDASAM requires a closure")

        closure = torch.enable_grad()(closure)

        clean_loss = closure()

        active_params = [
            p
            for group in self.param_groups
            for p in group["params"]
            if p.grad is not None
        ]

        if len(active_params) == 0:
            return clean_loss

        eps_unproj_list = []

        for p in active_params:
            state = self.state[p]

            if len(state) == 0:
                state["eps"] = torch.zeros_like(p.data)
                state["past_d"] = torch.zeros_like(p.data)

            d = p.grad.detach() - self.lambda_val * state["eps"]
            state["temp_d"] = d.detach()

            eps_unproj = state["eps"] + self.eta_eps * (2.0 * d - state["past_d"])
            eps_unproj_list.append(eps_unproj)

        # PROJECT
        global_norm = get_global_norm(eps_unproj_list)
        scale = self.rho / (global_norm.to(active_params[0].device) + 1e-12)
        # PROJECT

        eps_next_list = []

        for p, eps_unproj in zip(active_params, eps_unproj_list):
            eps_next = eps_unproj * scale # eps_next = eps_unproj 
            eps_next_list.append(eps_next)

            p.data.add_(eps_next)

            state = self.state[p]
            state["eps"] = eps_next.detach()
            state["past_d"] = state.pop("temp_d")

        self.zero_grad()
        closure()

        for p, eps_next in zip(active_params, eps_next_list):
            p.data.sub_(eps_next)

        self.base_optimizer.step()
        return clean_loss

    def zero_grad(self, set_to_none: bool = False) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)