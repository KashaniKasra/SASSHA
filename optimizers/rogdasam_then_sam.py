#  [NEW]

import torch

from optimizers.rogdasam import RegularizedOGDASAM
from optimizers.sam import SAM


class ROGDASAMThenSAM(torch.optim.Optimizer):
    """
    One optimizer that uses ROGDASAM for the first `rogdasam_epochs`
    epochs, then switches to SAM for the remaining epochs.
    """

    def __init__(
        self,
        params,
        base_optimizer=torch.optim.SGD,
        rogdasam_epochs=5,
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
        rho=0.05,
        lambda_val=0.1,
        eta_eps=1.0,
        adaptive=False,
        **kwargs,
    ):
        params = list(params)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            rho=rho,
            lambda_val=lambda_val,
            eta_eps=eta_eps,
            adaptive=adaptive,
        )

        super().__init__(params, defaults)

        self.rogdasam_epochs = int(rogdasam_epochs)
        self.current_epoch = 0

        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.rho = rho
        self.lambda_val = lambda_val
        self.eta_eps = eta_eps
        self.adaptive = adaptive

        self.rogdasam = RegularizedOGDASAM(
            params,
            base_optimizer=base_optimizer,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            rho=rho,
            lambda_val=lambda_val,
            eta_eps=eta_eps,
        )

        self.sam = SAM(
            params,
            base_optimizer,
            rho=rho,
            adaptive=adaptive,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        self.param_groups = self.sam.param_groups
        self.defaults.update(self.sam.defaults)

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def using_rogdasam(self):
        return self.current_epoch < self.rogdasam_epochs

    def _sync_hyperparams(self):
        for group in self.param_groups:
            lr = group.get("lr", self.lr)
            weight_decay = group.get("weight_decay", self.weight_decay)
            momentum = group.get("momentum", self.momentum)

            for inner in [self.rogdasam, self.sam]:
                for inner_group in inner.param_groups:
                    inner_group["lr"] = lr
                    inner_group["weight_decay"] = weight_decay

                    if "momentum" in inner_group:
                        inner_group["momentum"] = momentum

                    if "rho" in inner_group:
                        inner_group["rho"] = self.rho

    def zero_grad(self, set_to_none=False):
        self.rogdasam.zero_grad(set_to_none=set_to_none)
        self.sam.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("ROGDASAMThenSAM requires a closure")

        self._sync_hyperparams()

        if self.using_rogdasam():
            return self.rogdasam.step(closure)
            
        closure = torch.enable_grad()(closure)

        clean_loss = closure()
        self.sam.first_step(zero_grad=True)

        closure()
        self.sam.second_step(zero_grad=True)

        return clean_loss

    def state_dict(self):
        return {
            "current_epoch": self.current_epoch,
            "rogdasam_epochs": self.rogdasam_epochs,
            "rogdasam": self.rogdasam.state_dict(),
            "sam": self.sam.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self.current_epoch = state_dict.get("current_epoch", 0)
        self.rogdasam_epochs = state_dict.get("rogdasam_epochs", self.rogdasam_epochs)

        if "rogdasam" in state_dict:
            self.rogdasam.load_state_dict(state_dict["rogdasam"])

        if "sam" in state_dict:
            self.sam.load_state_dict(state_dict["sam"])