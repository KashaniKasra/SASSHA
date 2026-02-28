# Acknowledgement: This code is based on https://github.com/davda54/sam/blob/main/sam.py
import torch

# [CHANGED]
class SAM(torch.optim.Optimizer):
    def __init__(self, model, base_optimizer, rho=0.05, adaptive=False, hvp_every=1, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        assert hvp_every >= 1, f"Invalid hvp_every, should be >= 1: {hvp_every}"

        self.model = model
        self.param_names = []
        params = []
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.param_names.append(name)
                params.append(p)

        defaults = dict(rho=rho, adaptive=adaptive, hvp_every=hvp_every, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self._global_step = 0
        self._cached_proj = None

    # Dot product <a, b> across param tensors
    def _global_dot(self, a_list, b_list):
        s = None
        for a, b in zip(a_list, b_list):
            if a is None or b is None:
                continue

            ab = (a * b).sum()
            s = ab if s is None else (s + ab)

        if s is None:
            device = self.param_groups[0]["params"][0].device

            return torch.tensor(0.0, device=device)
        
        return s
    
    # Expand the projector times: (I/||g|| - gg^T/||g||^3) Hv = Hv/||g|| - g(g^T Hv)/||g||^3
    def _expanded_projector_times(self, Hv, g, g_norm):
        dot = self._global_dot(g, Hv)  # g^T Hv
        g_norm = g_norm + 1e-12
        g_norm3 = g_norm ** 3

        out = []
        for g_i, hv_i in zip(g, Hv):
            if hv_i is None:
                out.append(None)

                continue

            if g_i is None:
                out.append(hv_i / g_norm)

                continue

            out.append(hv_i / g_norm - g_i * (dot / g_norm3))

        return out
    
    # Gradient norm ||g|| from supplied grads
    def _grad_norm_from_grads(self, grads):
        shared_device = self.param_groups[0]["params"][0].device

        norms = []
        for group in self.param_groups:
            adaptive = group["adaptive"]
            for p, g in zip(group["params"], grads):
                if g is None:
                    continue

                scale = torch.abs(p) if adaptive else 1.0
                norms.append((scale * g).norm(p=2).to(shared_device))

        if len(norms) == 0:
            return torch.tensor(0.0, device=shared_device)
        
        return torch.norm(torch.stack(norms), p=2)
    
    # Step
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        # 1) Compute L(w) = loss(w) and g(w) = ∇L(w)
        loss = closure(None)

        params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    params.append(p)

        g = torch.autograd.grad(
            loss, params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )

        # replace None grads with 0 connected to p
        g = [gi if gi is not None else (p * 0.0) for p, gi in zip(params, g)]

        g_norm = self._grad_norm_from_grads(g)


        # 2) Build perturbed params OUT-OF-PLACE
        perturbed_params = []
        perturbed_dict = {}
        idx = 0
        for group in self.param_groups:
            rho = group["rho"]
            adaptive = group["adaptive"]
            scale = rho / (g_norm + 1e-12)

            for p in group["params"]:
                if not p.requires_grad:
                    continue

                gi = g[idx]
                idx += 1

                ew = ((p.pow(2) if adaptive else 1.0) * gi) * scale.to(p)
                p_pert = (p + ew).detach().requires_grad_(True)

                perturbed_params.append(p_pert)
                perturbed_dict[self.param_names[len(perturbed_params) - 1]] = p_pert


        # 3) L(w + eps) and v = grad wrt perturbed params
        loss_pert = closure(perturbed_dict)

        v = torch.autograd.grad(
            loss_pert, perturbed_params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )
        v = [vi if vi is not None else (pp * 0.0) for pp, vi in zip(perturbed_params, v)]

        self._global_step += 1
        group0 = self.param_groups[0]
        hvp_every = int(group0.get("hvp_every", 1))
        do_hvp = (hvp_every <= 1) or (self._global_step % hvp_every == 0)

        # 4) Hv + projector term (CONDITIONAL)
        proj_Hv = None

        if do_hvp:
            Hv = torch.autograd.grad(
                g, params,
                grad_outputs=v,
                create_graph=False,
                retain_graph=False,
                allow_unused=True
            )
            Hv = [hvi if hvi is not None else (p * 0.0) for p, hvi in zip(params, Hv)]

            proj_Hv = self._expanded_projector_times(Hv, g, g_norm)

        final_grads = []
        idx = 0
        for group in self.param_groups:
            rho = group["rho"]
            for p in group["params"]:
                if not p.requires_grad:
                    continue

                vi = v[idx]

                if proj_Hv is None:
                    final_grads.append(vi)
                else:
                    proj_i = proj_Hv[idx]
                    final_grads.append(vi + rho * proj_i)

                idx += 1

        # 5) Write grads to real params and step
        for p, fg in zip(params, final_grads):
            p.grad = fg

        self.base_optimizer.step()
        self.zero_grad(set_to_none=True)

        return loss

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
