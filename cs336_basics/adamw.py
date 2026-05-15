from collections.abc import Callable, Iterable
from typing import Any, Optional, Tuple, List
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: (
            Iterable[torch.Tensor]
            | Iterable[dict[str, Any]]
            | Iterable[tuple[str, torch.Tensor]]
        ),
        lr: float,
        betas: Tuple[float, float],
        eps: float,
        weight_decay: float,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "gamma": weight_decay}
        super().__init__(params, defaults)
        # for group in self.param_groups:
        #     for p in group["params"]:
        #         self.state[p]["m"] = torch.zeros_like(p)
        #         self.state[p]["v"] = torch.zeros_like(p)

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            gamma = group["gamma"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]  # Get state associated with p.
                t = state.get("t", 1)  # Get iteration number from the state, or 1.
                m = state.get(
                    "m", torch.zeros_like(p)
                )  # get moment estimate, or start from 0
                v = state.get(
                    "v", torch.zeros_like(p)
                )  # get second moment est, or start from zero
                grad = p.grad.data  # Get the gradient of loss with respect to p.
                lr_t = lr * (math.sqrt(1 - (beta2) ** t)) / (1 - (beta1) ** t)

                p.data -= lr * gamma * p.data  # weight decay

                m = (beta1 * m) + ((1 - beta1) * grad)
                v = (beta2 * v) + ((1 - beta2) * grad**2)
                state["m"] = m
                state["v"] = v

                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state["t"] = t + 1
        return loss


if __name__ == "__main__":
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    # opt = SGD([weights], lr=1)
    # for t in range(100):
    #     opt.zero_grad()  # Reset the gradients for all learnable parameters.
    #     loss = (weights**2).mean()  # Compute a scalar loss value.
    #     print(loss.cpu().item())
    #     loss.backward()  # Run backward pass, which computes gradients.
    #     opt.step()  # Run optimizer step.
    opt = AdamW([weights], lr=1, weight_decay=0.99, betas=(0.99, 0.999), eps=1e-6)
    for t in range(100):
        opt.zero_grad()  # Reset the gradients for all learnable parameters.
        loss = (weights**2).mean()  # Compute a scalar loss value.
        print(loss.cpu().item())
        loss.backward()  # Run backward pass, which computes gradients.
        opt.step()  # Run optimizer step.
