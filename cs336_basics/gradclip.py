from collections.abc import Iterable
import torch
import math

def gradient_clipping(params: Iterable[torch.nn.Parameter], max_norm):
    squard_grad = 0
    for p in params:
        if p.grad is None:
            continue
        squard_grad += p.grad.square().sum()
    grad_norm = math.sqrt(squard_grad)
    
    if grad_norm <= max_norm:
        return
        
    for p in params:
        if p.grad is None:
            continue
        p.grad = p.grad * (max_norm / (grad_norm + 1e-6))
