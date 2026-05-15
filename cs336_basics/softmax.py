import torch

def softmax(x: torch.Tensor, i: int):
    # apply softmax to ith dimension
    exp_vals = torch.exp(x - x.max(dim=i, keepdim=True).values)
    x_sftmx = exp_vals / exp_vals.sum(dim=i, keepdim=True)
    return x_sftmx
