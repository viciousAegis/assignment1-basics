import torch
from einops import rearrange, einsum

def logsumexp(x: torch.Tensor):
    # apply logsumexp to the vector
    c = x.max(dim=-1, keepdim=True).values
    return c + torch.exp(x - c).sum(dim=-1, keepdim=True).log()

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        logits (torch.Tensor): shape: [... b v]
        targets (torch.Tensor): shape: [... b]
    """
    
    # loss(target) = -log(exp[logits(target)]/sumexp(logits))
    # loss(target) = -logits(target) + logsumexp(logits - max(logits)) + max(logits)
    targets = rearrange(targets, "... -> ... 1")
    class_logits = torch.gather(logits, dim=-1, index=targets)
    
    loss = -class_logits + logsumexp(logits)
    
    return loss.mean()