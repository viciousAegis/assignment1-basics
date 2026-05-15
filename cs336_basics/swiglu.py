import torch
from torch import nn
from einops import reduce, repeat, einsum
from cs336_basics.linear import Linear


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff = None) -> None:
        super().__init__()
        self.d_model = d_model
        if d_ff:
            self.d_ff = d_ff
        else:
            self.d_ff = int((((8 / 3) * d_model) // 64) * 64)
        self.w1 = Linear(self.d_model, self.d_ff)
        self.w2 = Linear(self.d_ff, self.d_model)
        self.w3 = Linear(self.d_model, self.d_ff)

    def silu(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor):
        return self.w2(self.silu(self.w1(x)) * self.w3(x))


if __name__ == "__main__":
    swiglu = SwiGLU(d_model= 512)
    print(swiglu.state_dict())