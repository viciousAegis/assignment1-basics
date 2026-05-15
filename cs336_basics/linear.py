import torch
from torch import nn
from einops import einsum


class Linear(nn.Module):
    def __init__(self, in_feat, out_feat, device=None, dtype=None) -> None:
        super().__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.device = device
        self.dtype = dtype

        std = 2 / (self.in_feat + self.out_feat)
        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(size=(out_feat, in_feat)),
                mean=0.0,
                std=std,
                a=-3 * std,
                b=3 * std,
            )
        )

    def forward(self, x: torch.Tensor):
        return einsum(self.weight, x, "o i, b s i -> b s o")


if __name__ == "__main__":
    in_feat = 4
    out_feat = 2
    x = torch.ones(
        size=(
            1,
            in_feat,
        )
    )
    print(x.shape)
    lin = Linear(in_feat, out_feat)
    out = lin(x)
    print(lin.state_dict().keys())
    print(out, out.shape)
