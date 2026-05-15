import torch
from torch import nn
from einops import reduce, repeat


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device=None, dtype=None) -> None:
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(torch.ones(size=(d_model,)))

    def forward(self, x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms_x = torch.sqrt(
            reduce(torch.square(x), "b s d -> b s 1", reduction="mean") + self.eps
        )

        x = (x / rms_x) * self.weight

        return x.to(in_dtype)


if __name__ == "__main__":
    batch_size = 4
    seq_len = 12
    d_model = 64

    x = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 2.0]],
            [[-1.0, -2.0, -3.0, -4.0], [1.0, -1.0, 1.0, -1.0]],
        ]
    )

    rmsnorm = RMSNorm(d_model=4)
    out = rmsnorm(x)

    print(out)

    print("input shape:", x.shape)
    print("output shape:", out.shape)
