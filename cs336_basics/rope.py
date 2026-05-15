import torch
from torch import nn
from einops import einsum, rearrange


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device

        positions = rearrange(torch.arange(max_seq_len), "i -> i 1")
        pair_ids = rearrange(torch.arange(d_k // 2), "k -> 1 k")
        angles = positions / theta ** (2 * pair_ids / d_k)

        cos_table = torch.cos(angles)
        sin_table = torch.sin(angles)

        self.register_buffer("cos_table", cos_table, persistent=False)
        self.register_buffer("sin_table", sin_table, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_table[: token_positions.size(-1)]  # type: ignore
        sin = self.sin_table[: token_positions.size(-1)]  # type: ignore

        # size: [seqlen, dim_k]

        # input: [batch, seq_len, d_k]

        x_pairs = rearrange(x, "... (d two) -> ... d two", two=2)
        x_even = x_pairs[..., 0]
        x_odd = x_pairs[..., 1]

        rot_even = x_even * cos - sin * x_odd
        rot_odd = x_even * sin + cos * x_odd

        x_pairs[..., 0] = rot_even
        x_pairs[..., 1] = rot_odd

        x = rearrange(x_pairs, "... d two -> ... (d two)", two=2)

        return x


if __name__ == "__main__":
    rope = RotaryPositionalEmbedding(
        theta=10_000.0,
        d_k=8,
        max_seq_len=16,
    )

    # batch=2, seq_len=4, d_k=8
    x = torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)

    token_positions = torch.arange(4)

    out = rope(x, token_positions)

    print("x shape:", x.shape)
    print("token_positions shape:", token_positions.shape)
    print("cos_table shape:", rope.cos_table.shape)
    print("sin_table shape:", rope.sin_table.shape)
    print("out:", out)
