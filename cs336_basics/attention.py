import torch
from torch import nn
from einops import einsum, rearrange
import math
from cs336_basics.softmax import softmax
from cs336_basics.linear import Linear
from cs336_basics.rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(
    Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask
):
    """
    scaled dot product attention

    Args:
        Q: B,..., S, D_k
        K: B,..., S, D_k
        V: B,..., S, D_k/D_v
        mask: S,S
    """

    score_unmasked = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(
        Q.shape[-1]
    )
    score = torch.where(mask, score_unmasked, -float("inf"))

    attention = softmax(score, -1)
    return einsum(attention, V, "... q k, ... k d -> ... q d")


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, theta=None, max_seq_len=None):
        super().__init__()
        self.d_k = d_model // num_heads
        self.h = num_heads

        self.rope = None
        if theta and max_seq_len:
            self.rope = RotaryPositionalEmbedding(
                theta=theta, max_seq_len=max_seq_len, d_k=self.d_k
            )

        # params
        self.q_proj = Linear(self.h * self.d_k, d_model)
        self.k_proj = Linear(self.h * self.d_k, d_model)
        self.v_proj = Linear(self.h * self.d_k, d_model)
        self.output_proj = Linear(d_model, self.h * self.d_k)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None):
        # x.shape = [..., s, d_model]
        Q = self.q_proj(x)  # shape = [..., s, h * d_k]
        K = self.k_proj(x)  # shape = [..., s, h * d_k]
        V = self.v_proj(x)  # shape = [..., s, h * d_k]

        Q = rearrange(Q, "... s (h d_k) -> ... h s d_k", h=self.h)
        K = rearrange(K, "... s (h d_k) -> ... h s d_k", h=self.h)
        V = rearrange(V, "... s (h d_k) -> ... h s d_k", h=self.h)

        if self.rope and token_positions is not None:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        causal_mask = torch.ones(
            size=(x.size(-2), x.size(-2)), dtype=torch.bool, device=x.device
        ).tril(diagonal=0)

        out = scaled_dot_product_attention(Q, K, V, causal_mask)  # [... s d_k]
        out = rearrange(out, "... h s d_k -> ... s (h d_k)")
        O = self.output_proj(out)
        return O
