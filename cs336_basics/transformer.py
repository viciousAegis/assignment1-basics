import torch
from torch import nn

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.swiglu import SwiGLU
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model=d_model)
        self.ln2 = RMSNorm(d_model=d_model)
        self.attn = MultiHeadSelfAttention(
            d_model=d_model, num_heads=num_heads, theta=theta, max_seq_len=max_seq_len
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x: torch.Tensor):
        token_positions = torch.arange(x.shape[-2], device=x.device).expand(
            x.shape[:-1]
        )
        z = x + self.attn(self.ln1(x), token_positions)
        out = z + self.ffn(self.ln2(z))
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        theta: float | None = None,
    ) -> None:
        super().__init__()
        self.token_embeddings = Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=theta,
                    max_seq_len=context_length,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model=d_model)
        self.lm_head = Linear(in_feat=d_model, out_feat=vocab_size)

    def forward(self, token_ids):
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        x = self.lm_head(x)
        return x
