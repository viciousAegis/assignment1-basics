import torch
from torch import nn
from einops import einsum, rearrange


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) -> None:
        super().__init__()
        self.n_vocab = num_embeddings
        self.embedding_dim = embedding_dim
        self.device = device
        self.dtype = dtype

        self.weight = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(size=(self.n_vocab, self.embedding_dim)),
                mean=0.0,
                std=1,
                a=-3,
                b=3,
            )
        )
    
    def forward(self, token_ids: torch.Tensor):
        return self.weight[token_ids]
