from __future__ import annotations

import torch
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention layer used in the Q-Former implementation."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        attn_out, _ = self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)
        x = x + self.dropout(attn_out)
        x = self.norm(x)
        return x


class CrossAttention(nn.Module):
    """Cross-attention layer where *query* attends to *context* (key/value)."""

    def __init__(self, query_dim: int, context_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # For simplicity, project context to query dim if different
        self.key_proj = nn.Linear(context_dim, query_dim, bias=False) if context_dim != query_dim else nn.Identity()
        self.value_proj = nn.Linear(context_dim, query_dim, bias=False) if context_dim != query_dim else nn.Identity()
        self.attn = nn.MultiheadAttention(query_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(query_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor):
        # context projected if necessary
        k = self.key_proj(context)
        v = self.value_proj(context)
        attn_out, _ = self.attn(query, k, v, need_weights=False)
        query = query + self.dropout(attn_out)
        query = self.norm(query)
        return query


class FeedForward(nn.Module):
    """Position-wise feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        return self.norm(x + self.net(x))


class QFormerBlock(nn.Module):
    """A single block consisting of cross-attention, self-attention, and feed-forward."""

    def __init__(self, embed_dim: int, context_dim: int, num_heads: int, mlp_dim: int, dropout: float):
        super().__init__()
        self.cross_attn = CrossAttention(embed_dim, context_dim, num_heads, dropout)
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.ffn = FeedForward(embed_dim, mlp_dim, dropout)

    def forward(self, qry: torch.Tensor, context: torch.Tensor):
        qry = self.cross_attn(qry, context)
        qry = self.self_attn(qry)
        qry = self.ffn(qry)
        return qry


class QFormer(nn.Module):
    """Trainable Q-Former module as introduced in BLIP-2.

    The module maintains *num_query_tokens* learnable queries that interact with
    frozen *context* (visual) tokens through a stack of *num_layers* alternating
    cross-attention and self-attention blocks.
    """

    def __init__(
        self,
        context_dim: int,
        num_query_tokens: int = 16,
        embed_dim: int | None = None,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_dim: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        embed_dim = embed_dim or context_dim
        self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, embed_dim) * 0.02)
        self.layers = nn.ModuleList([
            QFormerBlock(embed_dim, context_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Run Q-Former.

        Parameters
        ----------
        context : torch.Tensor
            Visual tokens of shape (B, N, C).
        Returns
        -------
        torch.Tensor
            Query representations of shape (B, num_query_tokens, C).
        """
        B = context.size(0)
        query = self.query_tokens.expand(B, -1, -1)
        for layer in self.layers:
            query = layer(query, context)
        return query