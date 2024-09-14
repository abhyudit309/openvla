"""
layer_output_pooler.py

Layer Output Pooler that pools the hidden states of the LLM backbone.
"""

from typing import Optional

import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, llm_dim: int, mlp_dim: Optional[int], mlp_type: str) -> None:
        super().__init__()
        self.mlp_dim = mlp_dim if mlp_dim is not None else 4 * llm_dim

        if mlp_type == "linear":
            self.mlp_block = nn.Linear(llm_dim, llm_dim, bias=True)

        elif mlp_type == "relu":
            self.mlp_block = nn.Sequential(
                nn.Linear(llm_dim, self.mlp_dim, bias=True),
                nn.ReLU(),
                nn.Linear(self.mlp_dim, llm_dim, bias=True),
            )

        elif mlp_type == "gelu":
            self.mlp_block = nn.Sequential(
                nn.Linear(llm_dim, self.mlp_dim, bias=True),
                nn.GELU(),
                nn.Linear(self.mlp_dim, llm_dim, bias=True),
            )

        else:
            raise ValueError(f"LayerOutputPooler with `{mlp_type = }` is not supported!")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_block(x)


class LayerOutputPooler(nn.Module):
    def __init__(
            self, 
            llm_dim: int, 
            num_heads: int, 
            mlp_dim: Optional[int] = None, 
            mlp_type: str = "linear",
        ) -> None:
        super().__init__()

        # Learnable probe vector
        self.probe = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)

        # Multi-head attention
        self.map = nn.MultiheadAttention(embed_dim=llm_dim, num_heads=num_heads, batch_first=True)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(llm_dim)

        # MLP block
        self.mlp_block = MLPBlock(llm_dim=llm_dim, mlp_dim=mlp_dim, mlp_type=mlp_type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4

        batch_size, seq_len, num_layers, embed_dim = x.shape
        x = x.view(batch_size * seq_len, num_layers, embed_dim)

        probe = self.probe.expand(batch_size * seq_len, -1, -1) # (batch_size * seq_len, 1, embed_dim)
        attn_output, _ = self.map(probe, x, x) # (batch_size * seq_len, 1, embed_dim)

        # Layer Norm
        y = self.layer_norm(attn_output)

        # MLP block and residual connection
        output = attn_output + self.mlp_block(y) # (batch_size * seq_len, 1, embed_dim)

        output = output.view(batch_size, seq_len, embed_dim)
        return output