"""
action_head.py

Action head that produces actions from the output of the LLM backbone.
"""

from typing import Optional

import torch
import torch.nn as nn


# === Definitions for Various Action Heads === #
class LinearActionHead(nn.Module):
    def __init__(
            self, 
            llm_dim: int, 
            action_dim: int, 
            use_map: bool, 
            num_map_heads: Optional[int] = None
            ) -> None:
        super().__init__()
        self.action_head = nn.Linear(llm_dim, action_dim, bias=True)
        self.use_map = use_map

        if use_map:
            assert num_map_heads is not None, "Should pass in number of attention heads if 'use_map' is True!"
            self.map_head = MAPHead(llm_dim=llm_dim, num_heads=num_map_heads)

    def forward(self, llm_output: torch.Tensor) -> torch.Tensor:
        if self.use_map:
            pooled_output = self.map_head(llm_output)
        else:
            pooled_output = llm_output.mean(dim=1)

        return self.action_head(pooled_output)
        

class MLPActionHead(nn.Module):
    def __init__(
            self, 
            llm_dim: int, 
            action_dim: int, 
            use_map: bool, 
            num_map_heads: Optional[int] = None, 
            mlp_type: str = "gelu"
            ) -> None:
        super().__init__()
        if mlp_type == "gelu":
            self.action_head = nn.Sequential(
                nn.Linear(llm_dim, action_dim, bias=True),
                nn.GELU(),
                nn.Linear(action_dim, action_dim, bias=True),
            )
        elif mlp_type == "relu":
            self.action_head = nn.Sequential(
                nn.Linear(llm_dim, action_dim, bias=True),
                nn.ReLU(),
                nn.Linear(action_dim, action_dim, bias=True),
            )
        else:
            raise ValueError(f"Action Head with mlp_type = {mlp_type} is not supported!")
        
        self.use_map = use_map
        if use_map:
            assert num_map_heads is not None, "Should pass in number of attention heads if 'use_map' is True!"
            self.map_head = MAPHead(llm_dim=llm_dim, num_heads=num_map_heads)

    def forward(self, llm_output: torch.Tensor) -> torch.Tensor:
        if self.use_map:
            pooled_output = self.map_head(llm_output)
        else:
            pooled_output = llm_output.mean(dim=1)

        return self.action_head(pooled_output)


# === Definitions for Multi-Head Attention Pooling === #
class MLPBlock(nn.Module):
    def __init__(self, llm_dim: int, mlp_dim: Optional[int] = None) -> None:
        super().__init__()
        self.mlp_dim = mlp_dim if mlp_dim is not None else 4 * llm_dim

        self.mlp_block = nn.Sequential(
            nn.Linear(llm_dim, self.mlp_dim, bias=True),
            nn.GELU(),
            nn.Linear(self.mlp_dim, llm_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp_block(x)
   

class MAPHead(nn.Module):
    def __init__(self, llm_dim: int, num_heads: int, mlp_dim: Optional[int] = None) -> None:
        super().__init__()

        # Learnable probe vector
        self.probe = nn.Parameter(torch.randn(1, 1, llm_dim) * 0.02)

        # Multi-head attention
        self.map = nn.MultiheadAttention(embed_dim=llm_dim, num_heads=num_heads, batch_first=True)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(llm_dim)

        # MLP block
        self.mlp_block = MLPBlock(llm_dim=llm_dim, mlp_dim=mlp_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0] # x -> (batch_size, seq_len, embed_dim)
        probe = self.probe.expand(batch_size, -1, -1) # (batch_size, 1, embed_dim)
        attn_output, _ = self.map(probe, x, x) # (batch_size, 1, embed_dim)

        # Layer Norm
        y = self.layer_norm(attn_output)

        # MLP block and residual connection
        output = attn_output + self.mlp_block(y) # (batch_size, 1, embed_dim)

        return output[:, 0, :] # (batch_size, embed_dim)