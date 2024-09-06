"""
action_head.py

Action head that produces actions from the output of the LLM backbone.
"""

import torch
import torch.nn as nn


# === Definitions for Various Action Heads === #
class LinearActionHead(nn.Module):
    def __init__(self, llm_dim: int, action_dim: int) -> None:
        super().__init__()
        self.action_head = nn.Linear(llm_dim, action_dim, bias=True)

    def forward(self, llm_output: torch.Tensor) -> torch.Tensor:
        return self.action_head(llm_output)
        

class MLPActionHead(nn.Module):
    def __init__(self, llm_dim: int, action_dim: int, mlp_type: str = "gelu") -> None:
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

    def forward(self, llm_output: torch.Tensor) -> torch.Tensor:
        return self.action_head(llm_output)