"""
action_head.py

Action head that produces actions from the output of the LLM backbone.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from .diffusion import cosine_beta_schedule, create_diffusion_model


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

    def predict_action(self, llm_output: torch.Tensor) -> np.ndarray:
        assert llm_output.shape[0] == 1
        pred_actions = self(llm_output)
        return pred_actions[0].to(dtype=torch.float32).cpu().numpy()


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

    def predict_action(self, llm_output: torch.Tensor) -> np.ndarray:
        assert llm_output.shape[0] == 1
        pred_actions = self(llm_output)
        return pred_actions[0].to(dtype=torch.float32).cpu().numpy()


class DiffusionActionHead(nn.Module):
    def __init__(
            self,
            llm_dim: int,
            action_dim: int,
            time_dim: int = 32,
            num_blocks: int = 3,
            hidden_dim: int = 256,
            use_layer_norm: bool = True,
            diffusion_steps: int = 20,
            n_diffusion_samples: int = 1,
            use_map: bool = True,
            num_map_heads: Optional[int] = None,
            seed: int = 7,
        ) -> None:
        super().__init__()

        self.action_dim = action_dim
        self.diffusion_steps = diffusion_steps
        self.n_diffusion_samples = n_diffusion_samples
        self.use_map = use_map

        if use_map:
            assert num_map_heads is not None, "Should pass in number of attention heads if 'use_map' is True!"
            self.map_head = MAPHead(llm_dim=llm_dim, num_heads=num_map_heads)

        # create the diffusion model (score network)
        self.diffusion_model = create_diffusion_model(
            llm_dim=llm_dim,
            out_dim=action_dim,
            time_dim=time_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        )

        # create beta schedule
        self.betas = torch.tensor(cosine_beta_schedule(diffusion_steps), dtype=torch.bfloat16, device='cuda')
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

        self.rng = torch.Generator(device='cuda').manual_seed(seed)

    def forward(
            self, 
            llm_output: torch.Tensor,
            time: Optional[torch.Tensor] = None,
            noisy_actions: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:

        if self.use_map:
            pooled_output = self.map_head(llm_output)
        else:
            pooled_output = llm_output.mean(dim=1)

        # time and noisy_actions are None during initialization, so we replace them with a dummy array
        if time is None or noisy_actions is None:
            if not self.training:
                raise ValueError("Must provide time and noisy_actions when calling diffusion action head")
            else:
                time = torch.zeros((pooled_output.shape[0], 1), dtype=torch.bfloat16)
                noisy_actions = torch.zeros((pooled_output.shape[0], self.action_dim), dtype=torch.bfloat16)

        pred_eps = self.diffusion_model(pooled_output, noisy_actions, time)
        return pred_eps

    def predict_action(self, llm_output: torch.Tensor) -> np.ndarray:
        batch_size = llm_output.shape[0]
        assert batch_size == 1

        def scan_fn(current_x: torch.Tensor, time: torch.Tensor):
            input_time = time.expand(size=(batch_size, 1)).to(dtype=torch.bfloat16)

            pred_eps = self(llm_output, time=input_time, noisy_actions=current_x)

            alpha_1 = 1.0 / torch.sqrt(self.alphas[time])
            alpha_2 = (1.0 - self.alphas[time]) / (torch.sqrt(1.0 - self.alpha_hats[time]))
            current_x = alpha_1 * (current_x - alpha_2 * pred_eps)

            z = torch.randn(size=current_x.shape, generator=self.rng, device='cuda', dtype=torch.bfloat16)
            current_x = current_x + (time > 0) * (torch.sqrt(self.betas[time]) * z)

            return current_x

        noise = torch.randn(size=(batch_size, self.action_dim), generator=self.rng, device='cuda', dtype=torch.bfloat16)
        actions = noise

        for t in reversed(range(self.diffusion_steps)):
            actions = scan_fn(actions, torch.tensor(t, dtype=torch.long, device='cuda'))

        return actions[0].to(dtype=torch.float32).cpu().numpy()


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