"""
action_head.py

Action head that produces actions from the output of the LLM backbone.
"""

from typing import Optional

import torch
import torch.nn as nn

from diffusion import cosine_beta_schedule, create_diffusion_model


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


class DiffusionActionHead(nn.Module):
    def __init__(
            self,
            llm_dim: int,
            time_dim: int = 32,
            num_blocks: int = 3,
            hidden_dim: int = 256,
            use_layer_norm: bool = True,
            diffusion_steps: int = 20,
            n_diffusion_samples: int = 1,
            use_map: bool = True,
            num_map_heads: Optional[int] = None,
        ) -> None:

        self.diffusion_steps = diffusion_steps
        self.n_diffusion_samples = n_diffusion_samples
        self.use_map = use_map

        if use_map:
            assert num_map_heads is not None, "Should pass in number of attention heads if 'use_map' is True!"
            self.map_head = MAPHead(llm_dim=llm_dim, num_heads=num_map_heads)

        # create the diffusion model (score network)
        self.diffusion_model = create_diffusion_model(
            llm_dim=llm_dim,
            out_dim=7,
            time_dim=time_dim,
            num_blocks=num_blocks,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        )

        # create beta schedule
        self.betas = torch.tensor(cosine_beta_schedule(diffusion_steps), dtype=torch.float32)
        self.alphas = 1.0 - self.betas
        self.alpha_hats = torch.cumprod(self.alphas, dim=0)

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
                time = torch.zeros((pooled_output.shape[0], 1), dtype=torch.float32)
                noisy_actions = torch.zeros((pooled_output.shape[0], 7), dtype=torch.float32)

        pred_eps = self.diffusion_model(pooled_output, noisy_actions, time)
        return pred_eps

    def loss(self, llm_output: torch.Tensor, gt_actions: torch.Tensor, seed: int) -> torch.Tensor:
        """
        Computes the loss for the diffusion objective.
        """
        batch_size = llm_output.shape[0]

        # Generate random time and noise
        rng = torch.Generator().manual_seed(seed)
        time = torch.randint(0, self.diffusion_steps, (self.n_diffusion_samples, batch_size, 1), generator=rng)
        noise = torch.randn((self.n_diffusion_samples, batch_size, 7), generator=rng)

        # Add noise to the action according to the schedule
        scale = torch.sqrt(self.alpha_hats[time])
        std = torch.sqrt(1 - self.alpha_hats[time])
        noisy_actions = scale * gt_actions.unsqueeze(0) + std * noise

        pred_eps = self(llm_output, time=time, noisy_actions=noisy_actions)
        loss = torch.nn.functional.mse_loss(pred_eps, noise)

        return loss

    def predict_action(self, llm_output: torch.Tensor, seed: int = 7):
        rng = torch.Generator().manual_seed(seed)
        batch_size = llm_output.shape[0]
        assert batch_size == 1

        def scan_fn(current_x: torch.Tensor, time: torch.Tensor):
            input_time = time.expand(size=(batch_size, 1))

            eps_pred = self(llm_output, time=input_time, noisy_actions=current_x)

            alpha_1 = 1.0 / torch.sqrt(self.alphas[time])
            alpha_2 = (1.0 - self.alphas[time]) / (torch.sqrt(1.0 - self.alpha_hats[time]))
            current_x = alpha_1 * (current_x - alpha_2 * eps_pred)

            z = torch.randn(size=current_x.shape, generator=rng)
            current_x = current_x + (time > 0) * (torch.sqrt(self.betas[time]) * z)

            return current_x

        noise = torch.randn(size=(batch_size, 7), generator=rng)
        actions = noise

        for t in reversed(range(self.diffusion_steps)):
            actions = scan_fn(actions, torch.tensor(t, dtype=torch.long))

        return actions[0]


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