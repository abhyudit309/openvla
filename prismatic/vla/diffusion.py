from typing import Sequence, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> np.ndarray:
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    t = np.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = np.cos((t + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, 0, 0.999)


class FourierFeatures(nn.Module):
    def __init__(self, output_size: int, learnable: bool = True):
        super().__init__()
        self.learnable = learnable

        if learnable:
            self.kernel = nn.Parameter(torch.randn(output_size // 2, 1) * 0.2)
        else:
            half_dim = output_size // 2
            self.f = np.log(10000) / (half_dim - 1)
            self.f = torch.exp(-self.f * torch.arange(half_dim).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == 1
        if self.learnable:
            f = 2 * np.pi * x @ self.kernel.T
        else:
            f = x * self.f
        return torch.cat([torch.cos(f), torch.sin(f)], dim=-1)


class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dims: Sequence[int], 
            activate_final: bool = False, 
            use_layer_norm: bool = False,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.layers = nn.ModuleList()

        for i, size in enumerate(hidden_dims):
            self.layers.append(nn.Linear(hidden_dims[i - 1] if i > 0 else input_dim, size))

            if i + 1 < len(hidden_dims) or activate_final:
                if use_layer_norm:
                    self.layers.append(nn.LayerNorm(size))
                self.layers.append(nn.SiLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim
        for layer in self.layers:
            x = layer(x)
        return x


class MLPResNetBlock(nn.Module):
    def __init__(
            self,
            features: int, 
            activation: Callable = F.silu, 
            use_layer_norm: bool = False,
        ):
        super().__init__()
        self.features = features
        self.activation = activation
        self.use_layer_norm = use_layer_norm

        self.fc1 = nn.Linear(features, features * 4)
        self.fc2 = nn.Linear(features * 4, features)

        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.features
        residual = x

        if self.use_layer_norm:
            x = self.layer_norm(x)
        x = self.activation(self.fc1(x))
        x = self.fc2(x)

        return residual + x


class MLPResNet(nn.Module):
    def __init__(
            self,
            input_dim: int,
            num_blocks: int, 
            out_dim: int, 
            hidden_dim: int = 256, 
            activation: Callable = F.silu, 
            use_layer_norm: bool = False,
        ):
        super().__init__()
        self.input_dim = input_dim
        self.activation = activation

        self.fc = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            MLPResNetBlock(features=hidden_dim, activation=activation, use_layer_norm=use_layer_norm)
            for _ in range(num_blocks)
        ])
        self.out_fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.shape[-1] == self.input_dim

        x = self.fc(x)
        for block in self.blocks:
            x = block(x)
        return self.out_fc(self.activation(x))


class ScoreActor(nn.Module):
    def __init__(
            self, 
            time_preprocess: nn.Module, 
            cond_encoder: nn.Module, 
            reverse_network: nn.Module
        ):
        super().__init__()
        self.time_preprocess = time_preprocess
        self.cond_encoder = cond_encoder
        self.reverse_network = reverse_network

    def forward(
            self, 
            obs_enc: torch.Tensor, 
            actions: torch.Tensor, 
            time: torch.Tensor,
        ) -> torch.Tensor:
        """
        Args:
            obs_enc: (bd..., obs_dim) where bd... is broadcastable to batch_dims
            actions: (batch_dims..., action_dim)
            time: (batch_dims..., 1)
        """
        t_ff = self.time_preprocess(time)
        cond_enc = self.cond_encoder(t_ff)
        
        if obs_enc.shape[:-1] != cond_enc.shape[:-1]:
            new_shape = cond_enc.shape[:-1] + (obs_enc.shape[-1],)
            obs_enc = obs_enc.expand(new_shape)

        reverse_input = torch.cat([cond_enc, obs_enc, actions], dim=-1)
        eps_pred = self.reverse_network(reverse_input)
        return eps_pred


def create_diffusion_model(
    llm_dim: int,
    out_dim: int, 
    time_dim: int, 
    num_blocks: int, 
    hidden_dim: int,
    use_layer_norm: bool,
) -> nn.Module:
    return ScoreActor(
        time_preprocess=FourierFeatures(time_dim, learnable=True),
        cond_encoder=MLP(
            input_dim=time_dim,
            hidden_dims=[2 * time_dim, time_dim]),
        reverse_network=MLPResNet(
            input_dim=time_dim+llm_dim+7,
            num_blocks=num_blocks,
            out_dim=out_dim,
            hidden_dim=hidden_dim,
            use_layer_norm=use_layer_norm,
        ),
    )