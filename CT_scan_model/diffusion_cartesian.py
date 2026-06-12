"""Mask-aware DDPM diffusion utilities for cartesian CT.

Diffusion is applied only to the image channel if input is [image, mask].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch


def _cosine_beta_schedule(noise_steps: int, s: float = 0.008, max_beta: float = 0.999) -> torch.Tensor:
    """Cosine schedule for alpha_hat (Nichol & Dhariwal, 2021).

    Spends relatively more steps in the low-noise regime than a linear
    schedule, which helps preserve high-frequency detail (fine winding
    lines, CT grain texture, sharp can edge).
    """
    steps = noise_steps + 1
    t = torch.linspace(0, noise_steps, steps, dtype=torch.float64) / noise_steps
    alphas_cumprod = torch.cos(((t + s) / (1.0 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, max_beta).to(dtype=torch.float32)


@dataclass
class Diffusion:
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    schedule: str = "linear"

    def __post_init__(self):
        if self.schedule == "cosine":
            self.beta = _cosine_beta_schedule(self.noise_steps)
        elif self.schedule == "linear":
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule!r} (expected 'linear' or 'cosine')")
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def to(self, device: torch.device) -> "Diffusion":
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        return self

    def sample_timesteps(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=device)

    def noise_images(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise at timestep t.

        If x has channels [image, mask], only diffuse the image channel.
        Returns (x_t, epsilon) where epsilon is the noise added to the image channel.
        """

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]

        if x.dim() == 4 and x.shape[1] >= 2:
            # Channel 0 = image (noised), channels 1+ = mask/radial map (kept clean)
            x_img  = x[:, :1]
            x_rest = x[:, 1:]
            eps = torch.randn_like(x_img)
            x_noised = sqrt_alpha_hat * x_img + sqrt_one_minus_alpha_hat * eps
            return torch.cat([x_noised, x_rest], dim=1), eps

        eps = torch.randn_like(x)
        x_noised = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        return x_noised, eps
