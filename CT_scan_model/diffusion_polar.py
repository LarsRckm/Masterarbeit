"""Mask-aware DDPM diffusion utilities for polar CT.

This version does not depend on global variables from configuration.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

try:
    from model import config as project_config
except ImportError:  # pragma: no cover
    import config as project_config  # type: ignore


@dataclass
class Diffusion:
    noise_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

    def __post_init__(self):
        self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
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

        if x.dim() == 4 and x.shape[1] == 2:
            x_img, x_mask = x[:, :1], x[:, 1:]
            eps = torch.randn_like(x_img)
            x_noised = sqrt_alpha_hat * x_img + sqrt_one_minus_alpha_hat * eps
            return torch.cat([x_noised, x_mask], dim=1), eps

        eps = torch.randn_like(x)
        x_noised = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        return x_noised, eps

    def sample(
        self,
        model,
        n: int,
        cond,
        device: torch.device,
        cfg_scale: float = 0.0,
    ) -> torch.Tensor:
        """Generate samples of shape [n,1,R,Theta].

        Mask is currently set to ones (full valid radius). For radius-aware sampling,
        build a mask from cond and pass it as part of the model input.
        """
        model.eval()
        r = int(project_config.POLAR_R_MODEL)
        th = int(project_config.POLAR_THETA_BINS)
        with torch.no_grad():
            x_img = torch.randn((n, 1, r, th), device=device)
            x_mask = torch.ones((n, 1, r, th), device=device)
            for i in reversed(range(1, self.noise_steps)):
                t = torch.full((n,), i, device=device, dtype=torch.long)
                model_in = torch.cat([x_img, x_mask], dim=1)
                pred = model(model_in, t, cond)
                if cfg_scale and cfg_scale > 0:
                    uncond = model(model_in, t, None)
                    pred = torch.lerp(uncond, pred, cfg_scale)

                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x_img) if i > 1 else torch.zeros_like(x_img)
                x_img = (1.0 / torch.sqrt(alpha)) * (x_img - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * pred) + torch.sqrt(beta) * noise
        model.train()
        return x_img
