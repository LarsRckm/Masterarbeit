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
    zero_terminal_snr: bool = False

    def __post_init__(self):
        if self.schedule == "cosine":
            self.beta = _cosine_beta_schedule(self.noise_steps)
        elif self.schedule == "linear":
            self.beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule!r} (expected 'linear' or 'cosine')")
        self.alpha = 1.0 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        if self.zero_terminal_snr:
            self._rescale_zero_terminal_snr()

    def _rescale_zero_terminal_snr(self) -> None:
        """Rescale the schedule so alpha_hat[-1] = 0 (Lin et al., 2024).

        Keeps sqrt(alpha_hat)[0] unchanged, shifts/scales so sqrt(alpha_hat)[-1] = 0,
        then recomputes alpha and beta. After this the most-noised step is *pure*
        noise (no leaked signal), removing the train/test brightness mismatch.
        Requires v-prediction (eps-prediction is degenerate at alpha_hat = 0).
        """
        sqrt_ah = torch.sqrt(self.alpha_hat)
        s0 = sqrt_ah[0].clone()
        sT = sqrt_ah[-1].clone()
        sqrt_ah = (sqrt_ah - sT) * (s0 / (s0 - sT))      # s0 -> s0, sT -> 0
        alpha_hat = sqrt_ah ** 2
        alpha = torch.empty_like(alpha_hat)
        alpha[0] = alpha_hat[0]
        alpha[1:] = alpha_hat[1:] / alpha_hat[:-1]
        self.alpha_hat = alpha_hat
        self.alpha = alpha
        self.beta = 1.0 - alpha

    def to(self, device: torch.device) -> "Diffusion":
        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_hat = self.alpha_hat.to(device)
        return self

    def sample_timesteps(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.randint(low=1, high=self.noise_steps, size=(n,), device=device)

    @staticmethod
    def _offset(eps: torch.Tensor, offset_noise: float) -> torch.Tensor:
        """Add a per-image/channel constant DC offset to the noise (offset noise).

        eps -> eps + c * z, with one z ~ N(0,1) per (image, channel), broadcast over
        all pixels. Injects low-frequency/brightness content into the noise so the
        model learns to control the global brightness (counteracts DC drift).
        """
        if offset_noise <= 0.0:
            return eps
        z = torch.randn(eps.shape[0], eps.shape[1], 1, 1, device=eps.device, dtype=eps.dtype)
        return eps + float(offset_noise) * z

    def noise_images(self, x: torch.Tensor, t: torch.Tensor,
                     offset_noise: float = 0.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Add noise at timestep t.

        If x has channels [image, mask], only diffuse the image channel.
        ``offset_noise`` > 0 enables offset noise on the image channel.
        Returns (x_t, epsilon) where epsilon is the noise added to the image channel.
        """

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]

        if x.dim() == 4 and x.shape[1] >= 2:
            # Channel 0 = image (noised), channels 1+ = mask/radial map (kept clean)
            x_img  = x[:, :1]
            x_rest = x[:, 1:]
            eps = self._offset(torch.randn_like(x_img), offset_noise)
            x_noised = sqrt_alpha_hat * x_img + sqrt_one_minus_alpha_hat * eps
            return torch.cat([x_noised, x_rest], dim=1), eps

        eps = self._offset(torch.randn_like(x), offset_noise)
        x_noised = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        return x_noised, eps

    # --- v-prediction helpers (Salimans & Ho, 2022) -----------------------
    # v = sqrt(alpha_hat) * eps - sqrt(1-alpha_hat) * x0
    def get_v(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Velocity target v for v-prediction (x0, eps both [B,1,H,W])."""
        a = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        s = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        return a * eps - s * x0

    def v_to_eps(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert a predicted v back to the equivalent epsilon (for the reverse step)."""
        a = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        s = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        return s * x_t + a * v

    def v_to_x0(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert a predicted v back to the clean-image estimate x0 (division-free).

        x0 = sqrt(alpha_hat) * x_t - sqrt(1-alpha_hat) * v. Stable even when
        alpha_hat = 0 (zero terminal SNR), unlike (x_t - s*eps)/sqrt(alpha_hat).
        """
        a = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        s = torch.sqrt(1.0 - self.alpha_hat[t])[:, None, None, None]
        return a * x_t - s * v
