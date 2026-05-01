"""Polar CT UNet for conditional DDPM.

This module intentionally focuses on the *model* only.

Key differences vs. the original DDPM_Grayscale_512x512 modules:
- Input supports non-square tensors: [B, C, R, Theta]
- Theta is treated as periodic via mixed (r,theta) padding.
- Conditioning supports categorical + continuous inputs and is shape-agnostic.
- Optional self-attention is gated by a token budget.

Expected input:
  x:    float tensor [B, 2, POLAR_R_MODEL, POLAR_THETA_BINS] (image + mask)
  t:    int/long tensor [B] (diffusion timestep)
  cond: either None (unconditional) or a tuple (cat, cont)
        cat : long tensor  [B, 3] with (cell_format_id, manufacturer_id, chemistry_id)
        cont: float tensor [B, 3] with (slice_depth_relative, voxel_size_um, r_valid_rel)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from model import config as project_config
except ImportError:
    import config as project_config


# -----------------------------------------------------------------------------
# Building blocks
# -----------------------------------------------------------------------------


def _mixed_pad_2d(x: torch.Tensor, pad: int, pad_r_mode: str, pad_theta_mode: str) -> torch.Tensor:
    """Pad tensor with different modes for r (height) and theta (width).

    Padding order for F.pad is (left, right, top, bottom).
    """
    if pad == 0:
        return x

    # Pad theta (width) circular to avoid seam artefacts in polar representation.
    x = F.pad(x, (pad, pad, 0, 0), mode=pad_theta_mode)

    # Pad r (height) with a non-periodic mode.
    x = F.pad(x, (0, 0, pad, pad), mode=pad_r_mode)
    return x


class DoubleConvPolar(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: Optional[int] = None,
        residual: bool = False,
        pad_r_mode: str = "reflect",
        pad_theta_mode: str = "circular",
    ):
        super().__init__()
        self.residual = residual
        self.pad_r_mode = pad_r_mode
        self.pad_theta_mode = pad_theta_mode

        if mid_channels is None:
            mid_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=0, bias=False)
        self.gn1 = nn.GroupNorm(1, mid_channels)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=0, bias=False)
        self.gn2 = nn.GroupNorm(1, out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = _mixed_pad_2d(x, pad=1, pad_r_mode=self.pad_r_mode, pad_theta_mode=self.pad_theta_mode)
        y = self.conv1(y)
        y = self.gn1(y)
        y = self.act(y)

        y = _mixed_pad_2d(y, pad=1, pad_r_mode=self.pad_r_mode, pad_theta_mode=self.pad_theta_mode)
        y = self.conv2(y)
        y = self.gn2(y)

        if self.residual:
            return F.gelu(x + y)
        return y


class SelfAttentionDynamic(nn.Module):
    """Multi-head self-attention over a 2D feature map.

    To keep compute bounded, attention is skipped if H*W exceeds max_tokens.
    """

    def __init__(self, channels: int, num_heads: int = 4, max_tokens: int = 4096):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.max_tokens = max_tokens

        self.mha = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        n = h * w
        if n > self.max_tokens:
            return x

        x_seq = x.view(b, c, n).swapaxes(1, 2)  # [B, N, C]
        x_ln = self.ln(x_seq)
        attn, _ = self.mha(x_ln, x_ln, x_ln)
        attn = attn + x_seq
        attn = self.ff_self(attn) + attn
        return attn.swapaxes(2, 1).view(b, c, h, w)


class ConditionEncoder(nn.Module):
    def __init__(
        self,
        time_dim: int,
        cat_emb_dim: int,
        cont_dim: int,
        cell_format_vocab: int,
        manufacturer_vocab: int,
        chemistry_vocab: int,
    ):
        super().__init__()

        self.cell_format_emb = nn.Embedding(cell_format_vocab, cat_emb_dim)
        self.manufacturer_emb = nn.Embedding(manufacturer_vocab, cat_emb_dim)
        self.chemistry_emb = nn.Embedding(chemistry_vocab, cat_emb_dim)

        self.cont_mlp = nn.Sequential(
            nn.Linear(cont_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 256),
            nn.SiLU(),
        )

        fused_in = 3 * cat_emb_dim + 256
        self.fuse = nn.Sequential(
            nn.Linear(fused_in, 256),
            nn.SiLU(),
            nn.Linear(256, time_dim),
        )

        # A learned unconditional token can be used when cond=None.
        self.uncond = nn.Parameter(torch.zeros(time_dim))

    def forward(self, cond: Optional[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if cond is None:
            return self.uncond[None, :]

        cat, cont = cond
        # cat: [B,3]
        cell_id = cat[:, 0]
        man_id = cat[:, 1]
        chem_id = cat[:, 2]

        e1 = self.cell_format_emb(cell_id)
        e2 = self.manufacturer_emb(man_id)
        e3 = self.chemistry_emb(chem_id)
        e_cat = torch.cat([e1, e2, e3], dim=-1)

        e_cont = self.cont_mlp(cont)
        fused = torch.cat([e_cat, e_cont], dim=-1)
        return self.fuse(fused)


class DownPolar(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        cond_channels: int = 1,
        pad_r_mode: str = "reflect",
        pad_theta_mode: str = "circular",
    ):
        super().__init__()
        self.cond_channels = cond_channels

        self.maxpool = nn.MaxPool2d(2)
        self.conv1 = DoubleConvPolar(in_channels, in_channels, residual=True, pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.conv2 = DoubleConvPolar(in_channels, out_channels - cond_channels, pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))
        self.cond_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, cond_channels))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(x)
        x = self.conv1(x)
        x = self.conv2(x)

        b, _, h, w = x.shape
        t_map = self.time_proj(t_emb)[:, :, None, None].expand(b, -1, h, w)
        c_map = self.cond_proj(c_emb)[:, :, None, None].expand(b, -1, h, w)
        x = torch.cat([x, c_map], dim=1)
        return x + t_map


class UpPolar(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_dim: int,
        cond_channels: int = 1,
        pad_r_mode: str = "reflect",
        pad_theta_mode: str = "circular",
    ):
        super().__init__()
        self.cond_channels = cond_channels

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv1 = DoubleConvPolar(in_channels, in_channels, residual=True, pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.conv2 = DoubleConvPolar(in_channels, out_channels - cond_channels, mid_channels=in_channels // 2,
                                     pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)

        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(emb_dim, out_channels))
        self.cond_proj = nn.Sequential(
            nn.Linear(emb_dim, 64),
            nn.SiLU(),
            nn.Linear(64, cond_channels),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        b, _, h, w = x.shape
        t_map = self.time_proj(t_emb)[:, :, None, None].expand(b, -1, h, w)
        c_map = self.cond_proj(c_emb)[:, :, None, None].expand(b, -1, h, w)
        x = torch.cat([x, c_map], dim=1)
        return x + t_map


# -----------------------------------------------------------------------------
# UNet (conditional)
# -----------------------------------------------------------------------------


class UNet_conditional_polar(nn.Module):
    def __init__(
        self,
        c_in: int = project_config.UNET_IN_CHANNELS,
        c_out: int = project_config.UNET_OUT_CHANNELS,
        base_channels: int = project_config.UNET_BASE_CHANNELS,
        time_dim: int = project_config.TIME_EMB_DIM,
        num_downs: int = project_config.UNET_NUM_DOWNS,
        cond_cont_dim: int = project_config.COND_CONT_DIM,
    ):
        super().__init__()
        if num_downs != 5:
            raise ValueError("UNet_conditional_polar currently supports num_downs=5 (as requested).")

        pad_r_mode = getattr(project_config, "PAD_R_MODE", "reflect")
        pad_theta_mode = getattr(project_config, "PAD_THETA_MODE", "circular")

        self.time_dim = time_dim
        self.cond_channels = 1  # keep compatible with out_channels - 1 design

        self.inc = DoubleConvPolar(c_in, base_channels, pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)

        ch1 = base_channels * 2
        ch2 = base_channels * 4
        ch3 = base_channels * 8
        ch4 = base_channels * 16
        ch5 = base_channels * 32

        self.down1 = DownPolar(base_channels, ch1, emb_dim=time_dim, cond_channels=self.cond_channels,
                               pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.down2 = DownPolar(ch1, ch2, emb_dim=time_dim, cond_channels=self.cond_channels,
                               pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.down3 = DownPolar(ch2, ch3, emb_dim=time_dim, cond_channels=self.cond_channels,
                               pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.down4 = DownPolar(ch3, ch4, emb_dim=time_dim, cond_channels=self.cond_channels,
                               pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.down5 = DownPolar(ch4, ch5, emb_dim=time_dim, cond_channels=self.cond_channels,
                               pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)

        # Attention (gated)
        self.attn_enabled = project_config.ATTN_ENABLED
        max_tokens = project_config.ATTN_MAX_TOKENS
        heads = project_config.ATTN_HEADS
        self.sa4 = SelfAttentionDynamic(ch4, num_heads=heads, max_tokens=max_tokens)
        self.sa5 = SelfAttentionDynamic(ch5, num_heads=heads, max_tokens=max_tokens)

        # Bottleneck
        self.bot1 = DoubleConvPolar(ch5, ch5, pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.bot2 = DoubleConvPolar(ch5, ch5, pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.bot3 = DoubleConvPolar(ch5, ch5, pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)

        # Up path. in_channels account for skip concatenation.
        self.up5 = UpPolar(ch5 + ch4, ch4, emb_dim=time_dim, cond_channels=self.cond_channels,
                           pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.up4 = UpPolar(ch4 + ch3, ch3, emb_dim=time_dim, cond_channels=self.cond_channels,
                           pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.up3 = UpPolar(ch3 + ch2, ch2, emb_dim=time_dim, cond_channels=self.cond_channels,
                           pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.up2 = UpPolar(ch2 + ch1, ch1, emb_dim=time_dim, cond_channels=self.cond_channels,
                           pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)
        self.up1 = UpPolar(ch1 + base_channels, base_channels, emb_dim=time_dim, cond_channels=self.cond_channels,
                           pad_r_mode=pad_r_mode, pad_theta_mode=pad_theta_mode)

        self.outc = nn.Conv2d(base_channels, c_out, kernel_size=1)

        self.condition_encoder = ConditionEncoder(
            time_dim=time_dim,
            cat_emb_dim=project_config.COND_CAT_EMB_DIM,
            cont_dim=cond_cont_dim,
            cell_format_vocab=project_config.CELL_FORMAT_VOCAB_SIZE,
            manufacturer_vocab=project_config.MANUFACTURER_VOCAB_SIZE,
            chemistry_vocab=project_config.CHEMISTRY_VOCAB_SIZE,
        )

    @staticmethod
    def pos_encoding(t: torch.Tensor, channels: int) -> torch.Tensor:
        device = t.device
        inv_freq = 1.0 / (
            10000 ** (torch.arange(0, channels, 2, device=device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        # time embedding
        t = t.unsqueeze(-1).type(torch.float)
        t_emb = self.pos_encoding(t, self.time_dim)

        # condition embedding
        c_emb = self.condition_encoder(cond)
        if c_emb.shape[0] == 1 and x.shape[0] != 1:
            # broadcast unconditional token
            c_emb = c_emb.expand(x.shape[0], -1)

        x0 = self.inc(x)
        x1 = self.down1(x0, t_emb, c_emb)
        x2 = self.down2(x1, t_emb, c_emb)
        x3 = self.down3(x2, t_emb, c_emb)
        x4 = self.down4(x3, t_emb, c_emb)
        if self.attn_enabled:
            x4 = self.sa4(x4)
        x5 = self.down5(x4, t_emb, c_emb)
        if self.attn_enabled:
            x5 = self.sa5(x5)

        x5 = self.bot1(x5)
        x5 = self.bot2(x5)
        x5 = self.bot3(x5)

        x = self.up5(x5, x4, t_emb, c_emb)
        x = self.up4(x, x3, t_emb, c_emb)
        x = self.up3(x, x2, t_emb, c_emb)
        x = self.up2(x, x1, t_emb, c_emb)
        x = self.up1(x, x0, t_emb, c_emb)
        return self.outc(x)
