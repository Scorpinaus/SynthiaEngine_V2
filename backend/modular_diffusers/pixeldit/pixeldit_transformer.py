"""Diffusers-compatible PixelDiT T2I transformer component.

This module ports the inference-time PixelDiT T2I architecture into a
Diffusers ``ModelMixin`` so raw NVIDIA ``pixeldit_t2i_v1.pth`` checkpoints can
be stored in a normal component folder and loaded with ``from_pretrained``.

The implementation is adapted from the public NVlabs PixelDiT inference model
code and keeps the original module/key names, including the ``core.`` trainer
wrapper prefix used by the released checkpoint.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from torch.nn.functional import scaled_dot_product_attention


def _get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be even")
    emb_h = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = _get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    return _get_2d_sincos_pos_embed_from_grid(embed_dim, grid)


def _apply_adaln(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale) + shift


def _precompute_freqs_cis_2d(dim: int, height: int, width: int, theta: float = 10000.0, scale: float = 16.0):
    x_pos = torch.linspace(0, scale, width)
    y_pos = torch.linspace(0, scale, height)
    y_pos, x_pos = torch.meshgrid(y_pos, x_pos, indexing="ij")
    y_pos = y_pos.reshape(-1)
    x_pos = x_pos.reshape(-1)
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    x_freqs = torch.outer(x_pos, freqs).float()
    y_freqs = torch.outer(y_pos, freqs).float()
    x_cis = torch.polar(torch.ones_like(x_freqs), x_freqs)
    y_cis = torch.polar(torch.ones_like(y_freqs), y_freqs)
    freqs_cis = torch.cat([x_cis.unsqueeze(dim=-1), y_cis.unsqueeze(dim=-1)], dim=-1)
    return freqs_cis.reshape(height * width, -1)


def _apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    freqs_cis = freqs_cis[None, :, None, :]
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    xq_out = torch.view_as_real(xq_complex * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_complex * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class TimestepConditioner(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t[..., None].float() * freqs[None, ...]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        mlp_dtype = next(self.mlp.parameters()).dtype
        return self.mlp(t_freq.to(mlp_dtype))


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int):
        super().__init__()
        self.norm = RMSNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(x))


class PatchTokenEmbedder(nn.Module):
    def __init__(self, in_chans: int = 3, embed_dim: int = 768, norm_layer=None, bias: bool = True):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(x))


class RotaryAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, mask=None) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        qkv = self.qkv(x).reshape(batch_size, sequence_length, 3, self.num_heads, channels // self.num_heads)
        q, k, v = qkv.permute(2, 0, 1, 3, 4)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = _apply_rotary_emb(q, k, freqs_cis=pos)
        q = q.view(batch_size, -1, self.num_heads, channels // self.num_heads).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, channels // self.num_heads).transpose(1, 2).contiguous()
        v = v.view(batch_size, -1, self.num_heads, channels // self.num_heads).transpose(1, 2).contiguous()
        x = scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0)
        x = x.transpose(1, 2).reshape(batch_size, sequence_length, channels)
        return self.proj(x)


class PixelTokenEmbedder(nn.Module):
    def __init__(self, in_channels: int, hidden_size_output: int, use_pixel_abs_pos: bool = True):
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size_output = int(hidden_size_output)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        self.proj = nn.Linear(self.in_channels, self.hidden_size_output, bias=True)
        self._pos_cache: dict[tuple[str, int, int], torch.Tensor] = {}

    def _fetch_pixel_pos_image(self, height: int, width: int, device, dtype):
        key = ("image", height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device=device, dtype=dtype)
        if height == width:
            pos_np = _get_2d_sincos_pos_embed(self.hidden_size_output, height)
        else:
            grid_h = np.arange(height, dtype=np.float32)
            grid_w = np.arange(width, dtype=np.float32)
            grid = np.meshgrid(grid_w, grid_h)
            grid = np.stack(grid, axis=0).reshape(2, 1, height, width)
            pos_np = _get_2d_sincos_pos_embed_from_grid(self.hidden_size_output, grid)
        pos = torch.from_numpy(pos_np)
        self._pos_cache[key] = pos
        return pos.to(device=device, dtype=dtype)

    def forward(self, inputs: torch.Tensor, img_height: int, img_width: int, patch_size: int) -> torch.Tensor:
        if inputs.dim() != 4:
            raise ValueError("PixelTokenEmbedder expects inputs of shape [B,C,H,W]")
        batch_size, _, height, width = inputs.shape
        if height != img_height or width != img_width:
            raise ValueError("Image dimensions do not match PixelTokenEmbedder arguments")
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size")
        height_patches, width_patches = height // patch_size, width // patch_size
        patch_area = patch_size * patch_size
        x = inputs.permute(0, 2, 3, 1).contiguous()
        x = self.proj(x)
        if self.use_pixel_abs_pos:
            pos_full = self._fetch_pixel_pos_image(height, width, inputs.device, inputs.dtype)
            x = x + pos_full.view(height, width, self.hidden_size_output).unsqueeze(0)
        x = x.view(batch_size, height_patches, patch_size, width_patches, patch_size, self.hidden_size_output)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(batch_size * height_patches * width_patches, patch_area, self.hidden_size_output)


class PiTBlock(nn.Module):
    def __init__(
        self,
        pixel_hidden_size: int,
        patch_hidden_size: int,
        patch_size: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_hidden_size: int | None = None,
        attn_num_heads: int | None = None,
    ):
        super().__init__()
        self.pixel_dim = int(pixel_hidden_size)
        self.context_dim = int(patch_hidden_size)
        self.patch_size = int(patch_size)
        self.attn_dim = int(attn_hidden_size) if attn_hidden_size is not None else self.context_dim
        self.num_heads = int(attn_num_heads) if attn_num_heads is not None else int(num_heads)
        if self.attn_dim % self.num_heads != 0:
            raise ValueError("pixel attention hidden size must be divisible by pixel num_heads")
        patch_area = self.patch_size * self.patch_size
        self.compress_to_attn = nn.Linear(patch_area * self.pixel_dim, self.attn_dim, bias=True)
        self.expand_from_attn = nn.Linear(self.attn_dim, patch_area * self.pixel_dim, bias=True)
        self.norm1 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.attn = RotaryAttention(self.attn_dim, num_heads=self.num_heads, qkv_bias=False)
        self.norm2 = RMSNorm(self.pixel_dim, eps=1e-6)
        self.mlp = MLP(self.pixel_dim, mlp_ratio=mlp_ratio, drop=0.0)
        self.adaLN_modulation = nn.Sequential(nn.Linear(self.context_dim, 6 * self.pixel_dim * patch_area, bias=True))
        self._pos_cache: dict[tuple[int, int], torch.Tensor] = {}

    def _fetch_pos(self, height: int, width: int, device):
        key = (height, width)
        if key in self._pos_cache:
            return self._pos_cache[key].to(device)
        pos = _precompute_freqs_cis_2d(self.attn_dim // self.num_heads, height, width).to(device)
        self._pos_cache[key] = pos
        return pos

    def forward(
        self,
        x: torch.Tensor,
        s_cond: torch.Tensor,
        image_height: int,
        image_width: int,
        patch_size: int,
        mask=None,
    ) -> torch.Tensor:
        batch_length, patch_area, channels = x.shape
        if channels != self.pixel_dim:
            raise ValueError(f"PiTBlock expected pixel_dim={self.pixel_dim}, got {channels}")
        if image_height % patch_size != 0 or image_width % patch_size != 0:
            raise ValueError("Image dimensions must be divisible by patch_size")
        height_patches, width_patches = image_height // patch_size, image_width // patch_size
        length = height_patches * width_patches
        batch_size = batch_length // length
        cond_params = self.adaLN_modulation(s_cond).view(batch_length, patch_area, 6 * self.pixel_dim)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(cond_params, 6, dim=-1)
        x_norm = _apply_adaln(self.norm1(x), shift_msa, scale_msa)
        x_flat = x_norm.view(batch_length, patch_area * self.pixel_dim)
        x_comp = self.compress_to_attn(x_flat).view(batch_size, length, self.attn_dim)
        attn_out = self.attn(x_comp, self._fetch_pos(height_patches, width_patches, x.device), mask)
        attn_exp = self.expand_from_attn(attn_out.view(batch_size * length, self.attn_dim))
        attn_exp = attn_exp.view(batch_length, patch_area, self.pixel_dim)
        x = x + gate_msa * attn_exp
        return x + gate_mlp * self.mlp(_apply_adaln(self.norm2(x), shift_mlp, scale_mlp))


class MMDiTJointAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv_x = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_y = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm_x = RMSNorm(self.head_dim)
        self.k_norm_x = RMSNorm(self.head_dim)
        self.q_norm_y = RMSNorm(self.head_dim)
        self.k_norm_y = RMSNorm(self.head_dim)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        pos_img: torch.Tensor,
        pos_txt: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, image_tokens, channels = x.shape
        text_batch, text_tokens, text_channels = y.shape
        if batch_size != text_batch or channels != text_channels:
            raise ValueError("x and y must share batch and channel dimensions")

        qkv_x = self.qkv_x(x).reshape(batch_size, image_tokens, 3, self.num_heads, self.head_dim)
        qx, kx, vx = qkv_x.permute(2, 0, 1, 3, 4)
        qx = self.q_norm_x(qx)
        kx = self.k_norm_x(kx)

        qkv_y = self.qkv_y(y).reshape(batch_size, text_tokens, 3, self.num_heads, self.head_dim)
        qy, ky, vy = qkv_y.permute(2, 0, 1, 3, 4)
        qy = self.q_norm_y(qy)
        ky = self.k_norm_y(ky)

        qx, kx = _apply_rotary_emb(qx, kx, freqs_cis=pos_img)
        if pos_txt is not None:
            qy, ky = _apply_rotary_emb(qy, ky, freqs_cis=pos_txt)

        q_joint = torch.cat([qy.transpose(1, 2), qx.transpose(1, 2)], dim=2)
        k_joint = torch.cat([ky.transpose(1, 2), kx.transpose(1, 2)], dim=2)
        v_joint = torch.cat([vy.transpose(1, 2), vx.transpose(1, 2)], dim=2)
        out_joint = F.scaled_dot_product_attention(q_joint, k_joint, v_joint, dropout_p=0.0, attn_mask=attn_mask)
        out_y = out_joint[:, :, :text_tokens, :].transpose(1, 2).reshape(batch_size, text_tokens, channels)
        out_x = out_joint[:, :, text_tokens:, :].transpose(1, 2).reshape(batch_size, image_tokens, channels)
        return self.proj_x(out_x), self.proj_y(out_y)


class MMDiTBlockT2I(nn.Module):
    def __init__(self, hidden_size: int, groups: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm_x1 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = MMDiTJointAttention(hidden_size, num_heads=groups, qkv_bias=False)
        self.norm_x2 = RMSNorm(hidden_size, eps=1e-6)
        self.norm_y2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp_x = FeedForward(hidden_size, mlp_hidden_dim)
        self.mlp_y = FeedForward(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation_img = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.adaLN_modulation_txt = nn.Sequential(nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, y, c, pos_img, pos_txt=None, attn_mask=None):
        shift_msa_x, scale_msa_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaLN_modulation_img(c).chunk(
            6, dim=-1
        )
        shift_msa_y, scale_msa_y, gate_msa_y, shift_mlp_y, scale_mlp_y, gate_mlp_y = self.adaLN_modulation_txt(c).chunk(
            6, dim=-1
        )
        x_norm = _apply_adaln(self.norm_x1(x), shift_msa_x, scale_msa_x)
        y_norm = _apply_adaln(self.norm_y1(y), shift_msa_y, scale_msa_y)
        attn_x, attn_y = self.attn(x_norm, y_norm, pos_img, pos_txt, attn_mask)
        x = x + gate_msa_x * attn_x
        y = y + gate_msa_y * attn_y
        x = x + gate_mlp_x * self.mlp_x(_apply_adaln(self.norm_x2(x), shift_mlp_x, scale_mlp_x))
        y = y + gate_mlp_y * self.mlp_y(_apply_adaln(self.norm_y2(y), shift_mlp_y, scale_mlp_y))
        return x, y


class PixDiTCoreT2I(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_groups: int = 24,
        hidden_size: int = 1536,
        pixel_hidden_size: int = 16,
        pixel_attn_hidden_size: int | None = 1152,
        pixel_num_groups: int | None = 16,
        patch_depth: int = 14,
        pixel_depth: int = 2,
        num_text_blocks: int = 4,
        patch_size: int = 16,
        txt_embed_dim: int = 2304,
        txt_max_length: int = 300,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        repa_encoder_index: int = 6,
        use_pixel_abs_pos: bool = True,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.num_groups = int(num_groups)
        self.patch_depth = int(patch_depth)
        self.pixel_depth = int(pixel_depth)
        self.num_text_blocks = int(num_text_blocks)
        self.patch_size = int(patch_size)
        self.pixel_hidden_size = int(pixel_hidden_size)
        self.txt_embed_dim = int(txt_embed_dim)
        self.txt_max_length = int(txt_max_length)
        self.use_text_rope = bool(use_text_rope)
        self.text_rope_theta = float(text_rope_theta)
        self.repa_encoder_index = int(repa_encoder_index)
        self.use_pixel_abs_pos = bool(use_pixel_abs_pos)
        if self.pixel_depth <= 0:
            raise ValueError("PixelDiT expects pixel_depth > 0")

        self.pixel_embedder = PixelTokenEmbedder(
            self.in_channels, self.pixel_hidden_size, use_pixel_abs_pos=self.use_pixel_abs_pos
        )
        self.s_embedder = PatchTokenEmbedder(self.in_channels * self.patch_size**2, self.hidden_size, bias=True)
        self.t_embedder = TimestepConditioner(self.hidden_size)
        self.y_embedder = PatchTokenEmbedder(self.txt_embed_dim, self.hidden_size, bias=True, norm_layer=RMSNorm)
        self.y_pos_embedding = nn.Parameter(torch.randn(1, self.txt_max_length, self.hidden_size))
        self.patch_blocks = nn.ModuleList(
            [MMDiTBlockT2I(self.hidden_size, self.num_groups) for _ in range(self.patch_depth)]
        )
        self.text_refine_blocks = None
        self.pixel_attn_hidden_size = (
            int(pixel_attn_hidden_size) if pixel_attn_hidden_size is not None else self.hidden_size
        )
        self.pixel_num_groups = int(pixel_num_groups) if pixel_num_groups is not None else self.num_groups
        self.pixel_blocks = nn.ModuleList(
            [
                PiTBlock(
                    self.pixel_hidden_size,
                    self.hidden_size,
                    patch_size=self.patch_size,
                    num_heads=self.num_groups,
                    mlp_ratio=4.0,
                    attn_hidden_size=self.pixel_attn_hidden_size,
                    attn_num_heads=self.pixel_num_groups,
                )
                for _ in range(self.pixel_depth)
            ]
        )
        self.final_layer = FinalLayer(self.pixel_hidden_size, self.out_channels)
        self.precompute_pos: dict[tuple[int, int], torch.Tensor] = {}
        self.precompute_pos_txt: dict[int, torch.Tensor] = {}
        self.last_repa_tokens = None
        self.initialize_weights()

    def fetch_pos(self, height: int, width: int, device):
        if (height, width) in self.precompute_pos:
            return self.precompute_pos[(height, width)].to(device)
        pos = _precompute_freqs_cis_2d(self.hidden_size // self.num_groups, height, width).to(device)
        self.precompute_pos[(height, width)] = pos
        return pos

    def fetch_pos_text(self, length: int, device):
        if length in self.precompute_pos_txt:
            return self.precompute_pos_txt[length].to(device)
        head_dim = self.hidden_size // self.num_groups
        freqs = 1.0 / (self.text_rope_theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
        positions = torch.arange(0, length, device=device).float().unsqueeze(1)
        angles = positions * freqs.unsqueeze(0)
        freqs_cis = torch.polar(torch.ones_like(angles), angles)
        self.precompute_pos_txt[length] = freqs_cis
        return freqs_cis

    def initialize_weights(self):
        w = self.s_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.s_embedder.proj.bias, 0)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, _, height, width = x.shape
        height_patches = height // self.patch_size
        width_patches = width // self.patch_size
        length = height_patches * width_patches
        pos = self.fetch_pos(height_patches, width_patches, x.device)
        x_patches = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size).transpose(1, 2)
        t_emb = self.t_embedder(t.view(-1)).view(batch_size, -1, self.hidden_size)

        if y.dim() != 3:
            raise ValueError("Text embedding y must be [B, L, D]")
        text_length = min(y.shape[1], self.txt_max_length)
        y = y[:, :text_length, :]
        y_emb = self.y_embedder(y).view(batch_size, text_length, self.hidden_size)
        y_emb = y_emb + self.y_pos_embedding[:, :text_length, :].to(y_emb.dtype)
        condition = F.silu(t_emb)

        if s is None:
            s = self.s_embedder(x_patches)
            pos_txt = self.fetch_pos_text(text_length, x.device) if self.use_text_rope else None
            attn_mask_joint = None
            if mask is not None and isinstance(mask, torch.Tensor):
                m = mask
                while m.dim() > 2 and m.size(1) == 1:
                    m = m.squeeze(1)
                if m.dim() == 2:
                    pad = m == 0
                    pad_img = torch.zeros((batch_size, length), dtype=torch.bool, device=x.device)
                    attn_mask_joint = torch.cat([pad[:, :text_length], pad_img], dim=1).view(
                        batch_size, 1, 1, text_length + length
                    )
            self.last_repa_tokens = None
            for i, block in enumerate(self.patch_blocks):
                s, y_emb = block(s, y_emb, condition, pos, pos_txt, attn_mask_joint)
                if 0 < self.repa_encoder_index == (i + 1):
                    self.last_repa_tokens = s
            s = F.silu(t_emb + s)
        if not (0 < self.repa_encoder_index <= self.patch_depth):
            self.last_repa_tokens = s

        if s.shape[1] != length:
            if s.shape[1] > length:
                s = s[:, :length, :]
            else:
                pad_len = length - s.shape[1]
                s = torch.cat([s, s.new_zeros(batch_size, pad_len, s.shape[2])], dim=1)

        s_cond = s.view(batch_size * length, self.hidden_size)
        x_pixels = self.pixel_embedder(x, img_height=height, img_width=width, patch_size=self.patch_size)
        for block in self.pixel_blocks:
            x_pixels = block(x_pixels, s_cond, height, width, self.patch_size, mask)
        x_pixels = self.final_layer(x_pixels)
        channels_out = self.out_channels
        patch_area = self.patch_size * self.patch_size
        x_pixels = x_pixels.view(batch_size, length, patch_area, channels_out).permute(0, 3, 2, 1).contiguous()
        x_pixels = x_pixels.view(batch_size, channels_out * patch_area, length)
        return F.fold(x_pixels, (height, width), kernel_size=self.patch_size, stride=self.patch_size)


class PixelDiTTransformer2DModel(ModelMixin, ConfigMixin):
    """PixelDiT T2I transformer wrapper with Diffusers save/load support."""

    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        patch_size: int = 16,
        num_groups: int = 24,
        hidden_size: int = 1536,
        pixel_hidden_size: int = 16,
        pixel_attn_hidden_size: int = 1152,
        pixel_num_groups: int = 16,
        patch_depth: int = 14,
        pixel_depth: int = 2,
        num_text_blocks: int = 4,
        txt_embed_dim: int = 2304,
        txt_max_length: int = 300,
        use_text_rope: bool = True,
        text_rope_theta: float = 10000.0,
        repa_encoder_index: int = 6,
        use_pixel_abs_pos: bool = True,
        image_size: int = 1024,
        text_encoder: str = "gemma-2-2b-it",
        flow_shift: float = 4.0,
        default_steps: int = 50,
        default_cfg_scale: float = 2.75,
        default_negative_prompt: str = "low quality, worst quality, over-saturated, blurry, deformed, watermark",
        **kwargs: Any,
    ):
        super().__init__()
        self.core = PixDiTCoreT2I(
            in_channels=in_channels,
            num_groups=num_groups,
            hidden_size=hidden_size,
            pixel_hidden_size=pixel_hidden_size,
            pixel_attn_hidden_size=pixel_attn_hidden_size,
            pixel_num_groups=pixel_num_groups,
            patch_depth=patch_depth,
            pixel_depth=pixel_depth,
            num_text_blocks=num_text_blocks,
            patch_size=patch_size,
            txt_embed_dim=txt_embed_dim,
            txt_max_length=txt_max_length,
            use_text_rope=use_text_rope,
            text_rope_theta=text_rope_theta,
            repa_encoder_index=repa_encoder_index,
            use_pixel_abs_pos=use_pixel_abs_pos,
        )
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.pred_sigma = False
        self._txt_embed_dim = int(txt_embed_dim)
        projector_dim = 2048
        self._repa_projector = nn.Sequential(
            nn.Linear(self.core.hidden_size, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, projector_dim),
            nn.SiLU(),
            nn.Linear(projector_dim, 768),
        )

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
        **kwargs: Any,
    ):
        hidden_states = hidden_states.to(self.dtype)
        timestep = timestep.to(self.dtype)
        if encoder_hidden_states.dim() == 4:
            encoder_hidden_states = encoder_hidden_states.squeeze(1)
        elif encoder_hidden_states.dim() != 3:
            raise ValueError("PixelDiT expects encoder_hidden_states with shape [B,L,D] or [B,1,L,D]")
        encoder_hidden_states = encoder_hidden_states.to(self.dtype)
        if encoder_hidden_states.shape[-1] != self._txt_embed_dim:
            raise RuntimeError(
                f"PixelDiT text embedding dim {encoder_hidden_states.shape[-1]} != expected {self._txt_embed_dim}"
            )
        if hasattr(self.core, "last_repa_tokens"):
            self.core.last_repa_tokens = None
        sample = self.core(hidden_states, timestep, encoder_hidden_states, s=None, mask=attention_mask)
        if not return_dict:
            return (sample,)
        return {"sample": sample}

    def forward_with_dpmsolver(self, x, timestep, y, mask=None, **kwargs):
        out = self.forward(x, timestep, y, attention_mask=mask, return_dict=True, **kwargs)
        return out["sample"]
