"""
Temporal ViT: DA3-Inspired Student Model for Feature Distillation.

Architecture inspired by Depth Anything V3:
- Uses a plain transformer with partial alternation
- First 2/3 of layers: Spatial attention only (per-frame)
- Last 1/3 of layers: Alternating Spatial ↔ Temporal-Local

Temporal-Local attention enables explicit correspondence learning:
- Each patch attends to the same spatial position across time
- Efficient: O(S × T²) vs O((TS)²) for global temporal

Distillation target: Teacher's decoder_hidden tokens (before task heads)
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial
from typing import Optional, List

from .dinov2.layers import Mlp
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .dinov2.hub.backbones import dinov2_vitl14_reg


class TemporalViT(nn.Module):
    """
    DA3-inspired temporal ViT for single-camera sequences.

    Architecture:
    - Encoder: DINOv2 (frozen or trainable)
    - Decoder: 24 layers with DA3-style attention pattern
      - First 16 layers: Spatial only
      - Last 8 layers: Alternating Spatial ↔ Temporal-Local

    Output matches teacher's decoder_hidden: (B*T, S, 2*D)
    """

    def __init__(
        self,
        dec_depth: int = 24,
        dec_embed_dim: int = 1024,
        dec_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        spatial_ratio: float = 0.67,  # 2/3 spatial-only layers (DA3 default)
        temporal_mode: str = 'local',  # 'local' or 'global'
        pos_type: str = 'rope100',
        freeze_encoder: bool = True,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.temporal_mode = temporal_mode
        self.gradient_checkpointing = gradient_checkpointing

        # Compute layer split point
        self.spatial_only_layers = int(dec_depth * spatial_ratio)

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=True)
        self.patch_size = 14
        del self.encoder.mask_token

        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024

        # Projection if needed
        if enc_embed_dim != dec_embed_dim:
            self.enc_to_dec_proj = nn.Linear(enc_embed_dim, dec_embed_dim)
        else:
            self.enc_to_dec_proj = nn.Identity()

        # ----------------------
        #  Positional Encoding
        # ----------------------
        self.pos_type = pos_type
        if pos_type.startswith('rope'):
            freq = float(pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError(f"Position type {pos_type} not implemented")

        # ----------------------
        #   DA3-Style Decoder
        # ----------------------
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)
        ])

        # ----------------------
        #   Register Tokens
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #   Output Projection
        # ----------------------
        # Match teacher output dim: 2*D (concat of last two layers)
        self.output_dim = 2 * dec_embed_dim

        # Image normalization
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for temporal feature extraction.

        Args:
            images: (B, T, 3, H, W) single-camera sequence

        Returns:
            tokens: (B*T, S, 2*D) decoder output tokens (matches teacher format)
        """
        B, T, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        S = patch_h * patch_w

        # Normalize images
        images = (images - self.image_mean) / self.image_std

        # 1. Encode each frame with DINOv2
        imgs_flat = images.reshape(B * T, 3, H, W)

        if hasattr(self.encoder, 'forward_features'):
            hidden = self.encoder.forward_features(imgs_flat)
        else:
            hidden = self.encoder(imgs_flat, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # hidden: (B*T, S, enc_D)
        hidden = self.enc_to_dec_proj(hidden)  # (B*T, S, D)
        D = hidden.shape[2]

        # 2. Add register tokens
        register_token = self.register_token.expand(B, T, -1, -1)  # (B, T, reg, D)
        hidden = hidden.reshape(B, T, S, D)
        hidden = torch.cat([register_token, hidden], dim=2)  # (B, T, S+reg, D)
        S_full = hidden.shape[2]

        # 3. Get base positions for spatial attention
        base_pos = self.position_getter(1, patch_h, patch_w, hidden.device)  # (1, S, 2)

        # Add offset for register tokens
        if self.patch_start_idx > 0:
            base_pos = base_pos + 1
            pos_special = torch.zeros(1, self.patch_start_idx, 2, device=hidden.device, dtype=base_pos.dtype)
            base_pos = torch.cat([pos_special, base_pos], dim=1)  # (1, S_full, 2)

        # Spatial positions: (B*T, S_full, 2)
        spatial_pos = base_pos.expand(B * T, -1, -1)

        # Temporal positions for local temporal attention: (B*S_full, T, 2)
        # Each position gets a time index
        time_pos = torch.arange(T, device=hidden.device).float().view(1, T, 1)
        time_pos = time_pos.expand(B * S_full, -1, -1)
        # Add a dummy second dimension for RoPE compatibility
        time_pos = torch.cat([time_pos, torch.zeros_like(time_pos)], dim=-1)

        # 4. DA3-style decoding
        hidden = hidden.reshape(B * T, S_full, D)
        all_outputs = []

        for i, blk in enumerate(self.decoder):
            if i < self.spatial_only_layers:
                # Spatial only: (B*T, S_full, D)
                if self.gradient_checkpointing and self.training:
                    hidden = checkpoint(lambda x, p: blk(x, xpos=p), hidden, spatial_pos, use_reentrant=False)
                else:
                    hidden = blk(hidden, xpos=spatial_pos)
            else:
                # Alternating: spatial on even (relative to split), temporal on odd
                layer_in_alternating = i - self.spatial_only_layers

                if layer_in_alternating % 2 == 0:
                    # Spatial: (B*T, S_full, D)
                    if self.gradient_checkpointing and self.training:
                        hidden = checkpoint(lambda x, p: blk(x, xpos=p), hidden, spatial_pos, use_reentrant=False)
                    else:
                        hidden = blk(hidden, xpos=spatial_pos)
                else:
                    # Temporal-Local: (B*S_full, T, D)
                    # Each spatial position attends across time
                    hidden = hidden.reshape(B, T, S_full, D)
                    hidden = hidden.permute(0, 2, 1, 3).reshape(B * S_full, T, D)

                    if self.gradient_checkpointing and self.training:
                        hidden = checkpoint(lambda x, p: blk(x, xpos=p), hidden, time_pos, use_reentrant=False)
                    else:
                        hidden = blk(hidden, xpos=time_pos)

                    # Reshape back to (B*T, S_full, D)
                    hidden = hidden.reshape(B, S_full, T, D)
                    hidden = hidden.permute(0, 2, 1, 3).reshape(B * T, S_full, D)

            # Collect outputs for last two layers
            if i >= self.dec_depth - 2:
                all_outputs.append(hidden.clone())

        # 5. Concatenate last two layer outputs (like Pi3/teacher)
        assert len(all_outputs) == 2, f"Expected 2 outputs, got {len(all_outputs)}"
        output = torch.cat(all_outputs, dim=-1)  # (B*T, S_full, 2*D)

        return output

    def get_output_dim(self) -> int:
        """Return output dimension for distillation loss."""
        return self.output_dim

    def get_num_patches(self, H: int, W: int) -> int:
        """Return number of patches + register tokens."""
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        return patch_h * patch_w + self.patch_start_idx


class TemporalViTSmall(TemporalViT):
    """Smaller variant with 12 layers for faster training/inference."""

    def __init__(self, **kwargs):
        kwargs.setdefault('dec_depth', 12)
        kwargs.setdefault('dec_embed_dim', 768)
        kwargs.setdefault('dec_num_heads', 12)
        super().__init__(**kwargs)


class TemporalViTTiny(TemporalViT):
    """Tiny variant with 6 layers for edge deployment."""

    def __init__(self, **kwargs):
        kwargs.setdefault('dec_depth', 6)
        kwargs.setdefault('dec_embed_dim', 512)
        kwargs.setdefault('dec_num_heads', 8)
        super().__init__(**kwargs)
