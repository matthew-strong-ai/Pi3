"""
Cross-Camera Attention Layers for Multi-View Pi3.

Enables information sharing between different camera views at each timestep
through cross-attention mechanisms.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention
        XFORMERS_AVAILABLE = True
    else:
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False


class CrossCameraAttentionBlock(nn.Module):
    """
    Single cross-camera attention block.

    At each spatial location, tokens from different cameras attend to each other.
    This enables the model to learn correspondences between overlapping views.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        """
        Args:
            d_model: Hidden dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            mlp_ratio: MLP hidden dim ratio
        """
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)

        # Multi-head attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        # MLP
        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        camera_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with cross-camera attention.

        Args:
            x: (B, C, T, S, D) per-camera features
                B = batch, C = cameras, T = time, S = spatial tokens, D = dim
            camera_mask: Optional (C, C) attention mask between cameras
                True = can attend, False = cannot attend

        Returns:
            x: (B, C, T, S, D) updated features
        """
        B, C, T, S, D = x.shape

        # Reshape for cross-camera attention at each (time, spatial) position
        # We want cameras to attend to each other
        # Reshape to (B*T*S, C, D) - treat each position independently
        x_flat = x.permute(0, 2, 3, 1, 4).reshape(B * T * S, C, D)

        # Apply attention
        x_flat = x_flat + self._attention(self.norm1(x_flat), camera_mask)

        # Apply MLP
        x_flat = x_flat + self.mlp(self.norm2(x_flat))

        # Reshape back
        x = x_flat.reshape(B, T, S, C, D).permute(0, 3, 1, 2, 4)

        return x

    def _attention(
        self,
        x: torch.Tensor,
        camera_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Multi-head attention across cameras.

        Args:
            x: (N, C, D) where N = B*T*S
            camera_mask: Optional (C, C) mask

        Returns:
            out: (N, C, D)
        """
        N, C, D = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).reshape(N, C, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(N, C, self.n_heads, self.head_dim)
        v = self.v_proj(x).reshape(N, C, self.n_heads, self.head_dim)

        # Use xformers if available for efficiency
        # Skip xformers for small sequence lengths (C < 8) - flash attention has minimum seq requirements
        if XFORMERS_AVAILABLE and x.is_cuda and C >= 8:
            # xformers expects (B, M, H, K)
            out = memory_efficient_attention(q, k, v)
            out = out.reshape(N, C, D)
        else:
            # Standard attention: transpose for (N, heads, C, head_dim)
            q = q.transpose(1, 2)  # (N, heads, C, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (N, heads, C, C)

            # Apply camera mask if provided
            if camera_mask is not None:
                # camera_mask: (C, C) - True = can attend
                mask = ~camera_mask  # Invert: True = cannot attend
                scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)

            out = torch.matmul(attn, v)  # (N, heads, C, head_dim)
            out = out.transpose(1, 2).reshape(N, C, D)

        out = self.out_proj(out)
        out = self.proj_dropout(out)

        return out


class CrossCameraAttention(nn.Module):
    """
    Multi-layer cross-camera attention module.

    Stacks multiple CrossCameraAttentionBlocks to enable deep
    information sharing between camera views.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 6,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        """
        Args:
            d_model: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of attention layers
            dropout: Dropout rate
            mlp_ratio: MLP hidden dim ratio
        """
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            CrossCameraAttentionBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(n_layers)
        ])

        # Final normalization
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        camera_embed: Optional[torch.Tensor] = None,
        camera_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-camera attention.

        Args:
            hidden: (B, C, T, S, D) per-camera features
            camera_embed: Optional (B, C, 1, 1, D) camera embeddings
            camera_mask: Optional (C, C) attention mask

        Returns:
            hidden: (B, C, T, S, D) cross-camera enriched features
        """
        # Add camera embeddings if provided
        if camera_embed is not None:
            hidden = hidden + camera_embed

        # Apply cross-camera attention layers
        for layer in self.layers:
            hidden = layer(hidden, camera_mask)

        # Final normalization
        B, C, T, S, D = hidden.shape
        hidden = self.norm(hidden.reshape(-1, D)).reshape(B, C, T, S, D)

        return hidden


class SpatialCrossCameraAttention(nn.Module):
    """
    Spatial cross-camera attention that operates on full spatial feature maps.

    Instead of treating each spatial position independently, this allows
    attention over all spatial positions across all cameras. More expressive
    but also more memory-intensive.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 4,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_layers = n_layers

        self.layers = nn.ModuleList([
            SpatialCrossCameraBlock(
                d_model=d_model,
                n_heads=n_heads,
                dropout=dropout,
                mlp_ratio=mlp_ratio,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden: torch.Tensor,
        camera_embed: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply spatial cross-camera attention.

        Args:
            hidden: (B, C, T, S, D) per-camera features
            camera_embed: Optional (B, C, 1, 1, D) camera embeddings

        Returns:
            hidden: (B, C, T, S, D) cross-camera enriched features
        """
        if camera_embed is not None:
            hidden = hidden + camera_embed

        for layer in self.layers:
            hidden = layer(hidden)

        B, C, T, S, D = hidden.shape
        hidden = self.norm(hidden.reshape(-1, D)).reshape(B, C, T, S, D)

        return hidden


class SpatialCrossCameraBlock(nn.Module):
    """
    Block that attends over all (camera, spatial) pairs per timestep.

    This is more expressive than position-wise cross-camera attention
    but scales as O((C*S)^2) per timestep.
    """

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 16,
        dropout: float = 0.0,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.norm1 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        mlp_hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T, S, D)

        Returns:
            x: (B, C, T, S, D)
        """
        B, C, T, S, D = x.shape

        # Reshape: merge camera and spatial dims, process each timestep
        # (B, T, C*S, D)
        x_merged = x.permute(0, 2, 1, 3, 4).reshape(B, T, C * S, D)

        # Apply attention per timestep
        x_merged = x_merged + self._attention(self.norm1(x_merged))
        x_merged = x_merged + self.mlp(self.norm2(x_merged))

        # Reshape back
        x = x_merged.reshape(B, T, C, S, D).permute(0, 2, 1, 3, 4)

        return x

    def _attention(self, x: torch.Tensor) -> torch.Tensor:
        """
        Attention over merged (camera, spatial) dimension per timestep.

        Args:
            x: (B, T, CS, D) where CS = C * S

        Returns:
            out: (B, T, CS, D)
        """
        B, T, CS, D = x.shape

        # Reshape for batch processing all timesteps
        x_flat = x.reshape(B * T, CS, D)

        q = self.q_proj(x_flat).reshape(B * T, CS, self.n_heads, self.head_dim)
        k = self.k_proj(x_flat).reshape(B * T, CS, self.n_heads, self.head_dim)
        v = self.v_proj(x_flat).reshape(B * T, CS, self.n_heads, self.head_dim)

        if XFORMERS_AVAILABLE and x.is_cuda:
            out = memory_efficient_attention(q, k, v)
            out = out.reshape(B * T, CS, D)
        else:
            q = q.transpose(1, 2)  # (B*T, heads, CS, head_dim)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn = F.softmax(scores, dim=-1)
            attn = self.attn_dropout(attn)

            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).reshape(B * T, CS, D)

        out = self.out_proj(out)
        out = self.proj_dropout(out)

        return out.reshape(B, T, CS, D)
