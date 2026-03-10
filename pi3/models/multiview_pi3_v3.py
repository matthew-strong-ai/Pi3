"""
Multi-View Pi3 V3: DVGT-Inspired Factorized Attention with Single-View Compatibility.

Key innovation: Extends Pi3's 2-way alternating attention to 3-way (like V2),
but with proper single-view compatibility:
  - Layer % 3 == 0: Intra-View  (B*C*T, S, D) - spatial attention per frame
  - Layer % 3 == 1: Cross-View  (B*T, C*S, D) - cameras at same timestep [SKIP if C=1]
  - Layer % 3 == 2: Cross-Time  (B*C, T*S, D) - temporal per camera

When C=1 (single-view):
  - Cross-view layers are skipped entirely
  - Behaves identically to AutoregressivePi3 with pretrained weights

Cross-view layers are identity-initialized so pretrained single-view weights
produce the same output until multi-view training updates them.
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from functools import partial
from copy import deepcopy
from typing import Optional, Dict

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .layers.autoregressive_transformer import AutoregressiveTokenTransformer
from .layers.camera_embedding import CameraEmbedding, NormalizedPoseCameraEmbedding
from .dinov2.hub.backbones import dinov2_vitl14_reg


class EgomotionHead(nn.Module):
    """
    Predicts vehicle egomotion (pose) from aggregated features.
    Uses 6D rotation representation for continuity.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.rotation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),  # 6D rotation
        )
        self.translation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, T, D) aggregated features per timestep
        Returns:
            poses: (B, T, 4, 4) transformation matrices
        """
        B, T, D = features.shape
        rot_6d = self.rotation_head(features)
        R = self._rotation_6d_to_matrix(rot_6d)
        t = self.translation_head(features)

        poses = torch.eye(4, device=features.device, dtype=features.dtype)
        poses = poses.unsqueeze(0).unsqueeze(0).expand(B, T, 4, 4).clone()
        poses[:, :, :3, :3] = R
        poses[:, :, :3, 3] = t
        return poses

    def _rotation_6d_to_matrix(self, rot_6d: torch.Tensor) -> torch.Tensor:
        a1, a2 = rot_6d[..., :3], rot_6d[..., 3:]
        b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack([b1, b2, b3], dim=-1)


class MultiViewPi3V3(nn.Module):
    """
    Multi-View Pi3 V3 with DVGT-inspired factorized attention.

    The decoder uses 3-way alternating attention:
    1. Intra-View: Each (camera, timestep) attends spatially within itself
    2. Cross-View: Cameras at the same timestep attend to each other
    3. Cross-Time: Each camera attends across timesteps

    Key features for pretrained compatibility:
    - Cross-view layers skip when C=1 (single-view mode)
    - Cross-view layers are identity-initialized
    - Can load Pi3 weights and produce identical single-view outputs
    """

    def __init__(
        self,
        num_cameras: int = 6,
        n_current_frames: int = 3,
        n_future_frames: int = 3,
        pos_type: str = 'rope100',
        decoder_size: str = 'large',
        encoder_name: str = 'dinov2',
        # Camera embedding params
        camera_embedding_type: str = 'normalized_pose',  # 'normalized_pose', 'legacy', or 'none'
        # Autoregressive params
        ar_n_heads: int = 16,
        ar_n_layers: int = 8,
        ar_dropout: float = 0.1,
        # Optional heads
        use_segmentation_head: bool = False,
        segmentation_num_classes: int = 7,
        use_motion_head: bool = False,
        # Misc
        freeze_encoder: bool = True,
        freeze_decoders: bool = False,
        # V3-specific: identity init for cross-view layers
        cross_view_identity_init: bool = True,
        # Memory optimization
        gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.n_current_frames = n_current_frames
        self.n_future_frames = n_future_frames
        self.use_segmentation_head = use_segmentation_head
        self.segmentation_num_classes = segmentation_num_classes
        self.use_motion_head = use_motion_head
        self.cross_view_identity_init = cross_view_identity_init
        self.gradient_checkpointing = gradient_checkpointing

        # ----------------------
        #        Encoder
        # ----------------------
        if encoder_name == 'dinov2':
            self.encoder = dinov2_vitl14_reg(pretrained=True)
            self.patch_size = 14
            del self.encoder.mask_token
        else:
            raise NotImplementedError(f"Encoder {encoder_name} not implemented")

        if freeze_encoder:
            self.encoder.requires_grad_(False)
            self.encoder.eval()

        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024

        # ----------------------
        #   Decoder Config
        # ----------------------
        # V3 uses 3-way factorized attention (intra-view, cross-view, cross-time)
        # To match Pi3's effective depth when C=1 (cross-view skipped), we need:
        #   V3 depth = Pi3 depth * 3/2
        # This ensures (intra-view + cross-time) layers = Pi3 total layers
        #
        # Pi3 base (24 layers) → V3 36 layers (12 intra + 12 cross-view + 12 cross-time)
        # Pi3 large (36 layers) → V3 54 layers (18 intra + 18 cross-view + 18 cross-time)
        if decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 54  # 36 * 3/2 = 54 (matches Pi3 large when C=1)
            self.pi3_equiv_depth = 36  # Pi3 equivalent for weight loading
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 36  # 24 * 3/2 = 36 (matches Pi3 base when C=1)
            self.pi3_equiv_depth = 24  # Pi3 equivalent for weight loading
        else:
            raise NotImplementedError

        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth

        # Projection from encoder to decoder dim (if different)
        if enc_embed_dim != dec_embed_dim:
            self.enc_to_dec_proj = nn.Linear(enc_embed_dim, dec_embed_dim)
        else:
            self.enc_to_dec_proj = nn.Identity()

        # ----------------------
        #   Camera Embeddings
        # ----------------------
        self.camera_embedding_type = camera_embedding_type
        if camera_embedding_type == 'normalized_pose':
            self.camera_embedding = NormalizedPoseCameraEmbedding(d_model=dec_embed_dim)
        elif camera_embedding_type == 'legacy':
            self.camera_embedding = CameraEmbedding(
                d_model=dec_embed_dim,
                max_cameras=8,
                use_extrinsics=True,
            )
        else:
            self.camera_embedding = None

        # ----------------------
        #  Positional Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        if self.pos_type.startswith('rope'):
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #   3-Way Decoder
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

        # Initialize cross-view layers to identity (output ≈ 0 before residual)
        if cross_view_identity_init:
            self._init_cross_view_identity()

        # ----------------------
        #    Register Tokens
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, 1, num_register_tokens, self.dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        # Autoregressive Transformer
        # ----------------------
        self.autoregressive_transformer = AutoregressiveTokenTransformer(
            d_model=2 * self.dec_embed_dim,
            n_heads=ar_n_heads,
            n_layers=ar_n_layers,
            d_ff=3 * self.dec_embed_dim,
            dropout=ar_dropout,
            n_future_frames=n_future_frames,
            max_seq_len=15
        )

        # ----------------------
        #   Task Decoders
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(
            patch_size=self.patch_size,
            dec_embed_dim=1024,
            output_dim=3
        )

        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(
            patch_size=self.patch_size,
            dec_embed_dim=1024,
            output_dim=1
        )

        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # Egomotion (vehicle-level)
        self.egomotion_head = EgomotionHead(input_dim=2 * self.dec_embed_dim)

        # Optional heads
        if self.use_segmentation_head:
            self.segmentation_decoder = deepcopy(self.point_decoder)
            self.segmentation_head = LinearPts3d(
                patch_size=self.patch_size,
                dec_embed_dim=1024,
                output_dim=self.segmentation_num_classes,
            )

        if self.use_motion_head:
            self.motion_decoder = deepcopy(self.point_decoder)
            self.motion_head = LinearPts3d(
                patch_size=self.patch_size,
                dec_embed_dim=1024,
                output_dim=1,
            )

        # Image normalization
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # ----------------------
        #  Metric Scale Prediction (MapAnything-style)
        # ----------------------
        # Learnable scale token - global per batch
        self.scale_token = nn.Parameter(torch.zeros(1, 1, self.dec_embed_dim))
        nn.init.normal_(self.scale_token, std=0.02)

        # Scale prediction MLP (MapAnything: 2-layer with ReLU)
        # Input: aggregated decoder features, Output: log(scale)
        self.scale_head = nn.Sequential(
            nn.Linear(2 * self.dec_embed_dim, self.dec_embed_dim // 4),
            nn.ReLU(),
            nn.Linear(self.dec_embed_dim // 4, 1),
        )
        # Initialize to predict log(1) = 0 → scale = 1.0
        nn.init.zeros_(self.scale_head[-1].weight)
        nn.init.zeros_(self.scale_head[-1].bias)

        if freeze_decoders:
            self._freeze_decoders()

    def _init_cross_view_identity(self):
        """
        Initialize cross-view layers (layer % 3 == 1) to approximate identity.

        This ensures that when loading pretrained single-view weights,
        the cross-view layers contribute ~0 to the output (due to residual connection).
        """
        for i, blk in enumerate(self.decoder):
            if i % 3 == 1:  # Cross-view layers
                # Zero out the attention output projection
                # This makes attention output ≈ 0, so residual = input
                if hasattr(blk.attn, 'proj'):
                    nn.init.zeros_(blk.attn.proj.weight)
                    if blk.attn.proj.bias is not None:
                        nn.init.zeros_(blk.attn.proj.bias)
                # Also zero out MLP output for safety
                if hasattr(blk, 'mlp') and hasattr(blk.mlp, 'fc2'):
                    nn.init.zeros_(blk.mlp.fc2.weight)
                    if blk.mlp.fc2.bias is not None:
                        nn.init.zeros_(blk.mlp.fc2.bias)

    def _freeze_decoders(self):
        """Freeze point, conf, and camera decoders."""
        for module in [self.point_decoder, self.conf_decoder, self.camera_decoder,
                       self.point_head, self.conf_head, self.camera_head]:
            for param in module.parameters():
                param.requires_grad = False
        print("Froze point, conf, and camera decoders and heads")

    def decode_3way(self, hidden: torch.Tensor, C: int, T: int, H: int, W: int) -> tuple:
        """
        3-way factorized attention decoder (DVGT-inspired).

        Pattern (when C > 1):
          Layer % 3 == 0: Intra-View   (B*C*T, S, D) - spatial per camera per time
          Layer % 3 == 1: Cross-View   (B*T, C*S, D) - cameras at same timestep
          Layer % 3 == 2: Cross-Time   (B*C, T*S, D) - temporal per camera

        When C=1 (single-view), falls back to Pi3's 2-way pattern using only
        the intra-view (spatial) and cross-time (temporal) blocks, matching
        the exact layer execution order of Pi3 for weight compatibility.

        Args:
            hidden: (B*C*T, S, D) encoded features
            C: number of cameras
            T: number of timesteps
            H, W: image dimensions

        Returns:
            hidden: (B*C*T, S+reg, 2*D) decoded features
            pos: (B*C*T, S+reg, 2) positions
        """
        BCT, S, D = hidden.shape
        B = BCT // (C * T)
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Add register tokens: (B, C, T, reg, D)
        register_token = self.register_token.expand(B, C, T, -1, -1)
        hidden = hidden.reshape(B, C, T, S, D)
        hidden = torch.cat([register_token, hidden], dim=3)  # (B, C, T, S+reg, D)
        S_full = hidden.shape[3]  # S + num_register_tokens

        # Get base positions for spatial attention
        base_pos = self.position_getter(1, patch_h, patch_w, hidden.device)  # (1, S, 2)

        # Add offset for register tokens
        if self.patch_start_idx > 0:
            base_pos = base_pos + 1
            pos_special = torch.zeros(1, self.patch_start_idx, 2, device=hidden.device, dtype=base_pos.dtype)
            base_pos = torch.cat([pos_special, base_pos], dim=1)  # (1, S_full, 2)

        # Single-view mode: use Pi3-compatible 2-way pattern
        if C == 1:
            return self._decode_2way_pi3_compat(hidden, B, T, S_full, D, patch_h, patch_w, base_pos)

        # Multi-view mode: use full 3-way pattern
        return self._decode_3way_multiview(hidden, B, C, T, S_full, D, patch_h, patch_w, base_pos)

    def _decode_2way_pi3_compat(self, hidden, B, T, S_full, D, patch_h, patch_w, base_pos):
        """
        Pi3-compatible 2-way decoding for single-view (C=1).

        EXACTLY matches Pi3's decode() function:
        - Layer % 2 == 0: spatial (B*N, hw, D)
        - Layer % 2 == 1: temporal (B, N*hw, D)
        - Same pos reshaping (no time offsets - Pi3 doesn't use them)
        - Uses intra-view blocks (indices 0,3,6,...) for Pi3's spatial layers
        - Uses cross-time blocks (indices 2,5,8,...) for Pi3's temporal layers

        This ensures IDENTICAL outputs to Pi3 when loaded with Pi3 weights.
        """
        # Build list of blocks to use: intra-view (0,3,6,...) and cross-time (2,5,8,...)
        # These correspond to Pi3's spatial (0,2,4,...) and temporal (1,3,5,...) layers
        intra_view_blocks = [self.decoder[i] for i in range(0, self.dec_depth, 3)]
        cross_time_blocks = [self.decoder[i] for i in range(2, self.dec_depth, 3)]

        # Interleave them in Pi3's 2-way pattern: spatial, temporal, spatial, temporal, ...
        n_pairs = min(len(intra_view_blocks), len(cross_time_blocks))
        blocks_2way = []
        for pair_idx in range(n_pairs):
            blocks_2way.append(('spatial', intra_view_blocks[pair_idx]))
            blocks_2way.append(('temporal', cross_time_blocks[pair_idx]))
        # Handle any remaining blocks (if counts don't match)
        if len(intra_view_blocks) > n_pairs:
            blocks_2way.append(('spatial', intra_view_blocks[n_pairs]))

        final_output = []

        # Start with hidden in (B*T, S_full, D) shape (like Pi3's B*N, hw, D)
        hidden = hidden.reshape(B * T, S_full, D)
        hw = S_full  # Renamed for clarity with Pi3

        # pos is (1, S_full, 2) from base_pos, expand to (B*T, S_full, 2)
        pos = base_pos.expand(B * T, -1, -1).contiguous()

        for i, (pattern_name, blk) in enumerate(blocks_2way):
            if pattern_name == 'spatial':
                # Spatial: (B*T, hw, D) - matches Pi3's (B*N, hw, D)
                pos = pos.reshape(B * T, hw, -1)
                hidden = hidden.reshape(B * T, hw, -1)
            else:
                # Temporal: (B, T*hw, D) - matches Pi3's (B, N*hw, D)
                pos = pos.reshape(B, T * hw, -1)
                hidden = hidden.reshape(B, T * hw, -1)

            # Apply attention block - EXACTLY like Pi3 (with optional checkpointing)
            if self.gradient_checkpointing and self.training:
                def run_block(blk, x, pos):
                    return blk(x, xpos=pos)
                hidden = checkpoint(run_block, blk, hidden, pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)

            # Collect last two layers (like Pi3's i+1 in [len-1, len])
            if i + 1 in [len(blocks_2way) - 1, len(blocks_2way)]:
                final_output.append(hidden.reshape(B * T, hw, -1))

        # Concatenate last two layer outputs - EXACTLY like Pi3
        hidden_out = torch.cat([final_output[0], final_output[1]], dim=-1)  # (B*T, S_full, 2*D)
        pos_out = pos.reshape(B * T, hw, -1)

        return hidden_out, pos_out

    def _decode_3way_multiview(self, hidden, B, C, T, S_full, D, patch_h, patch_w, base_pos):
        """
        Full 3-way decoding for multi-view (C > 1).

        Pattern:
          Layer % 3 == 0: Intra-View   (B*C*T, S_full, D)
          Layer % 3 == 1: Cross-View   (B*T, C*S_full, D)
          Layer % 3 == 2: Cross-Time   (B*C, T*S_full, D)
        """
        all_outputs = []

        for i, blk in enumerate(self.decoder):
            pattern = i % 3

            if pattern == 0:
                # Intra-View: (B*C*T, S_full, D)
                hidden_flat = hidden.reshape(B * C * T, S_full, D)
                pos = base_pos.expand(B * C * T, -1, -1)

            elif pattern == 1:
                # Cross-View: (B*T, C*S_full, D)
                hidden_flat = hidden.permute(0, 2, 1, 3, 4).reshape(B * T, C * S_full, D)
                # Positions: tile base_pos for each camera with camera offset
                pos_per_cam = base_pos.expand(C, -1, -1)  # (C, S_full, 2)
                # Add camera index to y-position to distinguish cameras
                cam_offsets = torch.arange(C, device=hidden.device).float() * (patch_h + 1)
                pos_per_cam = pos_per_cam.clone()
                pos_per_cam[:, :, 1] = pos_per_cam[:, :, 1] + cam_offsets.view(C, 1)
                pos = pos_per_cam.reshape(1, C * S_full, 2).expand(B * T, -1, -1)

            else:  # pattern == 2
                # Cross-Time: (B*C, T*S_full, D)
                hidden_flat = hidden.reshape(B * C, T * S_full, D)
                # Positions: tile base_pos for each timestep with time offset
                pos_per_time = base_pos.expand(T, -1, -1)  # (T, S_full, 2)
                # Add time index to x-position to distinguish timesteps
                time_offsets = torch.arange(T, device=hidden.device).float() * (patch_w + 1)
                pos_per_time = pos_per_time.clone()
                pos_per_time[:, :, 0] = pos_per_time[:, :, 0] + time_offsets.view(T, 1)
                pos = pos_per_time.reshape(1, T * S_full, 2).expand(B * C, -1, -1)

            # Apply attention block (with optional gradient checkpointing)
            if self.gradient_checkpointing and self.training:
                def run_block(blk, x, pos):
                    return blk(x, xpos=pos)
                hidden_flat = checkpoint(run_block, blk, hidden_flat, pos, use_reentrant=False)
            else:
                hidden_flat = blk(hidden_flat, xpos=pos)

            # Reshape back to (B, C, T, S_full, D)
            if pattern == 0:
                hidden = hidden_flat.reshape(B, C, T, S_full, D)
            elif pattern == 1:
                hidden = hidden_flat.reshape(B, T, C, S_full, D).permute(0, 2, 1, 3, 4)
            else:
                hidden = hidden_flat.reshape(B, C, T, S_full, D)

            all_outputs.append(hidden.reshape(B * C * T, S_full, D))

        # Concatenate last two layer outputs
        final_outputs = all_outputs[-2:]
        hidden_out = torch.cat(final_outputs, dim=-1)  # (B*C*T, S_full, 2*D)
        pos_out = base_pos.expand(B * C * T, -1, -1)

        return hidden_out, pos_out

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        n_future_frames_override: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-view perception.

        Args:
            batch: Dict containing:
                - images: (B, C, T, 3, H, W) multi-camera images
                - camera_extrinsics: (B, C, 4, 4) camera extrinsics (optional)

        Returns:
            Dict with predictions for each camera
        """
        images = batch['images']  # (B, C, T, 3, H, W)
        camera_extrinsics = batch.get('camera_extrinsics', None)
        camera_intrinsics = batch.get('camera_intrinsics', None)

        B, C, T, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Normalize images
        images = (images - self.image_mean) / self.image_std

        # 1. Per-camera-per-frame DINOv2 encoding
        imgs_flat = images.reshape(B * C * T, 3, H, W)

        if hasattr(self.encoder, 'forward_features'):
            hidden = self.encoder.forward_features(imgs_flat)
        else:
            hidden = self.encoder(imgs_flat, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # hidden: (B*C*T, S, enc_D)
        S = hidden.shape[1]

        # Project encoder features to decoder dimension
        hidden = self.enc_to_dec_proj(hidden)  # (B*C*T, S, dec_D)
        D = hidden.shape[2]

        # 2. Add camera embeddings (only if multi-view)
        if self.camera_embedding is not None and C > 1:
            hidden = hidden.reshape(B, C, T, S, D)
            if self.camera_embedding_type == 'normalized_pose':
                cam_embed = self.camera_embedding(camera_extrinsics, camera_intrinsics, (H, W))
            else:
                cam_embed = self.camera_embedding(C, camera_extrinsics, B)
            hidden = hidden + cam_embed
            hidden = hidden.reshape(B * C * T, S, D)

        # 3. 3-way alternating attention decoder
        hidden_decoded, pos = self.decode_3way(hidden, C, T, H, W)
        # hidden_decoded: (B*C*T, S_full, 2*D)

        # 4. Autoregressive future prediction (per camera)
        n_future = n_future_frames_override if n_future_frames_override is not None else self.n_future_frames

        # Process AR transformer per camera (cameras are independent for future prediction)
        S_full = hidden_decoded.shape[1]
        hidden_per_cam = hidden_decoded.reshape(B * C, T, S_full, -1)
        hidden_per_cam = hidden_per_cam.reshape(B * C * T, S_full, -1)
        pos_per_cam = pos.reshape(B * C * T, S_full, -1)

        all_hidden, all_pos = self.autoregressive_transformer(
            hidden_per_cam, T, pos_per_cam, n_future_frames_override=n_future
        )
        total_frames = T + n_future
        # all_hidden: (B*C*total_frames, S_full, 2*D)

        # 5. Task predictions
        point_hidden = self.point_decoder(all_hidden, xpos=all_pos)
        conf_hidden = self.conf_decoder(all_hidden, xpos=all_pos)
        camera_hidden = self.camera_decoder(all_hidden, xpos=all_pos)

        if self.use_segmentation_head:
            seg_hidden = self.segmentation_decoder(all_hidden, xpos=all_pos)
        if self.use_motion_head:
            motion_hidden = self.motion_decoder(all_hidden, xpos=all_pos)

        # 5.5. Predict metric scale factor (MapAnything-style)
        # Aggregate decoder features across all cameras, timesteps, and spatial dims
        scale_features = all_hidden.reshape(B, C * total_frames, S_full, -1)
        scale_features = scale_features.mean(dim=[1, 2])  # (B, 2*D)
        log_scale = self.scale_head(scale_features)  # (B, 1)
        metric_scale = torch.exp(log_scale)  # Ensure positive, centered at 1.0

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Points (up-to-scale relative depth)
            point_hidden = point_hidden.float()
            local_points_flat = self.point_head(
                [point_hidden[:, self.patch_start_idx:]], (H, W)
            )
            local_points_raw = local_points_flat.reshape(B, C, total_frames, H, W, -1)
            xy, z_log = local_points_raw.split([2, 1], dim=-1)
            z_relative = torch.exp(z_log)  # Up-to-scale depth

            # Store relative (up-to-scale) local points
            local_points_relative = torch.cat([xy * z_relative, z_relative], dim=-1)

            # Apply metric scale: depth_metric = scale * depth_relative
            z_metric = metric_scale.view(B, 1, 1, 1, 1, 1) * z_relative
            local_points = torch.cat([xy * z_metric, z_metric], dim=-1)

            # Confidence
            conf_hidden = conf_hidden.float()
            conf_flat = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W))
            conf = conf_flat.reshape(B, C, total_frames, H, W, -1)

            # Camera poses (per-camera)
            camera_hidden = camera_hidden.float()
            camera_poses_flat = self.camera_head(
                camera_hidden[:, self.patch_start_idx:], patch_h, patch_w
            )
            camera_poses = camera_poses_flat.reshape(B, C, total_frames, 4, 4)

            # Unproject local points
            points = torch.einsum(
                'bcnij, bcnhwj -> bcnhwi',
                camera_poses,
                homogenize_points(local_points)
            )[..., :3]

            # Segmentation
            segmentation = None
            if self.use_segmentation_head:
                seg_hidden = seg_hidden.float()
                seg_flat = self.segmentation_head(
                    [seg_hidden[:, self.patch_start_idx:]], (H, W)
                )
                segmentation = seg_flat.reshape(B, C, total_frames, H, W, -1)

            # Motion
            motion = None
            if self.use_motion_head:
                motion_hidden = motion_hidden.float()
                motion_flat = self.motion_head(
                    [motion_hidden[:, self.patch_start_idx:]], (H, W)
                )
                motion = motion_flat.reshape(B, C, total_frames, H, W, -1)

        # 6. Egomotion (aggregate across cameras)
        # Mean pool across cameras and spatial dims
        ego_features = all_hidden.reshape(B, C, total_frames, S_full, -1)
        ego_features = ego_features.mean(dim=[1, 3])  # (B, total_frames, 2*D)
        vehicle_poses = self.egomotion_head(ego_features)  # (B, total_frames, 4, 4)

        # Compute camera world poses from egomotion + extrinsics
        if camera_extrinsics is not None:
            camera_world_poses = torch.einsum(
                'btij, bcjk -> bctik',
                vehicle_poses,
                camera_extrinsics
            )
        else:
            camera_world_poses = None

        result = {
            'points': points,                          # (B, C, T+M, H, W, 3) - metric world frame
            'local_points': local_points,              # (B, C, T+M, H, W, 3) - metric camera frame
            'local_points_relative': local_points_relative,  # (B, C, T+M, H, W, 3) - up-to-scale
            'metric_scale': metric_scale,              # (B, 1) - predicted scale factor
            'conf': conf,                              # (B, C, T+M, H, W, 1)
            'camera_poses': camera_poses,              # (B, C, T+M, 4, 4)
            'vehicle_poses': vehicle_poses,            # (B, T+M, 4, 4)
            'decoder_hidden': all_hidden,              # (B*C*(T+M), S, 2*D) - for distillation
            'n_current_frames': T,
            'n_future_frames': n_future,
        }

        if camera_world_poses is not None:
            result['camera_world_poses'] = camera_world_poses

        if segmentation is not None:
            result['segmentation'] = segmentation

        if motion is not None:
            result['motion'] = motion

        return result

    def load_pi3_weights(self, pi3_checkpoint_path: str, strict: bool = False):
        """
        Load weights from a single-camera Pi3 checkpoint.

        Maps Pi3's 2-way decoder layers to V3's 3-way structure:
        - Pi3 spatial layers (0,2,4,...) → V3 intra-view (0,3,6,...)
        - Pi3 temporal layers (1,3,5,...) → V3 cross-time (2,5,8,...)
        - V3 cross-view layers (1,4,7,...) → keep identity-initialized

        This ensures single-view inference produces identical results to Pi3.
        """
        state_dict = torch.load(pi3_checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Build mapping for decoder layers
        # Pi3 has 36 layers with 2-way pattern (spatial/temporal alternating)
        # V3 has 36 layers with 3-way pattern (intra/cross-view/cross-time)
        #
        # For V3 to match Pi3 in single-view mode:
        # - V3 intra-view (0,3,6,...) should match Pi3 spatial (0,2,4,...)
        # - V3 cross-time (2,5,8,...) should match Pi3 temporal (1,3,5,...)
        # - V3 cross-view (1,4,7,...) stay identity (skipped when C=1)

        new_state_dict = {}
        skipped = []
        decoder_mapped = 0
        encoder_loaded = 0

        for k, v in state_dict.items():
            # Load encoder weights directly (same architecture)
            if k.startswith('encoder.'):
                if k in self.state_dict() and v.shape == self.state_dict()[k].shape:
                    new_state_dict[k] = v
                    encoder_loaded += 1
                else:
                    skipped.append(f"{k} (encoder shape mismatch)")
                continue

            # Handle register token shape
            if k == 'register_token':
                # Pi3: (1, 1, reg, D) -> V3: (1, 1, 1, reg, D)
                if v.dim() == 4:
                    v = v.unsqueeze(2)
                if v.shape == self.state_dict()[k].shape:
                    new_state_dict[k] = v
                else:
                    skipped.append(f"{k} (shape mismatch)")
                continue

            # Map decoder layers
            if k.startswith('decoder.'):
                # Extract layer index: decoder.{idx}.{rest}
                parts = k.split('.')
                pi3_layer_idx = int(parts[1])
                rest = '.'.join(parts[2:])

                # Determine V3 target layer
                if pi3_layer_idx % 2 == 0:
                    # Pi3 spatial → V3 intra-view
                    # Pi3 layer 0,2,4,... → V3 layer 0,3,6,...
                    v3_layer_idx = (pi3_layer_idx // 2) * 3
                else:
                    # Pi3 temporal → V3 cross-time
                    # Pi3 layer 1,3,5,... → V3 layer 2,5,8,...
                    v3_layer_idx = (pi3_layer_idx // 2) * 3 + 2

                new_key = f'decoder.{v3_layer_idx}.{rest}'
                if new_key in self.state_dict() and v.shape == self.state_dict()[new_key].shape:
                    new_state_dict[new_key] = v
                    decoder_mapped += 1
                else:
                    skipped.append(f"{k} → {new_key} (not found or shape mismatch)")
                continue

            # Direct mapping for other weights
            if k in self.state_dict():
                if v.shape == self.state_dict()[k].shape:
                    new_state_dict[k] = v
                else:
                    skipped.append(f"{k} (shape mismatch: {v.shape} vs {self.state_dict()[k].shape})")
            else:
                skipped.append(k)

        # Load matched weights
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        print(f"Loaded Pi3 weights into V3:")
        print(f"  Matched: {len(new_state_dict)} (encoder: {encoder_loaded}, decoder: {decoder_mapped})")
        print(f"  Skipped: {len(skipped)}")
        print(f"  Missing in model: {len(missing)}")
        print(f"  Cross-view layers (1,4,7,...) kept identity-initialized")

        return {'matched': len(new_state_dict), 'skipped': skipped, 'missing': missing}
