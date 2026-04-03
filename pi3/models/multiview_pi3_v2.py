"""
Multi-View Pi3 V2: Interleaved Cross-Camera and Temporal Attention.

Key innovation: Extends Pi3's 2-way alternating attention to 3-way:
  - Layer % 3 == 0: Within-frame (B*C*T, S, D) - spatial attention
  - Layer % 3 == 1: Cross-camera (B*T, C*S, D) - cameras at same timestep
  - Layer % 3 == 2: Cross-time (B*C, T*S, D) - temporal per camera

This allows cameras to share information throughout temporal reasoning,
not just at the beginning like MultiViewPi3 V1.
"""

import torch
import torch.nn as nn
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
from .layers.camera_embedding import CameraEmbedding
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


class MultiViewPi3V2(nn.Module):
    """
    Multi-View Pi3 V2 with interleaved 3-way attention.

    The decoder alternates between three attention patterns:
    1. Within-frame: Each (camera, timestep) attends spatially
    2. Cross-camera: Cameras at the same timestep attend to each other
    3. Cross-time: Each camera attends across timesteps

    This is the natural extension of Pi3's 2-way pattern to multi-camera.
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
        use_camera_embeddings: bool = True,
        embed_extrinsics: bool = True,
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
    ):
        super().__init__()

        self.num_cameras = num_cameras
        self.n_current_frames = n_current_frames
        self.n_future_frames = n_future_frames
        self.use_segmentation_head = use_segmentation_head
        self.segmentation_num_classes = segmentation_num_classes
        self.use_motion_head = use_motion_head

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
        if decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
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
        if use_camera_embeddings:
            self.camera_embedding = CameraEmbedding(
                d_model=dec_embed_dim,  # Use decoder dim since we project first
                max_cameras=8,
                use_extrinsics=embed_extrinsics,
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
        # Same BlockRope as Pi3 - the magic is in how we reshape between layers
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
        self.dec_embed_dim = dec_embed_dim
        self.dec_depth = dec_depth

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

        if freeze_decoders:
            self._freeze_decoders()

    def _freeze_decoders(self):
        """Freeze point, conf, and camera decoders."""
        for module in [self.point_decoder, self.conf_decoder, self.camera_decoder,
                       self.point_head, self.conf_head, self.camera_head]:
            for param in module.parameters():
                param.requires_grad = False
        print("Froze point, conf, and camera decoders and heads")

    def decode_3way(self, hidden: torch.Tensor, C: int, T: int, H: int, W: int) -> tuple:
        """
        3-way alternating attention decoder.

        This is the key innovation: extends Pi3's 2-way pattern to 3-way.

        Pattern:
          Layer % 3 == 0: Within-frame   (B*C*T, S, D) - spatial per camera per time
          Layer % 3 == 1: Cross-camera   (B*T, C*S, D) - cameras at same timestep
          Layer % 3 == 2: Cross-time     (B*C, T*S, D) - temporal per camera

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

        final_outputs = []

        for i, blk in enumerate(self.decoder):
            pattern = i % 3

            if pattern == 0:
                # Within-frame: (B*C*T, S_full, D)
                hidden_flat = hidden.reshape(B * C * T, S_full, D)
                pos = base_pos.expand(B * C * T, -1, -1)

            elif pattern == 1:
                # Cross-camera: (B*T, C*S_full, D)
                hidden_flat = hidden.permute(0, 2, 1, 3, 4).reshape(B * T, C * S_full, D)
                # Positions: tile base_pos for each camera with camera offset
                pos_per_cam = base_pos.expand(C, -1, -1)  # (C, S_full, 2)
                # Add camera index to y-position to distinguish cameras
                cam_offsets = torch.arange(C, device=hidden.device).float() * (patch_h + 1)
                pos_per_cam = pos_per_cam.clone()
                pos_per_cam[:, :, 1] = pos_per_cam[:, :, 1] + cam_offsets.view(C, 1)
                pos = pos_per_cam.reshape(1, C * S_full, 2).expand(B * T, -1, -1)

            else:  # pattern == 2
                # Cross-time: (B*C, T*S_full, D)
                hidden_flat = hidden.reshape(B * C, T * S_full, D)
                # Positions: tile base_pos for each timestep with time offset
                pos_per_time = base_pos.expand(T, -1, -1)  # (T, S_full, 2)
                # Add time index to x-position to distinguish timesteps
                time_offsets = torch.arange(T, device=hidden.device).float() * (patch_w + 1)
                pos_per_time = pos_per_time.clone()
                pos_per_time[:, :, 0] = pos_per_time[:, :, 0] + time_offsets.view(T, 1)
                pos = pos_per_time.reshape(1, T * S_full, 2).expand(B * C, -1, -1)

            # Apply attention block
            hidden_flat = blk(hidden_flat, xpos=pos)

            # Reshape back to (B, C, T, S_full, D)
            if pattern == 0:
                hidden = hidden_flat.reshape(B, C, T, S_full, D)
            elif pattern == 1:
                hidden = hidden_flat.reshape(B, T, C, S_full, D).permute(0, 2, 1, 3, 4)
            else:
                hidden = hidden_flat.reshape(B, C, T, S_full, D)

            # Collect final layer outputs (last 2 layers for concatenation)
            if i >= self.dec_depth - 2:
                final_outputs.append(hidden.reshape(B * C * T, S_full, D))

        # Concatenate last two layer outputs (like Pi3)
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

        # 2. Add camera embeddings
        if self.camera_embedding is not None:
            # Reshape hidden to (B, C, T, S, D) to add camera embeddings
            hidden = hidden.reshape(B, C, T, S, D)
            cam_embed = self.camera_embedding(C, camera_extrinsics, B)  # (B, C, 1, 1, D)
            hidden = hidden + cam_embed
            hidden = hidden.reshape(B * C * T, S, D)

        # 3. 3-way alternating attention decoder
        hidden_decoded, pos = self.decode_3way(hidden, C, T, H, W)
        # hidden_decoded: (B*C*T, S_full, 2*D)

        # 4. Autoregressive future prediction (per camera)
        n_future = n_future_frames_override if n_future_frames_override is not None else self.n_future_frames

        # Process AR transformer per camera (cameras are independent for future prediction)
        # Reshape to (B*C, T, S_full, 2*D) -> treat each camera as batch item
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

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Points
            point_hidden = point_hidden.float()
            local_points_flat = self.point_head(
                [point_hidden[:, self.patch_start_idx:]], (H, W)
            )
            local_points_raw = local_points_flat.reshape(B, C, total_frames, H, W, -1)
            xy, z = local_points_raw.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

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
            'points': points,                          # (B, C, T+M, H, W, 3)
            'local_points': local_points,              # (B, C, T+M, H, W, 3)
            'conf': conf,                              # (B, C, T+M, H, W, 1)
            'camera_poses': camera_poses,              # (B, C, T+M, 4, 4)
            'vehicle_poses': vehicle_poses,            # (B, T+M, 4, 4)
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

        Maps Pi3 weights to MultiViewPi3V2, handling the architectural differences.
        """
        state_dict = torch.load(pi3_checkpoint_path, map_location='cpu', weights_only=False)

        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        # Filter and map weights
        new_state_dict = {}
        skipped = []

        for k, v in state_dict.items():
            # Skip encoder weights if we're using pretrained
            if k.startswith('encoder.'):
                skipped.append(k)
                continue

            # Handle register token shape difference
            if k == 'register_token':
                # Pi3: (1, 1, reg, D) -> V2: (1, 1, 1, reg, D)
                if v.dim() == 4:
                    v = v.unsqueeze(2)
                if v.shape == self.state_dict()[k].shape:
                    new_state_dict[k] = v
                else:
                    skipped.append(f"{k} (shape mismatch: {v.shape} vs {self.state_dict()[k].shape})")
                continue

            # Direct mapping for decoder, task heads, etc.
            if k in self.state_dict():
                if v.shape == self.state_dict()[k].shape:
                    new_state_dict[k] = v
                else:
                    skipped.append(f"{k} (shape mismatch: {v.shape} vs {self.state_dict()[k].shape})")
            else:
                skipped.append(k)

        # Load matched weights
        missing, unexpected = self.load_state_dict(new_state_dict, strict=False)

        print(f"Loaded Pi3 weights:")
        print(f"  Matched: {len(new_state_dict)}")
        print(f"  Skipped: {len(skipped)}")
        print(f"  Missing in model: {len(missing)}")

        return {'matched': len(new_state_dict), 'skipped': skipped, 'missing': missing}
