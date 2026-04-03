"""
Multi-View Pi3 Model for Multi-Camera 3D Perception.

Takes synchronized images from multiple cameras and predicts:
- Per-camera 3D point maps
- Vehicle egomotion (camera poses are derived from egomotion + extrinsics)
- Optional: segmentation, motion masks

Architecture:
1. DINOv2 encoding per camera
2. Camera embeddings (learnable + extrinsics-based)
3. Cross-camera attention for multi-view reasoning
4. Temporal BlockRope decoder (from Pi3)
5. Autoregressive transformer for future prediction
6. Task-specific heads for points, confidence, egomotion
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
from .layers.cross_camera_attention import CrossCameraAttention
from .dinov2.hub.backbones import dinov2_vitl14_reg


class EgomotionHead(nn.Module):
    """
    Predicts vehicle egomotion (pose) from aggregated features.

    Output is a 4x4 transformation matrix representing the vehicle's
    pose in world frame for each timestep.
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim

        # Separate heads for rotation and translation
        self.rotation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),  # 6D rotation representation
        )

        self.translation_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),  # x, y, z
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict vehicle poses from features.

        Args:
            features: (B, T, D) aggregated features per timestep

        Returns:
            poses: (B, T, 4, 4) transformation matrices
        """
        B, T, D = features.shape

        # Predict rotation (6D representation)
        rot_6d = self.rotation_head(features)  # (B, T, 6)
        R = self._rotation_6d_to_matrix(rot_6d)  # (B, T, 3, 3)

        # Predict translation
        t = self.translation_head(features)  # (B, T, 3)

        # Compose transformation matrix
        poses = torch.eye(4, device=features.device, dtype=features.dtype)
        poses = poses.unsqueeze(0).unsqueeze(0).expand(B, T, 4, 4).clone()
        poses[:, :, :3, :3] = R
        poses[:, :, :3, 3] = t

        return poses

    def _rotation_6d_to_matrix(self, rot_6d: torch.Tensor) -> torch.Tensor:
        """
        Convert 6D rotation representation to rotation matrix.

        Uses the continuous 6D representation from Zhou et al.
        "On the Continuity of Rotation Representations in Neural Networks"

        Args:
            rot_6d: (..., 6) tensor

        Returns:
            R: (..., 3, 3) rotation matrices
        """
        a1 = rot_6d[..., :3]
        a2 = rot_6d[..., 3:]

        # Gram-Schmidt orthogonalization
        b1 = a1 / (torch.norm(a1, dim=-1, keepdim=True) + 1e-8)
        b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
        b2 = b2 / (torch.norm(b2, dim=-1, keepdim=True) + 1e-8)
        b3 = torch.cross(b1, b2, dim=-1)

        R = torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)
        return R


class MultiViewPi3(nn.Module):
    """
    Multi-View Pi3 Model for Multi-Camera 3D Perception.

    Takes synchronized images from multiple cameras and predicts 3D point
    maps and camera poses using cross-camera attention and temporal modeling.
    """

    def __init__(
        self,
        num_cameras: int = 5,
        n_current_frames: int = 3,
        n_future_frames: int = 3,
        pos_type: str = 'rope100',
        decoder_size: str = 'large',
        encoder_name: str = 'dinov2',
        # Cross-camera attention params
        cross_camera_layers: int = 6,
        cross_camera_heads: int = 16,
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

        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024

        # ----------------------
        #   Camera Embeddings
        # ----------------------
        if use_camera_embeddings:
            self.camera_embedding = CameraEmbedding(
                d_model=enc_embed_dim,
                max_cameras=7,
                use_extrinsics=embed_extrinsics,
            )
        else:
            self.camera_embedding = None

        # ----------------------
        #  Cross-Camera Attention
        # ----------------------
        self.cross_camera_attention = CrossCameraAttention(
            d_model=enc_embed_dim,
            n_heads=cross_camera_heads,
            n_layers=cross_camera_layers,
            dropout=0.0,
            mlp_ratio=4.0,
        )

        # ----------------------
        #  Positional Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #   Temporal Decoder
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
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #    Register Tokens
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
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
        # Point decoder
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

        # Confidence decoder
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(
            patch_size=self.patch_size,
            dec_embed_dim=1024,
            output_dim=1
        )

        # Camera pose decoder (per-camera)
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # Egomotion head (vehicle-level)
        self.egomotion_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.egomotion_head = EgomotionHead(input_dim=512)

        # Optional segmentation head
        if self.use_segmentation_head:
            self.segmentation_decoder = deepcopy(self.point_decoder)
            self.segmentation_head = LinearPts3d(
                patch_size=self.patch_size,
                dec_embed_dim=1024,
                output_dim=self.segmentation_num_classes,
            )

        # Optional motion head
        if self.use_motion_head:
            self.motion_decoder = deepcopy(self.point_decoder)
            self.motion_head = LinearPts3d(
                patch_size=self.patch_size,
                dec_embed_dim=1024,
                output_dim=1,
            )

        # Image normalization
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        if freeze_decoders:
            self._freeze_decoders()

    def _freeze_decoders(self):
        """Freeze point, conf, and camera decoders (not egomotion or segmentation)."""
        for param in self.point_decoder.parameters():
            param.requires_grad = False
        for param in self.conf_decoder.parameters():
            param.requires_grad = False
        for param in self.camera_decoder.parameters():
            param.requires_grad = False
        for param in self.point_head.parameters():
            param.requires_grad = False
        for param in self.conf_head.parameters():
            param.requires_grad = False
        for param in self.camera_head.parameters():
            param.requires_grad = False

        print("Froze point, conf, and camera decoders and heads")

    def decode(self, hidden, N, H, W):
        """Temporal BlockRope decoder (same as Pi3)."""
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []

        hidden = hidden.reshape(B * N, hw, -1)
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(
            B * N, *self.register_token.shape[-2:]
        )

        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(
                B * N, H // self.patch_size, W // self.patch_size, hidden.device
            )

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(
                B * N, self.patch_start_idx, 2
            ).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B * N, hw, -1)

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
                - camera_extrinsics: (B, C, 4, 4) camera extrinsics
                - camera_intrinsics: (B, C, 4) camera intrinsics [fx, fy, cx, cy]
            n_future_frames_override: Override for future frame count

        Returns:
            Dict with predictions for each camera
        """

        import ipdb; ipdb.set_trace()
        images = batch['images']  # (B, C, T, 3, H, W)
        camera_extrinsics = batch['camera_extrinsics']  # (B, C, 4, 4)

        B, C, T, _, H, W = images.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Normalize images
        images = (images - self.image_mean) / self.image_std

        # 1. Per-camera DINOv2 encoding
        # Reshape to (B*C*T, 3, H, W) for batch encoding
        imgs_flat = images.reshape(B * C * T, 3, H, W)
        hidden = self.encoder(imgs_flat, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        # hidden: (B*C*T, S, D) where S = patch_h * patch_w
        S = hidden.shape[1]
        D = hidden.shape[2]

        # Reshape to (B, C, T, S, D)
        hidden = hidden.reshape(B, C, T, S, D)

        # 2. Add camera embeddings
        if self.camera_embedding is not None:
            cam_embed = self.camera_embedding(C, camera_extrinsics, B)  # (B, C, 1, 1, D)
            hidden = hidden + cam_embed

        # 3. Cross-camera attention
        hidden = self.cross_camera_attention(hidden)  # (B, C, T, S, D)

        # 4. Process each camera through temporal decoder
        # Reshape to process per-camera: treat each camera as separate batch
        # (B*C, T, S, D) -> (B*C*T, S, D) for decode
        hidden_per_cam = hidden.reshape(B * C, T, S, D)
        hidden_per_cam = hidden_per_cam.reshape(B * C * T, S, D)

        # Temporal decode
        hidden_decoded, pos = self.decode(hidden_per_cam, T, H, W)
        # hidden_decoded: (B*C*T, S+register, 2*D)

        # 5. Autoregressive future prediction
        n_future = n_future_frames_override if n_future_frames_override is not None else self.n_future_frames
        all_hidden, all_pos = self.autoregressive_transformer(
            hidden_decoded, T, pos, n_future_frames_override=n_future
        )
        total_frames = T + n_future

        # all_hidden: (B*C*total_frames, S, 2*D)

        # 6. Per-camera task predictions
        point_hidden = self.point_decoder(all_hidden, xpos=all_pos)
        conf_hidden = self.conf_decoder(all_hidden, xpos=all_pos)
        camera_hidden = self.camera_decoder(all_hidden, xpos=all_pos)

        if self.use_segmentation_head:
            segmentation_hidden = self.segmentation_decoder(all_hidden, xpos=all_pos)

        if self.use_motion_head:
            motion_hidden = self.motion_decoder(all_hidden, xpos=all_pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Points - per camera
            point_hidden = point_hidden.float()
            local_points_flat = self.point_head(
                [point_hidden[:, self.patch_start_idx:]], (H, W)
            )
            # Reshape to (B, C, total_frames, H, W, 3)
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

            # Unproject local points using camera poses
            points = torch.einsum(
                'bcnij, bcnhwj -> bcnhwi',
                camera_poses,
                homogenize_points(local_points)
            )[..., :3]

            # Segmentation
            segmentation = None
            if self.use_segmentation_head:
                segmentation_hidden = segmentation_hidden.float()
                segmentation_flat = self.segmentation_head(
                    [segmentation_hidden[:, self.patch_start_idx:]], (H, W)
                )
                segmentation = segmentation_flat.reshape(B, C, total_frames, H, W, -1)

            # Motion
            motion = None
            if self.use_motion_head:
                motion_hidden = motion_hidden.float()
                motion_flat = self.motion_head(
                    [motion_hidden[:, self.patch_start_idx:]], (H, W)
                )
                motion = motion_flat.reshape(B, C, total_frames, H, W, -1)

        # 7. Egomotion prediction (aggregate across cameras)
        # Use mean pooling across cameras and spatial dims
        ego_features = all_hidden.reshape(B, C, total_frames, -1, all_hidden.shape[-1])
        ego_features = ego_features.mean(dim=1)  # (B, total_frames, S, D)
        ego_features = ego_features.mean(dim=2)  # (B, total_frames, D)

        # Project down and predict egomotion
        ego_hidden = self.egomotion_decoder(
            all_hidden.reshape(B * C, total_frames, -1, all_hidden.shape[-1]).mean(dim=1),
            xpos=None
        )
        # Aggregate across spatial for egomotion
        ego_pooled = ego_hidden.reshape(B, C, -1, ego_hidden.shape[-1]).mean(dim=[1, 2])
        ego_pooled = ego_pooled.unsqueeze(1).expand(B, total_frames, -1)
        vehicle_poses = self.egomotion_head(ego_pooled)  # (B, total_frames, 4, 4)

        # 8. Compute camera world poses from egomotion + extrinsics
        # camera_world_pose = egomotion @ camera_extrinsic
        camera_world_poses = torch.einsum(
            'btij, bcjk -> bctik',
            vehicle_poses,
            camera_extrinsics
        )  # (B, C, total_frames, 4, 4)

        result = {
            'points': points,                          # (B, C, T+M, H, W, 3)
            'local_points': local_points,              # (B, C, T+M, H, W, 3)
            'conf': conf,                              # (B, C, T+M, H, W, 1)
            'camera_poses': camera_poses,              # (B, C, T+M, 4, 4)
            'vehicle_poses': vehicle_poses,            # (B, T+M, 4, 4)
            'camera_world_poses': camera_world_poses,  # (B, C, T+M, 4, 4)
            'n_current_frames': T,
            'n_future_frames': n_future,
        }

        if self.use_segmentation_head and segmentation is not None:
            result['segmentation'] = segmentation  # (B, C, T+M, H, W, num_classes)

        if self.use_motion_head and motion is not None:
            result['motion'] = motion  # (B, C, T+M, H, W, 1)

        return result
