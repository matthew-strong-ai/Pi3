"""
Camera Embedding Module for Multi-View Pi3.

Provides learnable camera embeddings that can be added to per-camera features
to help the model distinguish between different camera viewpoints.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class CameraEmbedding(nn.Module):
    """
    Learnable camera embeddings for multi-view models.

    Provides two types of embeddings:
    1. Learnable camera ID embeddings (one per camera position)
    2. Optional extrinsics-based embeddings (from camera pose)

    These embeddings are added to per-camera features to help the model
    understand spatial relationships between cameras.
    """

    def __init__(
        self,
        d_model: int = 1024,
        max_cameras: int = 7,
        use_extrinsics: bool = True,
        extrinsics_dim: int = 7,  # quat (4) + position (3)
    ):
        """
        Args:
            d_model: Model hidden dimension
            max_cameras: Maximum number of cameras supported
            use_extrinsics: Whether to use extrinsics-based embeddings
            extrinsics_dim: Dimension of extrinsics input (7 = quat + pos)
        """
        super().__init__()

        self.d_model = d_model
        self.max_cameras = max_cameras
        self.use_extrinsics = use_extrinsics

        # Learnable camera ID embeddings
        self.camera_id_embed = nn.Parameter(
            torch.randn(max_cameras, d_model) * 0.02
        )

        # Optional extrinsics projection
        if use_extrinsics:
            self.extrinsics_proj = nn.Sequential(
                nn.Linear(extrinsics_dim, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, d_model),
            )

    def forward(
        self,
        num_cameras: int,
        extrinsics: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """
        Get camera embeddings.

        Args:
            num_cameras: Number of cameras to get embeddings for
            extrinsics: Optional (B, C, 4, 4) camera extrinsic matrices
            batch_size: Batch size (used when extrinsics not provided)

        Returns:
            embeddings: (B, C, 1, 1, D) camera embeddings ready to broadcast
                to per-camera feature tensors of shape (B, C, T, S, D)
        """
        # Get learnable camera ID embeddings
        camera_ids = self.camera_id_embed[:num_cameras]  # (C, D)

        if self.use_extrinsics and extrinsics is not None:
            # Extract position and quaternion from extrinsics
            B, C = extrinsics.shape[:2]
            extrinsics_features = self._extract_extrinsics_features(extrinsics)
            extrinsics_embed = self.extrinsics_proj(extrinsics_features)  # (B, C, D)

            # Combine with camera ID embeddings
            embeddings = camera_ids.unsqueeze(0) + extrinsics_embed  # (B, C, D)
        else:
            # Just use camera ID embeddings, expand for batch
            B = batch_size if extrinsics is None else extrinsics.shape[0]
            embeddings = camera_ids.unsqueeze(0).expand(B, -1, -1)  # (B, C, D)

        # Reshape for broadcasting: (B, C, 1, 1, D)
        return embeddings.unsqueeze(2).unsqueeze(3)

    def _extract_extrinsics_features(self, extrinsics: torch.Tensor) -> torch.Tensor:
        """
        Extract compact features from extrinsics matrices.

        Args:
            extrinsics: (B, C, 4, 4) transformation matrices

        Returns:
            features: (B, C, 7) [position (3) + quaternion (4)]
        """
        B, C = extrinsics.shape[:2]

        # Extract position (translation)
        position = extrinsics[:, :, :3, 3]  # (B, C, 3)

        # Extract rotation and convert to quaternion
        rotation = extrinsics[:, :, :3, :3]  # (B, C, 3, 3)
        quaternion = self._rotation_matrix_to_quaternion(rotation)  # (B, C, 4)

        # Concatenate position and quaternion
        features = torch.cat([position, quaternion], dim=-1)  # (B, C, 7)

        return features

    def _rotation_matrix_to_quaternion(self, R: torch.Tensor) -> torch.Tensor:
        """
        Convert rotation matrices to quaternions.

        Uses a numerically stable method from "Accurate Computation of Quaternions
        from Rotation Matrices" by Soheil Sarabandi and Federico Thomas.

        Args:
            R: (..., 3, 3) rotation matrices

        Returns:
            q: (..., 4) quaternions [qx, qy, qz, qw]
        """
        batch_shape = R.shape[:-2]
        R = R.reshape(-1, 3, 3)

        # Method from scipy's Rotation class
        decision_matrix = torch.stack([
            R[:, 0, 0] - R[:, 1, 1] - R[:, 2, 2],
            R[:, 1, 1] - R[:, 0, 0] - R[:, 2, 2],
            R[:, 2, 2] - R[:, 0, 0] - R[:, 1, 1],
            R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2],
        ], dim=-1)

        choice = decision_matrix.argmax(dim=-1)
        quat = torch.zeros(R.shape[0], 4, device=R.device, dtype=R.dtype)

        # Case 0: qx largest
        mask = choice == 0
        if mask.any():
            i = 0
            j = 1
            k = 2
            quat[mask, 0] = 1 - decision_matrix[mask, 1] - decision_matrix[mask, 2] + decision_matrix[mask, 0]
            quat[mask, 1] = R[mask, j, i] + R[mask, i, j]
            quat[mask, 2] = R[mask, k, i] + R[mask, i, k]
            quat[mask, 3] = R[mask, k, j] - R[mask, j, k]

        # Case 1: qy largest
        mask = choice == 1
        if mask.any():
            i = 1
            j = 2
            k = 0
            quat[mask, 1] = 1 - decision_matrix[mask, 2] - decision_matrix[mask, 0] + decision_matrix[mask, 1]
            quat[mask, 2] = R[mask, j, i] + R[mask, i, j]
            quat[mask, 3] = R[mask, k, i] + R[mask, i, k]
            quat[mask, 0] = R[mask, k, j] - R[mask, j, k]

        # Case 2: qz largest
        mask = choice == 2
        if mask.any():
            i = 2
            j = 0
            k = 1
            quat[mask, 2] = 1 - decision_matrix[mask, 0] - decision_matrix[mask, 1] + decision_matrix[mask, 2]
            quat[mask, 3] = R[mask, j, i] + R[mask, i, j]
            quat[mask, 0] = R[mask, k, i] + R[mask, i, k]
            quat[mask, 1] = R[mask, k, j] - R[mask, j, k]

        # Case 3: qw largest
        mask = choice == 3
        if mask.any():
            quat[mask, 3] = 1 + decision_matrix[mask, 0] + decision_matrix[mask, 1] + decision_matrix[mask, 2]
            quat[mask, 0] = R[mask, 2, 1] - R[mask, 1, 2]
            quat[mask, 1] = R[mask, 0, 2] - R[mask, 2, 0]
            quat[mask, 2] = R[mask, 1, 0] - R[mask, 0, 1]

        # Normalize
        quat = quat / (torch.norm(quat, dim=-1, keepdim=True) + 1e-8)

        return quat.reshape(*batch_shape, 4)


class NormalizedPoseCameraEmbedding(nn.Module):
    """
    DriveVGGT-style normalized pose camera embedding.

    Uses a continuous, data-driven representation instead of discrete camera IDs:
    - Normalized translation (3D): mean-centered, std-normalized across cameras per sample
    - 6D rotation: first two columns of rotation matrix (continuous representation)
    - Normalized intrinsics (4D): [fx/W, fy/H, cx/W, cy/H]

    Total pose vector: 13D → MLP → d_model
    """

    def __init__(self, d_model: int = 1024):
        super().__init__()
        self.d_model = d_model

        # 13D input: 3 (translation) + 6 (rotation) + 4 (intrinsics)
        self.pose_mlp = nn.Sequential(
            nn.Linear(13, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model),
        )

        # Initialize final layer with small weights for safe start
        nn.init.normal_(self.pose_mlp[2].weight, std=0.02)
        nn.init.zeros_(self.pose_mlp[2].bias)

    def forward(
        self,
        extrinsics: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        image_size: tuple = (224, 224),
    ) -> torch.Tensor:
        """
        Compute normalized pose camera embeddings.

        Args:
            extrinsics: (B, C, 4, 4) camera extrinsic matrices
            intrinsics: (B, C, 4) camera intrinsics [fx, fy, cx, cy] or None
            image_size: (H, W) image dimensions for intrinsics normalization

        Returns:
            embeddings: (B, C, 1, 1, D) camera embeddings ready to broadcast
        """
        B, C = extrinsics.shape[:2]
        H, W = image_size

        # --- Translation (3D): normalize across cameras per sample ---
        translation = extrinsics[:, :, :3, 3]  # (B, C, 3)
        t_mean = translation.mean(dim=1, keepdim=True)  # (B, 1, 3)
        t_std = translation.std(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1, 3)
        t_normalized = (translation - t_mean) / t_std * 0.1  # (B, C, 3)
        t_normalized = t_normalized.clamp(-10, 10)

        # --- Rotation (6D): first two columns of rotation matrix ---
        rotation = extrinsics[:, :, :3, :3]  # (B, C, 3, 3)
        rot_6d = rotation[:, :, :3, :2].reshape(B, C, 6)  # (B, C, 6)

        # --- Intrinsics (4D): normalized by image dimensions ---
        if intrinsics is not None:
            # intrinsics: (B, C, 4) = [fx, fy, cx, cy]
            fx = intrinsics[:, :, 0:1] / W
            fy = intrinsics[:, :, 1:2] / H
            cx = intrinsics[:, :, 2:3] / W
            cy = intrinsics[:, :, 3:4] / H
            intrinsics_normalized = torch.cat([fx, fy, cx, cy], dim=-1)  # (B, C, 4)
        else:
            intrinsics_normalized = torch.zeros(B, C, 4, device=extrinsics.device, dtype=extrinsics.dtype)

        # --- Concatenate pose vector (13D) ---
        pose_vector = torch.cat([t_normalized, rot_6d, intrinsics_normalized], dim=-1)  # (B, C, 13)

        # --- MLP projection ---
        embeddings = self.pose_mlp(pose_vector)  # (B, C, D)

        # Reshape for broadcasting: (B, C, 1, 1, D)
        return embeddings.unsqueeze(2).unsqueeze(3)


class PositionalCameraEmbedding(nn.Module):
    """
    Alternative camera embedding using sinusoidal positional encoding
    based on camera position in 3D space.

    This provides a continuous embedding that depends on the actual
    camera position rather than discrete camera IDs.
    """

    def __init__(
        self,
        d_model: int = 1024,
        max_freq: float = 10000.0,
        num_freqs: int = 64,
    ):
        """
        Args:
            d_model: Model hidden dimension
            max_freq: Maximum frequency for positional encoding
            num_freqs: Number of frequency bands
        """
        super().__init__()

        self.d_model = d_model
        self.num_freqs = num_freqs

        # Sinusoidal frequencies for 3D position encoding
        # 3 dimensions (x, y, z) * 2 (sin, cos) * num_freqs = input dim
        input_dim = 3 * 2 * num_freqs

        # Project to model dimension
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Precompute frequency bands
        freqs = 2.0 ** torch.linspace(0, np.log2(max_freq), num_freqs)
        self.register_buffer('freqs', freqs)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute positional camera embeddings.

        Args:
            positions: (B, C, 3) camera positions in world/vehicle frame

        Returns:
            embeddings: (B, C, 1, 1, D) camera embeddings
        """
        B, C, _ = positions.shape

        # Apply sinusoidal encoding to each position dimension
        # positions: (B, C, 3)
        # freqs: (num_freqs,)

        pos_scaled = positions.unsqueeze(-1) * self.freqs  # (B, C, 3, num_freqs)
        pos_encoded = torch.cat([
            torch.sin(pos_scaled),
            torch.cos(pos_scaled),
        ], dim=-1)  # (B, C, 3, 2*num_freqs)

        pos_flat = pos_encoded.reshape(B, C, -1)  # (B, C, 3*2*num_freqs)

        # Project to model dimension
        embeddings = self.proj(pos_flat)  # (B, C, D)

        # Reshape for broadcasting
        return embeddings.unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1, D)
