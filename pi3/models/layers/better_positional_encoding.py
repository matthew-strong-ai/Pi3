import torch
import torch.nn as nn
import math


class SpatioTemporalPositionalEncoding(nn.Module):
    """
    Learnable spatio-temporal positional encoding.
    Adds both spatial and temporal position information.
    """
    
    def __init__(self, d_model, max_temporal_len=10, max_spatial_len=64*64):
        super().__init__()
        self.d_model = d_model
        self.max_temporal_len = max_temporal_len
        
        # Split embedding dimensions between temporal and spatial
        self.temporal_dim = d_model // 2
        self.spatial_dim = d_model - self.temporal_dim
        
        # Learnable temporal embeddings
        self.temporal_embed = nn.Parameter(
            torch.zeros(1, max_temporal_len, 1, self.temporal_dim)
        )
        
        # Learnable spatial embeddings (using a smaller embedding table)
        # We'll use 2D spatial embeddings that can be interpolated
        spatial_size = int(math.sqrt(max_spatial_len))
        self.spatial_embed = nn.Parameter(
            torch.zeros(1, 1, spatial_size, spatial_size, self.spatial_dim)
        )
        
        # Initialize
        nn.init.normal_(self.temporal_embed, std=0.02)
        nn.init.normal_(self.spatial_embed, std=0.02)
        
    def forward(self, x, start_idx=0, spatial_shape=None):
        """
        Args:
            x: [B, T, S, D] - batch, temporal, spatial, feature
            start_idx: starting temporal position
            spatial_shape: Optional (H, W) tuple. If not provided, assumes closest square
        Returns:
            x + positional encoding
        """
        B, T, S, D = x.shape
        
        # Handle spatial dimensions
        if spatial_shape is not None:
            H, W = spatial_shape
            assert H * W == S, f"Spatial shape {H}x{W}={H*W} doesn't match S={S}"
        else:
            # For non-square, find closest factors
            # Common case: 782 = 23 * 34 (from 37x37 patches with some removed)
            # Just use 1D spatial encoding for simplicity
            H = S
            W = 1
        
        assert D == self.d_model, f"Feature dim mismatch: got {D}, expected {self.d_model}"
        
        # Get temporal embeddings
        temporal_pos = self.temporal_embed[:, start_idx:start_idx+T, :, :]  # [1, T, 1, temporal_dim]
        temporal_pos = temporal_pos.expand(B, -1, S, -1)  # [B, T, S, temporal_dim]
        
        # Get spatial embeddings and interpolate to match current resolution
        spatial_pos = self.spatial_embed  # [1, 1, H_embed, W_embed, spatial_dim]
        if spatial_pos.shape[2] != H or spatial_pos.shape[3] != W:
            # Interpolate spatial embeddings to current resolution
            spatial_pos = spatial_pos.permute(0, 1, 4, 2, 3)  # [1, 1, spatial_dim, H, W]
            spatial_pos = nn.functional.interpolate(
                spatial_pos.squeeze(1), 
                size=(H, W), 
                mode='bilinear', 
                align_corners=False
            )  # [1, spatial_dim, H, W]
            spatial_pos = spatial_pos.unsqueeze(1).permute(0, 1, 3, 4, 2)  # [1, 1, H, W, spatial_dim]
        
        # Reshape spatial embeddings to [B, T, S, spatial_dim]
        spatial_pos = spatial_pos.view(1, 1, S, self.spatial_dim)
        spatial_pos = spatial_pos.expand(B, T, -1, -1)  # [B, T, S, spatial_dim]
        
        # Concatenate temporal and spatial embeddings
        pos_embed = torch.cat([temporal_pos, spatial_pos], dim=-1)  # [B, T, S, D]
        
        return x + pos_embed


class SinusoidalSpatioTemporalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for both spatial and temporal dimensions.
    More generalizable than learned embeddings.
    """
    
    def __init__(self, d_model, max_temporal_len=10000):
        super().__init__()
        self.d_model = d_model
        
        # Split dimensions: 1/3 for temporal, 2/3 for spatial (2D)
        self.temporal_dim = d_model // 3
        self.spatial_dim = d_model - self.temporal_dim
        self.spatial_x_dim = self.spatial_dim // 2
        self.spatial_y_dim = self.spatial_dim - self.spatial_x_dim
        
    def forward(self, x, start_idx=0):
        """
        Args:
            x: [B, T, S, D]
        Returns:
            x + positional encoding
        """
        B, T, S, D = x.shape
        H = W = int(math.sqrt(S))
        device = x.device
        
        # Create temporal encoding
        temporal_pos = torch.arange(start_idx, start_idx + T, device=device).float()
        temporal_encoding = self._sinusoidal_encoding(
            temporal_pos, self.temporal_dim
        ).unsqueeze(1).expand(-1, S, -1)  # [T, S, temporal_dim]
        
        # Create spatial encoding
        y_pos = torch.arange(H, device=device).float()
        x_pos = torch.arange(W, device=device).float()
        
        # Create meshgrid
        y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
        y_grid = y_grid.reshape(-1)  # [S]
        x_grid = x_grid.reshape(-1)  # [S]
        
        y_encoding = self._sinusoidal_encoding(y_grid, self.spatial_y_dim)  # [S, spatial_y_dim]
        x_encoding = self._sinusoidal_encoding(x_grid, self.spatial_x_dim)  # [S, spatial_x_dim]
        spatial_encoding = torch.cat([y_encoding, x_encoding], dim=-1)  # [S, spatial_dim]
        spatial_encoding = spatial_encoding.unsqueeze(0).expand(T, -1, -1)  # [T, S, spatial_dim]
        
        # Combine encodings
        pos_encoding = torch.cat([temporal_encoding, spatial_encoding], dim=-1)  # [T, S, D]
        pos_encoding = pos_encoding.unsqueeze(0).expand(B, -1, -1, -1)  # [B, T, S, D]
        
        return x + pos_encoding
    
    def _sinusoidal_encoding(self, positions, d_model):
        """Generate sinusoidal encoding for given positions."""
        pe = torch.zeros(positions.shape[0], d_model, device=positions.device)
        
        div_term = torch.exp(torch.arange(0, d_model, 2, device=positions.device).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(positions.unsqueeze(1) * div_term)
        pe[:, 1::2] = torch.cos(positions.unsqueeze(1) * div_term)
        
        return pe