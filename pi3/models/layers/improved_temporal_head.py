import torch
import torch.nn as nn
import torch.nn.functional as F


class ImprovedFutureLinearPts3d(nn.Module):
    """ 
    Improved version of FutureLinearPts3d with better temporal aggregation.
    Drop-in replacement that enhances the temporal MLP approach.
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3, extra_tokens=3, 
                 use_cross_attention=False, cross_attention_layers=2):
        super().__init__()
        self.patch_size = patch_size
        self.extra_tokens = extra_tokens
        self.use_cross_attention = use_cross_attention
        
        # Current frame projection (same as original)
        self.current_proj = nn.Linear(dec_embed_dim, output_dim * self.patch_size**2)
        
        # Improved temporal aggregation
        # 1. Per-frame importance weights
        self.frame_importance = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim // 4),
            nn.ReLU(),
            nn.Linear(dec_embed_dim // 4, 1)
        )
        
        # 2. Advanced temporal projection with frame-specific generation
        # Instead of just expanding features, generate distinct future frames
        
        # Temporal dynamics encoder - captures motion/change patterns
        self.temporal_dynamics = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim),
            nn.LayerNorm(dec_embed_dim),
            nn.GELU(),
            nn.Linear(dec_embed_dim, dec_embed_dim // 2)
        )
        
        # Frame-specific generators - each future frame gets its own pathway
        self.frame_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dec_embed_dim + dec_embed_dim // 2 + 16, dec_embed_dim * 2),  # +16 for time encoding
                nn.LayerNorm(dec_embed_dim * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(dec_embed_dim * 2, dec_embed_dim),
                nn.GELU(),
                nn.Linear(dec_embed_dim, dec_embed_dim)
            ) for _ in range(extra_tokens)
        ])
        
        # Learnable time embeddings for each future frame
        self.future_time_embeds = nn.Parameter(torch.randn(extra_tokens, 16) * 0.02)
        
        # Motion extrapolation - predicts how scene changes over time
        self.motion_predictor = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim // 2),
            nn.GELU(),
            nn.Linear(dec_embed_dim // 2, dec_embed_dim // 4),
            nn.GELU(),
            nn.Linear(dec_embed_dim // 4, dec_embed_dim // 2)
        )
        
        # 3. Learnable temporal position bias
        self.temporal_bias = nn.Parameter(torch.zeros(1, 10, 1))  # Support up to 10 frames
        
        # 4. Lightweight cross-attention from current to future tokens
        if use_cross_attention:
            # Single lightweight attention layer
            self.query_proj = nn.Linear(dec_embed_dim, dec_embed_dim // 4)  # Reduce dimension
            self.key_proj = nn.Linear(dec_embed_dim, dec_embed_dim // 4)
            self.value_proj = nn.Linear(dec_embed_dim, dec_embed_dim // 4)
            self.out_proj = nn.Linear(dec_embed_dim // 4, dec_embed_dim)
            self.attn_scale = (dec_embed_dim // 4) ** -0.5
        
        # Enhanced future projection with more capacity
        self.future_proj = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim * 2),
            nn.LayerNorm(dec_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dec_embed_dim * 2, dec_embed_dim),
            nn.GELU(),
            nn.Linear(dec_embed_dim, output_dim * self.patch_size**2)
        )

    def forward(self, decout, img_shape, batch_size, num_current_frames):
        H, W = img_shape
        tokens = decout[-1]  # [B*N, S, D] where N is current frames  
        BN, S, D = tokens.shape
        B, N = batch_size, num_current_frames
        M = self.extra_tokens
        
        # Current frame predictions - directly use input tokens (same as original)
        current_feat = self.current_proj(tokens)  # [B*N, S, output_dim*patch_size^2]
        current_feat = current_feat.transpose(-1, -2).view(BN, -1, H//self.patch_size, W//self.patch_size)
        current_feat = F.pixel_shuffle(current_feat, self.patch_size)  # [B*N, output_dim, H, W]
        current_points = current_feat.permute(0, 2, 3, 1)  # [B*N, H, W, output_dim]
        
        # Improved future frame predictions
        # Reshape [B*N, S, D] -> [B, N, S, D] for proper temporal processing
        tokens_batched = tokens.view(B, N, S, D)
        
        # Compute importance weights for each frame
        importance_scores = self.frame_importance(tokens_batched)  # [B, N, S, 1]
        
        # Add temporal position bias (later frames might be more important)
        temporal_positions = self.temporal_bias[:, :N, :].unsqueeze(2)  # [1, N, 1, 1]
        importance_scores = importance_scores + temporal_positions
        
        # Normalize importance weights across frames
        importance_weights = F.softmax(importance_scores, dim=1)  # [B, N, S, 1]
        
        # Weighted aggregation instead of simple mean
        global_context = (tokens_batched * importance_weights).sum(dim=1)  # [B, S, D]
        
        # Generate M future frame predictions using frame-specific generators
        # Encode temporal dynamics from global context
        temporal_dynamics = self.temporal_dynamics(global_context)  # [B, S, D//2]
        
        # Generate each future frame with its specific generator
        future_frame_tokens = []
        for frame_idx in range(M):
            # Get time embedding for this frame
            time_embed = self.future_time_embeds[frame_idx].unsqueeze(0).unsqueeze(0)  # [1, 1, 16]
            time_embed = time_embed.repeat(B, S, 1)  # [B, S, 16]
            
            # Concatenate context, dynamics, and time embedding
            frame_input = torch.cat([
                global_context,      # [B, S, D]
                temporal_dynamics,   # [B, S, D//2]
                time_embed          # [B, S, 16]
            ], dim=-1)  # [B, S, D + D//2 + 16]
            
            # Generate frame-specific features
            frame_feat = self.frame_generators[frame_idx](frame_input)  # [B, S, D]
            
            # Add motion prediction for temporal consistency
            motion_delta = self.motion_predictor(global_context)  # [B, S, D//2]
            # Scale motion by frame distance (further frames have more motion)
            motion_scale = (frame_idx + 1) / M
            motion_delta = motion_delta * motion_scale
            
            # Add motion to base features (broadcast motion_delta to match dimensions)
            motion_expanded = torch.cat([motion_delta, motion_delta], dim=-1)[:, :, :D]  # [B, S, D]
            frame_feat = frame_feat + motion_expanded
            
            future_frame_tokens.append(frame_feat)
        
        # Stack future frames: [B, M, S, D]
        temporal_feat = torch.stack(future_frame_tokens, dim=1)
        temporal_feat = temporal_feat.permute(0, 2, 1, 3).contiguous()  # [B, S, M, D] -> [B, M, S, D]
        temporal_feat = temporal_feat.view(B * M, S, D)  # [B*M, S, D]
        
        # Project to final output
        future_feat = self.future_proj(temporal_feat)  # [B*M, S, output_dim*patch_size^2]
        future_feat = future_feat.transpose(-1, -2).view(B * M, -1, H//self.patch_size, W//self.patch_size)
        future_feat = F.pixel_shuffle(future_feat, self.patch_size)  # [B*M, output_dim, H, W]
        future_points = future_feat.permute(0, 2, 3, 1)  # [B*M, H, W, output_dim]
        
        # Concatenate: [B*N, H, W, output_dim] + [B*M, H, W, output_dim] = [B*(N+M), H, W, output_dim]
        all_points = torch.cat([current_points, future_points], dim=0)
        
        return all_points


class AttentionFutureLinearPts3d(nn.Module):
    """ 
    FutureLinearPts3d with lightweight attention-based temporal aggregation.
    More sophisticated than weighted average but still efficient.
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3, extra_tokens=3):
        super().__init__()
        self.patch_size = patch_size
        self.extra_tokens = extra_tokens
        
        # Current frame projection
        self.current_proj = nn.Linear(dec_embed_dim, output_dim * self.patch_size**2)
        
        # Lightweight temporal attention
        # Query: learnable aggregation query
        self.temporal_query = nn.Parameter(torch.randn(1, 1, dec_embed_dim))
        
        # Key and Value projections (shared to save parameters)
        self.kv_proj = nn.Linear(dec_embed_dim, dec_embed_dim * 2)
        
        # Number of attention heads
        self.n_heads = 8
        self.head_dim = dec_embed_dim // self.n_heads
        
        # Output projection after attention
        self.attn_out_proj = nn.Linear(dec_embed_dim, dec_embed_dim)
        
        # Temporal MLP (same structure as improved version)
        self.temporal_proj = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim * 2),
            nn.LayerNorm(dec_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dec_embed_dim * 2, dec_embed_dim * extra_tokens)
        )
        
        # Enhanced future projection
        self.future_proj = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim * 2),
            nn.LayerNorm(dec_embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dec_embed_dim * 2, dec_embed_dim),
            nn.GELU(),
            nn.Linear(dec_embed_dim, output_dim * self.patch_size**2)
        )

    def forward(self, decout, img_shape, batch_size, num_current_frames):
        H, W = img_shape
        tokens = decout[-1]  # [B*N, S, D]
        BN, S, D = tokens.shape
        B, N = batch_size, num_current_frames
        M = self.extra_tokens
        
        # Current frame predictions
        current_feat = self.current_proj(tokens)
        current_feat = current_feat.transpose(-1, -2).view(BN, -1, H//self.patch_size, W//self.patch_size)
        current_feat = F.pixel_shuffle(current_feat, self.patch_size)
        current_points = current_feat.permute(0, 2, 3, 1)
        
        # Temporal attention aggregation
        tokens_batched = tokens.view(B, N, S, D)
        
        # Expand query for each batch and spatial location
        query = self.temporal_query.expand(B, S, D)  # [B, S, D]
        
        # Process each spatial location independently
        aggregated_features = []
        for s in range(S):
            # Get tokens for this spatial location across all frames
            spatial_tokens = tokens_batched[:, :, s, :]  # [B, N, D]
            
            # Compute key and value
            kv = self.kv_proj(spatial_tokens)  # [B, N, 2*D]
            k, v = kv.chunk(2, dim=-1)  # Each [B, N, D]
            
            # Reshape for multi-head attention
            q = query[:, s:s+1, :].view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
            k = k.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            v = v.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
            
            # Compute attention
            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            attn_weights = F.softmax(scores, dim=-1)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, v)  # [B, n_heads, 1, head_dim]
            attn_output = attn_output.transpose(1, 2).reshape(B, 1, D)
            
            aggregated_features.append(attn_output)
        
        # Combine all spatial locations
        global_context = torch.cat(aggregated_features, dim=1)  # [B, S, D]
        global_context = self.attn_out_proj(global_context)
        
        # Generate future frames
        temporal_feat = self.temporal_proj(global_context)
        temporal_feat = temporal_feat.view(B, S, M, D)
        temporal_feat = temporal_feat.permute(0, 2, 1, 3).contiguous()
        temporal_feat = temporal_feat.view(B * M, S, D)
        
        future_feat = self.future_proj(temporal_feat)
        future_feat = future_feat.transpose(-1, -2).view(B * M, -1, H//self.patch_size, W//self.patch_size)
        future_feat = F.pixel_shuffle(future_feat, self.patch_size)
        future_points = future_feat.permute(0, 2, 3, 1)
        
        all_points = torch.cat([current_points, future_points], dim=0)
        return all_points