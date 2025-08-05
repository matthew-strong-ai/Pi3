from .attention import FlashAttentionRope
from .block import BlockRope
from ..dinov2.layers import Mlp

import torch
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
   
class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
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
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out

class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape

        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat.permute(0, 2, 3, 1)
    

class FutureLinearPts3d (nn.Module):
    """ 
    Linear head for dust3r, with current + future frames.
    Outputs: (B*(N+M), H, W, output_dim) where N=current, M=future frames
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3, extra_tokens=3):
        super().__init__()
        self.patch_size = patch_size
        self.extra_tokens = extra_tokens
        
        # Current frame projection
        self.current_proj = nn.Linear(dec_embed_dim, output_dim * self.patch_size**2)
        
        # Future frame prediction using temporal MLP
        self.temporal_proj = nn.Sequential(
            nn.Linear(dec_embed_dim, dec_embed_dim),
            nn.ReLU(),
            nn.Linear(dec_embed_dim, dec_embed_dim * extra_tokens)
        )
        self.future_proj = nn.Linear(dec_embed_dim, output_dim * self.patch_size**2)

    def forward(self, decout, img_shape, batch_size, num_current_frames):
        H, W = img_shape
        tokens = decout[-1]  # [B*N, S, D] where N is current frames  
        BN, S, D = tokens.shape
        B, N = batch_size, num_current_frames
        M = self.extra_tokens
        
        # Current frame predictions - directly use input tokens
        current_feat = self.current_proj(tokens)  # [B*N, S, output_dim*patch_size^2]
        current_feat = current_feat.transpose(-1, -2).view(BN, -1, H//self.patch_size, W//self.patch_size)
        current_feat = F.pixel_shuffle(current_feat, self.patch_size)  # [B*N, output_dim, H, W]
        current_points = current_feat.permute(0, 2, 3, 1)  # [B*N, H, W, output_dim]
        
        # Future frame predictions - proper batching
        # Reshape [B*N, S, D] -> [B, N, S, D] for proper pooling within batches
        tokens_batched = tokens.view(B, N, S, D)
        
        # Pool information within each batch (mean across N current frames)
        global_context = tokens_batched.mean(dim=1)  # [B, S, D] - per-batch global context
        
        # Generate M future frame predictions for each batch
        temporal_feat = self.temporal_proj(global_context)  # [B, S, D*M]
        temporal_feat = temporal_feat.view(B, S, M, D)  # [B, S, M, D]
        temporal_feat = temporal_feat.permute(0, 2, 1, 3).contiguous()  # [B, M, S, D]
        temporal_feat = temporal_feat.view(B * M, S, D)  # [B*M, S, D]
        
        future_feat = self.future_proj(temporal_feat)  # [B*M, S, output_dim*patch_size^2]
        future_feat = future_feat.transpose(-1, -2).view(B * M, -1, H//self.patch_size, W//self.patch_size)
        future_feat = F.pixel_shuffle(future_feat, self.patch_size)  # [B*M, output_dim, H, W]
        future_points = future_feat.permute(0, 2, 3, 1)  # [B*M, H, W, output_dim]
        
        # Concatenate: [B*N, H, W, output_dim] + [B*M, H, W, output_dim] = [B*(N+M), H, W, output_dim]
        all_points = torch.cat([current_points, future_points], dim=0)
        
        return all_points
    

# BatchTokenAdder class removed - replaced with proper temporal modeling