from .attention import FlashAttentionRope
from .block import BlockRope
from ..dinov2.layers import Mlp

import torch
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
import math
   
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


class PositionalEncoding(nn.Module):
    """Temporal positional encoding for sequence modeling."""
    
    def __init__(self, d_model, max_len=10):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x: [B, T, S, D]
        B, T, S, D = x.shape
        # Add temporal position encoding
        return x + self.pe[:T].unsqueeze(0).unsqueeze(2)  # [1, T, 1, D]


class CausalMultiHeadAttention(nn.Module):
    """Causal multi-head attention for autoregressive generation."""
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Register causal mask buffer
        self.register_buffer("causal_mask", torch.triu(torch.ones(10, 10), diagonal=1).bool())
        
    def forward(self, x):
        # x: [B, T, S, D]
        B, T, S, D = x.shape
        
        # Reshape to treat spatial locations independently
        x = x.view(B * S, T, D)  # [B*S, T, D]
        
        # Generate Q, K, V
        qkv = self.qkv_proj(x)  # [B*S, T, 3*D]
        qkv = qkv.reshape(B * S, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B*S, n_heads, T, head_dim]
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.causal_mask[:T, :T]
        scores.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)  # [B*S, n_heads, T, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(B * S, T, D)
        
        # Output projection
        output = self.out_proj(attn_output)
        
        # Reshape back to [B, T, S, D]
        output = output.view(B, T, S, D)
        
        return output


class TransformerBlock(nn.Module):
    """Transformer block with causal self-attention."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = CausalMultiHeadAttention(d_model, n_heads, dropout)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.self_attn(self.norm1(x))
        x = x + attn_output
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        
        return x


class AutoregressiveTokenGenerator(nn.Module):
    """
    Autoregressive transformer that generates future tokens.
    
    Input: [B, 3, 777, 1024] - 3 prior frames with 777 spatial tokens each
    Output: [B, 6, 777, 1024] - 3 prior + 3 future frames
    """
    
    def __init__(self, d_model=1024, n_heads=16, n_layers=6, d_ff=4096, 
                 dropout=0.1, max_future_frames=3):
        super().__init__()
        self.d_model = d_model
        self.max_future_frames = max_future_frames
        
        # Positional encoding for temporal dimension
        self.pos_encoding = PositionalEncoding(d_model, max_len=10)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Token generation head
        self.token_generator = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, prior_tokens, num_future_frames=None):
        """
        Generate future tokens autoregressively.
        
        Args:
            prior_tokens: [B, 3, 777, 1024] - prior frame tokens
            num_future_frames: int - number of future frames to generate (default: 3)
            
        Returns:
            all_tokens: [B, 3+num_future_frames, 777, 1024] - prior + future tokens
        """
        if num_future_frames is None:
            num_future_frames = self.max_future_frames
            
        B, T_prior, S, D = prior_tokens.shape
        device = prior_tokens.device
        
        # Start with prior tokens
        all_tokens = prior_tokens  # [B, 3, 777, 1024]
        
        # Generate future tokens one by one
        for t in range(num_future_frames):
            # Current sequence length
            T_current = T_prior + t
            
            # Add positional encoding to current sequence
            tokens_with_pos = self.pos_encoding(all_tokens)  # [B, T_current, 777, 1024]
            
            # Pass through transformer layers
            hidden = tokens_with_pos
            for layer in self.transformer_layers:
                hidden = layer(hidden)
            
            # Generate next token from the last time step
            last_hidden = hidden[:, -1]  # [B, 777, 1024] - most recent frame
            next_token = self.token_generator(last_hidden)  # [B, 777, 1024]
            
            # Add to sequence
            next_token = next_token.unsqueeze(1)  # [B, 1, 777, 1024]
            all_tokens = torch.cat([all_tokens, next_token], dim=1)  # [B, T_current+1, 777, 1024]
        
        return all_tokens
    
    def forward_training(self, sequence_tokens):
        """
        Training forward pass with teacher forcing.
        
        Args:
            sequence_tokens: [B, 6, 777, 1024] - full sequence (3 prior + 3 future)
            
        Returns:
            predictions: [B, 6, 777, 1024] - predicted sequence
            targets: [B, 3, 777, 1024] - target future tokens
        """
        B, T_total, S, D = sequence_tokens.shape
        T_prior = 3
        
        # Add positional encoding
        tokens_with_pos = self.pos_encoding(sequence_tokens)
        
        # Pass through transformer layers
        hidden = tokens_with_pos
        for layer in self.transformer_layers:
            hidden = layer(hidden)
        
        # Generate predictions for all positions
        predictions = self.token_generator(hidden)
        
        # Extract targets (future tokens)
        targets = sequence_tokens[:, T_prior:]  # [B, 3, 777, 1024]
        pred_future = predictions[:, T_prior:]   # [B, 3, 777, 1024]
        
        return pred_future, targets


class AutoregressiveFuturePts3d(nn.Module):
    """ 
    Autoregressive transformer for future frame prediction.
    Complete replacement for FutureLinearPts3d using autoregressive approach.
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3, extra_tokens=3,
                 n_heads=16, n_layers=6, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.extra_tokens = extra_tokens
        self.output_dim = output_dim
        
        # Autoregressive token generator
        self.token_generator = AutoregressiveTokenGenerator(
            d_model=dec_embed_dim,
            n_heads=n_heads, 
            n_layers=n_layers,
            dropout=dropout,
            max_future_frames=extra_tokens
        )
        
        # Final projection to 3D points
        self.point_proj = nn.Linear(dec_embed_dim, output_dim * patch_size**2)
        
    def forward(self, decout, img_shape, batch_size, num_current_frames):
        """
        Drop-in replacement for FutureLinearPts3d.forward()
        
        Args:
            decout: List containing hidden tokens
            img_shape: (H, W) image dimensions  
            batch_size: B
            num_current_frames: N (should be 3)
            
        Returns:
            all_points: [B*(N+M), H, W, output_dim] - current + future points
        """
        H, W = img_shape
        tokens = decout[-1]  # [B*N, S, D] where N=3, S=777
        BN, S, D = tokens.shape
        B, N = batch_size, num_current_frames
        M = self.extra_tokens
        
        # Reshape to sequence format: [B, N, S, D]
        prior_tokens = tokens.view(B, N, S, D)
        
        if self.training:
            # During training, we need ground truth future tokens
            # For now, we'll just generate and return current tokens
            # The training loss will be handled separately
            
            # Generate future tokens
            all_tokens = self.token_generator(prior_tokens, num_future_frames=M)
            # all_tokens: [B, N+M, S, D]
        else:
            # Inference: generate future tokens autoregressively  
            all_tokens = self.token_generator(prior_tokens, num_future_frames=M)
            # all_tokens: [B, N+M, S, D]
        
        # Convert tokens to 3D points
        BNM, S, D = all_tokens.view(-1, S, D).shape
        
        # Project to points
        point_features = self.point_proj(all_tokens.view(BNM, S, D))  # [B*(N+M), S, output_dim*patch_size^2]
        
        # Reshape to spatial format
        point_features = point_features.transpose(-1, -2).view(BNM, -1, H//self.patch_size, W//self.patch_size)
        point_features = F.pixel_shuffle(point_features, self.patch_size)  # [B*(N+M), output_dim, H, W]
        all_points = point_features.permute(0, 2, 3, 1)  # [B*(N+M), H, W, output_dim]
        
        return all_points
    
    def training_forward(self, decout, img_shape, batch_size, num_current_frames, future_tokens):
        """
        Training forward pass with ground truth future tokens.
        
        Args:
            decout: List containing current hidden tokens  
            future_tokens: [B, M, S, D] - ground truth future tokens
            
        Returns:
            predicted_future: [B, M, S, D] - predicted future tokens
            target_future: [B, M, S, D] - target future tokens
        """
        tokens = decout[-1].view(batch_size, num_current_frames, -1, tokens.shape[-1])
        
        # Concatenate current and future for teacher forcing
        full_sequence = torch.cat([tokens, future_tokens], dim=1)  # [B, N+M, S, D]
        
        # Forward with teacher forcing
        predicted_future, target_future = self.token_generator.forward_training(full_sequence)
        
        return predicted_future, target_future


# Loss function for autoregressive token generation
def autoregressive_token_loss(predicted_tokens, target_tokens, loss_type='mse'):
    """
    Loss function for autoregressive token generation.
    
    Args:
        predicted_tokens: [B, M, S, D] - predicted future tokens
        target_tokens: [B, M, S, D] - target future tokens  
        loss_type: 'mse' or 'cosine' or 'combined'
        
    Returns:
        loss: scalar loss value
    """
    if loss_type == 'mse':
        return F.mse_loss(predicted_tokens, target_tokens)
    elif loss_type == 'cosine':
        # Cosine similarity loss in embedding space
        pred_norm = F.normalize(predicted_tokens, dim=-1)
        target_norm = F.normalize(target_tokens, dim=-1) 
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)
        return (1 - cosine_sim).mean()
    elif loss_type == 'combined':
        mse_loss = F.mse_loss(predicted_tokens, target_tokens)
        
        pred_norm = F.normalize(predicted_tokens, dim=-1)
        target_norm = F.normalize(target_tokens, dim=-1)
        cosine_sim = (pred_norm * target_norm).sum(dim=-1)
        cosine_loss = (1 - cosine_sim).mean()
        
        return mse_loss + 0.1 * cosine_loss
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")