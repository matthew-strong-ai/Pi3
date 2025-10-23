import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .simple_positional_encoding import SimpleLearnablePositionalEncoding, RoPE2DStyle


class TemporalPositionalEncoding(nn.Module):
    """Learnable temporal positional encoding for sequence modeling."""
    
    def __init__(self, d_model, max_len=10):
        super().__init__()
        self.max_len = max_len
        self.d_model = d_model
        # Learnable positional embeddings for temporal dimension
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, max_len, 1, d_model))
        nn.init.normal_(self.temporal_pos_embed, std=0.02)
        
    def forward(self, x, start_idx=0):
        """
        Args:
            x: [B, T, S, D] - temporal, spatial, feature dimensions
            start_idx: starting position for positional encoding
        """
        B, T, S, D = x.shape
        
        # Get positional embeddings for the temporal positions
        pos_embed = self.temporal_pos_embed[:, start_idx:start_idx+T, :, :]  # [1, T, 1, D]
        # Properly broadcast to all spatial locations
        pos_embed = pos_embed.expand(B, -1, S, -1)  # [B, T, S, D]
        return x + pos_embed


class TemporalCausalAttention(nn.Module):
    """
    Causal self-attention across temporal dimension.
    Each spatial token attends to previous temporal positions at the same spatial location.
    """
    
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: [B, T, S, D] - batch, temporal, spatial, feature
        Returns:
            output: [B, T, S, D]
        """
        B, T, S, D = x.shape
        
        # Process each spatial location independently
        # Reshape to [B*S, T, D] to handle temporal attention per spatial location
        x_reshaped = x.permute(0, 2, 1, 3).contiguous().view(B * S, T, D)
        
        # Compute QKV
        qkv = self.qkv(x_reshaped)  # [B*S, T, 3*D]
        qkv = qkv.view(B * S, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B*S, n_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B*S, n_heads, T, T]
        
        # Apply causal mask (upper triangular)
        if T > 1:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # [B*S, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B * S, T, D)
        
        # Output projection
        out = self.out_proj(out)
        out = self.dropout(out)
        
        # Reshape back to [B, T, S, D]
        out = out.view(B, S, T, D).permute(0, 2, 1, 3).contiguous()
        
        return out


class AutoregressiveTransformerBlock(nn.Module):
    """Single transformer block for autoregressive modeling."""
    
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Pre-norm architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TemporalCausalAttention(d_model, n_heads, dropout)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # x: [B, T, S, D]
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class AutoregressiveTokenTransformer(nn.Module):
    """
    Autoregressive transformer that takes scene tokens and generates future tokens.
    Designed to work after Pi3's decode() function.
    """
    
    def __init__(self,
                 d_model=2048,  # 2 * dec_embed_dim from Pi3
                 n_heads=16,
                 n_layers=8,
                 d_ff=2048,
                 dropout=0.1,
                 n_future_frames=3,
                 max_seq_len=10,
                 positional_encoding='spatiotemporal',  # 'temporal', 'spatiotemporal', or 'sinusoidal'
                 max_spatial_size=64*64):
        super().__init__()
        
        self.d_model = d_model
        self.n_future_frames = n_future_frames
        self.n_heads = n_heads
        
        # Choose positional encoding type
        if positional_encoding == 'temporal':
            self.temporal_pos_embed = TemporalPositionalEncoding(d_model, max_seq_len)
        elif positional_encoding == 'spatiotemporal':
            self.temporal_pos_embed = SimpleLearnablePositionalEncoding(
                d_model, max_seq_len, max_spatial_size
            )
        elif positional_encoding == 'rope':
            self.temporal_pos_embed = RoPE2DStyle(d_model, max_seq_len)
        else:
            raise ValueError(f"Unknown positional encoding: {positional_encoding}")
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AutoregressiveTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final norm
        self.norm = nn.LayerNorm(d_model)
        
        # No token predictor - use transformer output directly
        # The transformer blocks already transform the representation
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
    def reshape_to_temporal(self, hidden, N):
        """
        Reshape tokens from decode() to temporal format.
        
        Args:
            hidden: [B*N, S, D] - output from decode()
            N: number of current frames
            
        Returns:
            tokens: [B, N, S, D]
        """
        BN, S, D = hidden.shape
        B = BN // N
        return hidden.view(B, N, S, D)
    
    def reshape_to_spatial(self, tokens):
        """
        Reshape tokens back to spatial format for decoders.
        
        Args:
            tokens: [B, T, S, D]
            
        Returns:
            hidden: [B*T, S, D]
        """
        B, T, S, D = tokens.shape
        return tokens.view(B * T, S, D)
    
    def forward(self, hidden, N, pos=None):
        """
        Forward pass to generate future tokens autoregressively.
        
        Args:
            hidden: [B*N, S, D] - aggregated tokens from decode()
            N: number of current frames
            pos: [B*N, S, 2] - positional encoding from decode()
            
        Returns:
            all_hidden: [B*(N+M), S, D] - current + future tokens
            all_pos: [B*(N+M), S, 2] - positional encoding for all tokens
        """
        # Validate inputs
        BN, S, D = hidden.shape
        assert BN % N == 0, f"Batch size mismatch: {BN} not divisible by {N}"
        assert D == self.d_model, f"Feature dimension mismatch: got {D}, expected {self.d_model}"
        
        # Reshape to temporal sequence
        tokens = self.reshape_to_temporal(hidden, N)  # [B, N, S, D]
        B, _, S, D = tokens.shape
        
        # Start with current tokens
        all_tokens = tokens
        
        # Generate future tokens autoregressively
        for i in range(self.n_future_frames):
            # Current sequence length
            current_seq_len = all_tokens.shape[1]
            
            # Add temporal positional encoding (always start from 0 for the full sequence)
            tokens_with_pos = self.temporal_pos_embed(all_tokens, start_idx=0)
            
            # Pass through transformer blocks (causal attention ensures autoregressive behavior)
            x = tokens_with_pos
            
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            
            # Use the last frame's transformer output directly as the next token
            last_frame = x[:, -1, :, :]  # [B, S, D]
            next_token = last_frame.unsqueeze(1)  # [B, 1, S, D]
            
            # Append to sequence
            all_tokens = torch.cat([all_tokens, next_token], dim=1)
        
        # Reshape back to spatial format
        all_hidden = self.reshape_to_spatial(all_tokens)  # [B*(N+M), S, D]
        
        # Extend positional encoding for future frames
        if pos is not None:
            # Strategy: Future frames use the same spatial grid as current frames
            # This makes sense because we're predicting at the same spatial locations
            pos_per_frame = pos.view(B, N, S, -1)  # [B, N, S, pos_dim]
            
            # Use the first frame's spatial pattern for all future frames
            # (spatial positions should be consistent across time)
            spatial_pattern = pos_per_frame[:, 0:1, :, :]  # [B, 1, S, pos_dim]
            future_pos = spatial_pattern.repeat(1, self.n_future_frames, 1, 1)  # [B, M, S, pos_dim]
            
            all_pos_frames = torch.cat([pos_per_frame, future_pos], dim=1)  # [B, N+M, S, pos_dim]
            all_pos = all_pos_frames.view(-1, S, pos_per_frame.shape[-1])  # [B*(N+M), S, pos_dim]
        else:
            all_pos = None
            
        return all_hidden, all_pos
    
    def forward_with_targets(self, hidden, N, future_hidden=None):
        """
        Training forward pass with optional teacher forcing.
        
        Args:
            hidden: [B*N, S, D] - current frame tokens
            N: number of current frames
            future_hidden: [B*M, S, D] - ground truth future tokens (optional)
            
        Returns:
            predicted_future: [B*M, S, D] - predicted future tokens
            all_hidden: [B*(N+M), S, D] - all tokens
        """
        tokens = self.reshape_to_temporal(hidden, N)
        B, _, S, D = tokens.shape
        
        if future_hidden is not None and self.training:
            # Teacher forcing: use ground truth future tokens during training
            future_tokens = self.reshape_to_temporal(future_hidden, self.n_future_frames)
            all_tokens = torch.cat([tokens, future_tokens], dim=1)  # [B, N+M, S, D]
            
            # Add positional encoding and process full sequence
            tokens_with_pos = self.temporal_pos_embed(all_tokens)
            x = tokens_with_pos
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            
            # In teacher forcing, the transformer output at each position predicts the next token
            # Use positions 0:(N+M-1) to predict positions 1:(N+M)
            predicted_tokens = x[:, :-1, :, :]  # [B, N+M-1, S, D] - transformer output directly
            
            # Extract only the future predictions (positions N:(N+M))
            predicted_future = predicted_tokens[:, N-1:N-1+self.n_future_frames, :, :]  # [B, M, S, D]
            
            # Reshape predictions
            predicted_future = self.reshape_to_spatial(predicted_future)
            all_hidden = self.reshape_to_spatial(all_tokens)
            
            return predicted_future, all_hidden
        else:
            # Normal autoregressive generation
            all_hidden, _ = self.forward(hidden, N)
            predicted_future = all_hidden[B*N:]  # Future tokens only
            return predicted_future, all_hidden