import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter but part of the module)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input tensor
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        # Projection layers for queries, keys, values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        
    def forward(self, query, key, value, mask=None):
        """
        Perform multi-head attention
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model]
            value: Value tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor after attention [batch_size, seq_len, d_model]
        """
        batch_size = query.shape[0]
        
        # Project and reshape for multi-head attention
        q = self.q_proj(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project to output dimension
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_proj(attn_output)

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        """
        Process input through one transformer encoder layer
        
        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            src_mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor after processing [batch_size, seq_len, d_model]
        """
        # Pre-LN architecture (more stable training)
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, mask=src_mask)
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class TransformerBlock(nn.Module):
    """Transformer encoder block with multiple layers"""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Stack of encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, src, src_mask=None):
        """
        Process input through transformer encoder
        
        Args:
            src: Input tensor [batch_size, seq_len, d_model]
            src_mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Output tensor after transformer encoding [batch_size, seq_len, d_model]
        """
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through each encoder layer
        for layer in self.layers:
            src = layer(src, src_mask)
            
        # Apply final normalization
        return self.norm(src)

class CausalSelfAttention(nn.Module):
    """Self-attention layer with causal masking for autoregressive models"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, nhead, dropout)
        
    def forward(self, x):
        """
        Apply causal self-attention
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output tensor after causal self-attention [batch_size, seq_len, d_model]
        """
        # Create causal mask (lower triangular)
        seq_len = x.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        mask = ~mask  # Invert to get causal mask
        
        # Apply self-attention with causal mask
        return self.attn(x, x, x, mask=mask.unsqueeze(0))