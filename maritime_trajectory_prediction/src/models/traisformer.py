import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float))
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        q = self.q_proj(query).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out_proj(attn_output)

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        src2 = self.norm1(src)
        src2 = self.self_attn(src2, src2, src2, mask=src_mask)
        src = src + self.dropout1(src2)
        
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src

class TrAISformer(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        
        # Model dimensions
        self.d_model = config.d_model
        self.nhead = config.nhead
        self.num_layers = config.num_layers
        self.dim_feedforward = config.dim_feedforward
        
        # Feature dimensions
        self.lat_bins = config.lat_bins
        self.lon_bins = config.lon_bins
        self.sog_bins = config.sog_bins
        self.cog_bins = config.cog_bins
        
        # Embedding layers
        self.lat_embed = nn.Embedding(self.lat_bins, self.d_model // 4)
        self.lon_embed = nn.Embedding(self.lon_bins, self.d_model // 4)
        self.sog_embed = nn.Embedding(self.sog_bins, self.d_model // 4)
        self.cog_embed = nn.Embedding(self.cog_bins, self.d_model // 4)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model)
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward
            ) for _ in range(self.num_layers)
        ])
        
        # Output classification heads
        self.lat_classifier = nn.Linear(self.d_model, self.lat_bins)
        self.lon_classifier = nn.Linear(self.d_model, self.lon_bins)
        self.sog_classifier = nn.Linear(self.d_model, self.sog_bins)
        self.cog_classifier = nn.Linear(self.d_model, self.cog_bins)
        
    def embed_inputs(self, lat_idx, lon_idx, sog_idx, cog_idx):
        # Embed each feature
        lat_emb = self.lat_embed(lat_idx)
        lon_emb = self.lon_embed(lon_idx)
        sog_emb = self.sog_embed(sog_idx)
        cog_emb = self.cog_embed(cog_idx)
        
        # Concatenate embeddings
        return torch.cat([lat_emb, lon_emb, sog_emb, cog_emb], dim=-1)
    
    def forward(self, lat_idx, lon_idx, sog_idx, cog_idx, src_mask=None):
        # Embed and apply positional encoding
        x = self.embed_inputs(lat_idx, lon_idx, sog_idx, cog_idx)
        x = self.pos_encoder(x)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x, src_mask)
        
        # Classification heads for next timestep prediction
        lat_logits = self.lat_classifier(x)
        lon_logits = self.lon_classifier(x)
        sog_logits = self.sog_classifier(x)
        cog_logits = self.cog_classifier(x)
        
        return lat_logits, lon_logits, sog_logits, cog_logits
    
    def training_step(self, batch, batch_idx):
        lat_idx, lon_idx, sog_idx, cog_idx = batch['inputs']
        next_lat_idx, next_lon_idx, next_sog_idx, next_cog_idx = batch['targets']
        
        # Create causal mask for autoregressive training
        seq_len = lat_idx.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(self.device)
        mask = ~mask
        
        # Forward pass
        lat_logits, lon_logits, sog_logits, cog_logits = self(
            lat_idx, lon_idx, sog_idx, cog_idx, src_mask=mask
        )
        
        # Compute losses for each attribute
        lat_loss = F.cross_entropy(lat_logits.view(-1, self.lat_bins), next_lat_idx.view(-1))
        lon_loss = F.cross_entropy(lon_logits.view(-1, self.lon_bins), next_lon_idx.view(-1))
        sog_loss = F.cross_entropy(sog_logits.view(-1, self.sog_bins), next_sog_idx.view(-1))
        cog_loss = F.cross_entropy(cog_logits.view(-1, self.cog_bins), next_cog_idx.view(-1))
        
        # Total loss is the sum of individual losses
        total_loss = lat_loss + lon_loss + sog_loss + cog_loss
        
        # Log losses
        self.log('train_loss', total_loss)
        self.log('lat_loss', lat_loss)
        self.log('lon_loss', lon_loss)
        self.log('sog_loss', sog_loss)
        self.log('cog_loss', cog_loss)
        
        return total_loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.optimizer.lr,
            weight_decay=self.hparams.optimizer.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_epochs,
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
