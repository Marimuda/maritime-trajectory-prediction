# File: src/models/blocks/motion_transformer.py
"""
Motion Transformer building blocks for maritime trajectory prediction.
Contains:
- TransformerDecoderLayer
- ContextEncoder
- MotionDecoder
- MotionTransformer
"""

import torch
import torch.nn.functional as F
from torch import nn

from .transformer import MultiHeadAttention, PositionalEncoding, TransformerEncoderLayer


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with self-attention, cross-attention, and feed-forward.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Self-attention
        sa = self.self_attn(x, x, x, mask=tgt_mask)
        x = self.norm1(x + self.dropout(sa))
        # Cross-attention
        ca = self.cross_attn(x, memory, memory, mask=memory_mask)
        x = self.norm2(x + self.dropout(ca))
        # Feed-forward
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff))
        return x


class ContextEncoder(nn.Module):
    """
    Encoder for historical trajectory context using transformer layers.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # x: [B, T, input_dim]
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        for layer in self.encoder_layers:
            x = layer(x, mask)
        return x


class MotionDecoder(nn.Module):
    """
    Decoder generating multimodal future trajectories.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_queries: int = 6,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        prediction_horizon: int = 30,
        output_dim: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_queries = n_queries
        self.prediction_horizon = prediction_horizon
        # learnable queries for modes
        self.query_emb = nn.Parameter(torch.randn(n_queries, d_model))
        # positional embeddings for each future step
        self.query_pos = nn.Parameter(torch.randn(prediction_horizon, d_model))
        # decoder layers
        self.decoder_layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        # output heads
        self.traj_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim),
        )
        self.conf_head = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, context: torch.Tensor, memory_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        B = context.size(0)
        # expand queries to batch
        queries = self.query_emb.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]
        # build time-expanded sequence of queries
        all_queries = []
        for t in range(self.prediction_horizon):
            qp = queries + self.query_pos[t].unsqueeze(0)
            all_queries.append(qp)
        # [B, H, Q, D] -> flatten to [B, H*Q, D]
        dq = torch.cat(all_queries, dim=1)
        x = dq
        for layer in self.decoder_layers:
            x = layer(x, context, tgt_mask=None, memory_mask=memory_mask)
        # reshape back [B, H, Q, D]
        x = x.view(B, self.prediction_horizon, self.n_queries, self.d_model)
        # compute outputs
        trajectories = self.traj_head(x)
        confidences = self.conf_head(x.mean(dim=1)).squeeze(-1)
        return {
            "trajectories": trajectories,
            "confidences": confidences,
            "query_features": x,
        }


class MotionTransformer(nn.Module):
    """
    Full Motion Transformer combining context encoder and decoder.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_queries: int = 6,
        encoder_layers: int = 4,
        decoder_layers: int = 6,
        n_heads: int = 8,
        d_ff: int = 1024,
        dropout: float = 0.1,
        prediction_horizon: int = 30,
        output_dim: int = 4,
    ):
        super().__init__()
        self.context_encoder = ContextEncoder(
            input_dim, d_model, encoder_layers, n_heads, d_ff, dropout
        )
        self.motion_decoder = MotionDecoder(
            d_model,
            n_queries,
            decoder_layers,
            n_heads,
            d_ff,
            dropout,
            prediction_horizon,
            output_dim,
        )
        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self, x: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        context = self.context_encoder(x, context_mask)
        return self.motion_decoder(context, memory_mask=context_mask)

    def predict_best_trajectory(
        self, x: torch.Tensor, context_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        outputs = self(x, context_mask)
        conf = outputs["confidences"]  # [B, Q]
        trajs = outputs["trajectories"]  # [B, H, Q, D]
        best = conf.argmax(dim=-1)
        # select best per batch
        return torch.stack([trajs[b, :, best[b], :] for b in range(x.size(0))], dim=0)

    def compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        loss_type: str = "best_of_n",
    ) -> dict[str, torch.Tensor]:
        trajs = outputs["trajectories"]  # [B, H, Q, D]
        conf = outputs["confidences"]  # [B, Q]
        B, H, Q, D = trajs.shape
        # expand targets
        targ = targets.unsqueeze(2).expand(-1, -1, Q, -1)
        if loss_type == "best_of_n":
            errs = F.mse_loss(trajs, targ, reduction="none").mean(
                dim=(-2, -1)
            )  # [B, Q]
            best_errs, idx = errs.min(dim=-1)
            reg_loss = best_errs.mean()
            cls_loss = F.cross_entropy(conf, idx)
            total = reg_loss + 0.1 * cls_loss
            return {
                "total_loss": total,
                "regression_loss": reg_loss,
                "classification_loss": cls_loss,
                "best_mode_errors": best_errs,
            }
        elif loss_type == "weighted":
            errs = F.mse_loss(trajs, targ, reduction="none").mean(dim=-1)
            w_errs = (errs * conf).sum(dim=-1).mean()
            reg_loss = w_errs
            conf_reg = -torch.log(conf + 1e-8).mean()
            total = reg_loss + 0.01 * conf_reg
            return {
                "total_loss": total,
                "regression_loss": reg_loss,
                "confidence_regularization": conf_reg,
            }
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
