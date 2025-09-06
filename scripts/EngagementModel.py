import torch.nn as nn
import torch

# --- Transformer + Residual + Separate Heads + Quantiles ---
class EngagementModel(nn.Module):
    def __init__(self, seq_input_dim, static_input_dim, hidden_size, transformer_heads, dropout, embed_dim=128):
        super().__init__()
        
        self.seq_embed = nn.Linear(seq_input_dim, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=transformer_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.attn = nn.Linear(embed_dim, 1)

        self.seq_fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.static_fc = nn.Sequential(
            nn.Linear(static_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Point estimates
        self.views_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.likes_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Quantile estimates
        self.views_head_p10 = nn.Linear(hidden_size, 1)
        self.views_head_p90 = nn.Linear(hidden_size, 1)
        self.likes_head_p10 = nn.Linear(hidden_size, 1)
        self.likes_head_p90 = nn.Linear(hidden_size, 1)

    def forward(self, x_seq, x_static):
        x_seq = self.seq_embed(x_seq)  # Add embedding layer
        x_seq = self.transformer(x_seq)
        attn_weights = torch.softmax(self.attn(x_seq), dim=1)
        seq_repr = (x_seq * attn_weights).sum(dim=1)
        seq_out = self.seq_fc(seq_repr)
        static_out = self.static_fc(x_static)
        combined = seq_out + static_out

        return {
            'views_mean': self.views_head(combined),
            'likes_mean': self.likes_head(combined),
            'views_p10': self.views_head_p10(combined),
            'views_p90': self.views_head_p90(combined),
            'likes_p10': self.likes_head_p10(combined),
            'likes_p90': self.likes_head_p90(combined)
        }
 