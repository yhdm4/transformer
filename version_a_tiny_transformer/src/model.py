import torch
from torch import nn

from attention import MultiHeadSelfAttention


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attention_mask=None):
        attn_out = self.attn(x, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        return x


class TinyTransformerClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        dim,
        num_heads,
        ff_dim,
        num_layers,
        num_classes,
        max_len,
        pad_id=0,
        dropout=0.1,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, dim, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderBlock(dim, num_heads, ff_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        attention_mask = input_ids.ne(self.pad_id)
        positions = torch.arange(seq_len, device=input_ids.device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        for layer in self.layers:
            x = layer(x, attention_mask)

        mask = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.classifier(pooled)
