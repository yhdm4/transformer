from torch import nn
import torch
from multi_head_self_attention import MultiHeadSelfAttention
class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.norm1 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dim)
        )
        self.norm2 = nn.LayerNorm(dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.attn(x)
        x = self.norm1(x + self.dropout(attn_out))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))

        return x
    
    
class TinyTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, dim, num_heads, ff_dim, num_layers, num_classes, max_len=128):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])

        self.classifier = nn.Linear(dim, num_classes)

    def forward(self, input_ids):
        B, L = input_ids.shape

        positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, L)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        for layer in self.layers:
            x = layer(x)

        pooled = x.mean(dim=1)

        logits = self.classifier(pooled)
        return logits
    
B, L = 4, 20
vocab_size = 5000
num_classes = 2

input_ids = torch.randint(0, vocab_size, (B, L))

model = TinyTransformerClassifier(
    vocab_size=vocab_size,
    dim=128,
    num_heads=4,
    ff_dim=256,
    num_layers=2,
    num_classes=num_classes,
    max_len=128
)

logits = model(input_ids)
print(logits.shape)  # [4, 2]