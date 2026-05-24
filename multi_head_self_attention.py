from torch import nn
import torch
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()

        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape

        qkv = self.qkv(x)  # [B, L, 3D]
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        Q, K, V = qkv[0], qkv[1], qkv[2]
        # Q/K/V: [B, H, L, head_dim]

        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)

        out = weights @ V  # [B, H, L, head_dim]

        out = out.transpose(1, 2)  # [B, L, H, head_dim]
        out = out.reshape(B, L, D) # [B, L, D]

        out = self.out_proj(out)
        return out