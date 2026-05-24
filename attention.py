from torch import nn
import torch
class SelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)

    def forward(self, x):
        Q = self.q(x)  # [B, L, D]
        K = self.k(x)  # [B, L, D]
        V = self.v(x)  # [B, L, D]

        scores = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
        weights = torch.softmax(scores, dim=-1)

        out = weights @ V
        return out
    
B, L, D = 4, 6, 8
x = torch.randn(B, L, D)

attn = SelfAttention(D)
out = attn(x)

print(out.shape)  # [4, 6, 8]