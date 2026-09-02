import torch
import math
from torch import nn

class MultiHeadCausalAttention(nn.Module):

    def __init__(self, d: int, n_heads: int) -> None:
        super().__init__()
        self.q = nn.Linear(d, d, bias=False)
        self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False)
        self.d = d
        assert d % n_heads == 0

        self.n_heads = n_heads
        self.d_head = d // n_heads

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":

        B, T, d = x.shape
        Q, K, V = self.q(x), self.k(x), self.v(x)
        
        Q = Q.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        K = K.view(B, T, self.n_heads, self.d_head).transpose(1,2)
        V = V.view(B, T, self.n_heads, self.d_head).transpose(1,2)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_head)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        out = weights @ V
        return out.transpose(2,1).reshape(B, T, d)

class Block(nn.Module):

    def __init__(self, d: int, n_heads: int = 4, mlp_mult: int = 4) -> None:
        super().__init__()
        self.ln1 = nn.Layernorm(d)
        self.attn = MultiHeadCausalAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * mlp_multi),
            nn.GELU(),
            nn.Linear(d * mpl_multi, d)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        T = x.shape(1)
        h = self.ln1(x)
        attn_out = self.attn(h)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

