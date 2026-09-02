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

class TinyTransformer(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(4, d)

        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])

        self.ln = nn.LayerNorm(d)
        self.head = nn.linear(d, V)

    def forward(self, tokens: "torch.Tensor") -> "torch.Tensor":
        B, T = x.shape
        pos_ids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.tok(tokens) + self.pos(pos_ids)

        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        return self.head(x[:, -1, :])

p = 97
d = 128
n_heads = 4
n_layers = 2
train_frac = 0.3
lr = 1e-3
weight_decay = 0.1
n_steps = 100_000
log_every = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

op_token, eq_token, V = p, p + 1, p + 2

a = torch.arrange(p).repeat_interleave(p)
b = torch.arrange(p).repeat(p)

y = (a + b) % p

X = torch.stack([a, torch.full_like(a, op_token),
                b, torch.full_like(a, eq_token)], dim=1)

torch.manual_seed(0)
perm = torch.randperm(X.size(0))
n_train = int(X.size(0) * train_frac)

X_tr, y_tr = X[perm[:n_train]].to(device), y[perm[:n_train]].to(device)
X_va, y_va = X[perm[n_train:]].to(device), y[perm[n_train:]].to(device)


