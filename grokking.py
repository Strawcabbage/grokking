import torch
import math
from torch import nn
import os
import json
import argparse

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
        self.ln1 = nn.LayerNorm(d)
        self.attn = MultiHeadCausalAttention(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, d * mlp_mult),
            nn.GELU(),
            nn.Linear(d * mlp_mult, d)
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        T = x.size(1)
        h = self.ln1(x)
        attn_out = self.attn(h)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x

class TinyTransformer(nn.Module):

    def __init__(self, V: int, d: int, n_heads: int, n_layers: int) -> None:
        super().__init__()
        self.tok = nn.Embedding(V, d)
        self.pos = nn.Embedding(4, d)

        self.blocks = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])

        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, V)

    def forward(self, tokens: "torch.Tensor") -> "torch.Tensor":
        B, T = tokens.shape
        pos_ids = torch.arange(T, device=tokens.device).unsqueeze(0).expand(B, T)
        x = self.tok(tokens) + self.pos(pos_ids)

        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        return self.head(x[:, -1, :])

def parse_args() -> argparse.Namespace:
    # Defaults follow Power et al. (2022) except full-batch training:
    # d=128, AdamW lr 1e-3, weight decay 1.0, betas (0.9, 0.98), 10-step warmup.
    ap = argparse.ArgumentParser(description="Grokking on (a + b) mod p")
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--train_frac", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1.0, help="AdamW weight decay")
    ap.add_argument("--beta2", type=float, default=0.98)
    ap.add_argument("--warmup", type=int, default=10, help="linear lr warmup steps")
    ap.add_argument("--steps", type=int, default=100_000, help="max steps")
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=2_000,
                    help="stop this many steps after val acc >= 99%% (0 = never stop early)")
    ap.add_argument("--seed", type=int, default=0, help="model init / training seed")
    ap.add_argument("--data_seed", type=int, default=0, help="train/val split seed")
    ap.add_argument("--out", type=str, default="logs")
    return ap.parse_args()

def weight_norm(model: nn.Module) -> float:
    return torch.sqrt(sum((w.detach() ** 2).sum() for w in model.parameters())).item()

def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = args.p

    op_token, eq_token, V = p, p + 1, p + 2

    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)

    y = (a + b) % p

    X = torch.stack([a, torch.full_like(a, op_token),
                    b, torch.full_like(a, eq_token)], dim=1)

    # Separate seeds: the split depends only on data_seed, the init only on seed.
    g = torch.Generator().manual_seed(args.data_seed)
    perm = torch.randperm(X.size(0), generator=g)
    n_train = int(X.size(0) * args.train_frac)

    X_tr, y_tr = X[perm[:n_train]].to(device), y[perm[:n_train]].to(device)
    X_va, y_va = X[perm[n_train:]].to(device), y[perm[n_train:]].to(device)

    torch.manual_seed(args.seed)
    model = TinyTransformer(V, args.d, args.n_heads, args.n_layers).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd,
                              betas=(0.9, args.beta2))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / args.warmup) if args.warmup > 0 else 1.0)
    loss_fn = nn.CrossEntropyLoss()

    history = {"step": [], "train_acc": [], "val_acc": [],
               "train_loss": [], "val_loss": [], "weight_norm": []}
    memorize_step = grok_step = None

    for step in range(1, args.steps + 1):
        model.train()
        optim.zero_grad()
        logits = model(X_tr)
        loss = loss_fn(logits, y_tr)
        loss.backward()
        optim.step()
        sched.step()

        if step % args.log_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                tr_logits, va_logits = model(X_tr), model(X_va)
                tr_acc = (tr_logits.argmax(-1) == y_tr).float().mean().item()
                va_acc = (va_logits.argmax(-1) == y_va).float().mean().item()
                tr_loss = loss_fn(tr_logits, y_tr).item()
                va_loss = loss_fn(va_logits, y_va).item()
            wnorm = weight_norm(model)
            history["step"].append(step)
            history["train_acc"].append(tr_acc)
            history["val_acc"].append(va_acc)
            history["train_loss"].append(tr_loss)
            history["val_loss"].append(va_loss)
            history["weight_norm"].append(wnorm)
            if memorize_step is None and tr_acc >= 0.99:
                memorize_step = step
            if grok_step is None and va_acc >= 0.99:
                grok_step = step
            print(f"[lesson 11] step={step:6d}  loss={tr_loss:.4f}  val_loss={va_loss:.4f}  "
                  f"train_acc={tr_acc:.3f}  val_acc={va_acc:.3f}  |w|={wnorm:.1f}")

            if args.patience and grok_step is not None and step - grok_step >= args.patience:
                break

    history["config"] = vars(args)
    history["memorize_step"] = memorize_step
    history["grok_step"] = grok_step

    os.makedirs(args.out, exist_ok=True)
    name = f"wd{args.wd:g}_d{args.d}_b2{args.beta2:g}_s{args.seed}.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved {os.path.join(args.out, name)}  memorize_step={memorize_step}  grok_step={grok_step}")

if __name__ == "__main__":
    main()
