# Grokking on modular addition

A 2-layer transformer written from scratch in PyTorch, trained on (a + b) mod 97,
to reproduce grokking: generalization long after the training set is memorized.

## Run

```bash
python grokking.py                        # baseline (Power et al. 2022 settings, full batch)
python grokking.py --wd 0.1 --d 512       # override any setting; see --help
python sweep.py                           # all sweeps (27 runs), resumable
python sweep.py wd                        # one sweep: wd | width | dip
python sweep.py --dry-run                 # list the commands only
```

Baseline: d=128, 4 heads, 2 layers, AdamW (lr 1e-3, weight decay 1.0, betas 0.9/0.98),
10-step warmup, full batch, 30% train split. `--seed` sets the model init;
`--data_seed` sets the train/val split, so seeds can vary without changing the data.

## Sweeps (`sweep.py`)

| Name | Varies | Runs |
|------|--------|------|
| `wd` | weight decay 0, 0.1, 0.3, 1.0, 3.0 × 3 seeds | 15 |
| `width` | d = 64, 128, 256, 512 × 2 seeds (wd 1.0) | 8 |
| `dip` | Adam β2 0.999 vs 0.98 at wd 0.1, full 100k steps | 4 |

Runs stop 2,000 steps after validation accuracy reaches 99% (`--patience`),
except the `dip` sweep, which keeps the whole curve.

## Logs

`logs/<sweep>/wd{wd}_d{d}_b2{beta2}_s{seed}.json` contains, every `--log_every` steps:
`step`, `train_acc`, `val_acc`, `train_loss`, `val_loss`, `weight_norm` (global L2 norm),
plus `config`, `memorize_step` (train acc ≥ 99%) and `grok_step` (val acc ≥ 99%).
