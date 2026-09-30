"""Run the grokking sweeps one after another, skipping runs whose log exists.

    python sweep.py                 # all experiments
    python sweep.py wd width        # just some of them
    python sweep.py --dry-run       # print the commands without running
    python sweep.py --device=cuda   # other --flags are passed to every run

Each run writes logs/<experiment>/wd{wd}_d{d}_b2{beta2}_s{seed}.json.
"""
import itertools
import os
import subprocess
import sys

EXPERIMENTS = {
    # 1. Main result: how weight decay changes time-to-grok. 5 x 3 = 15 runs.
    "wd": dict(wd=[0, 0.1, 0.3, 1.0, 3.0], d=[128], beta2=[0.98], seed=[0, 1, 2]),
    # 2. Does a wider model grok faster? 4 x 2 = 8 runs.
    "width": dict(wd=[1.0], d=[64, 128, 256, 512], beta2=[0.98], seed=[0, 1]),
    # 3. Is the step-55k dip an Adam beta2 instability? Original config vs beta2=0.98.
    #    2 x 2 = 4 runs, no early stopping so the whole curve is kept.
    "dip": dict(wd=[0.1], d=[128], beta2=[0.999, 0.98], seed=[0, 1], patience=[0]),
}


def runs(name):
    grid = EXPERIMENTS[name]
    keys = list(grid)
    for values in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, values))


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    extra = [a for a in sys.argv[1:] if a.startswith("--") and a != "--dry-run"]
    names = args or list(EXPERIMENTS)
    for name in names:
        out = os.path.join("logs", name)
        for cfg in runs(name):
            log = os.path.join(out, f"wd{cfg['wd']:g}_d{cfg['d']}_b2{cfg['beta2']:g}_s{cfg['seed']}.json")
            if os.path.exists(log):
                print(f"skip {log}")
                continue
            cmd = [sys.executable, "grokking.py", "--out", out] + \
                  [f"--{k}={v}" for k, v in cfg.items()] + extra
            print(" ".join(cmd), flush=True)
            if not dry:
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
