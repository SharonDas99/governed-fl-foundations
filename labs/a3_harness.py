"""
ASSIGNMENT 3 HARNESS: run the experiment grid and write a CSV you can analyze.

    python labs/a3_harness.py                         default grid, about 30 seconds
    python labs/a3_harness.py --clients 3 5 10 --skews none label quantity --E 1 5 20
    python labs/a3_harness.py --out results/my_grid.csv

For every (number of hospitals, skew, local epochs E) it records:
    global accuracy and AUC on the neutral test set after 20 rounds
    the worst hospital's accuracy and the gap between best and worst
    rounds needed to get within 2 percent of the centralized model's loss
    the centralized and local-only references

Your report is about what these numbers MEAN. The harness is only plumbing.
Columns you add to the CSV (for example a defense or a noise level) are
welcome; say so in the report.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl import federation as F, model
from fl.world import make_world

ROOT = Path(__file__).resolve().parent.parent


def rounds_to_target(world, E, target, lr=0.05, max_rounds=60):
    rng = np.random.default_rng(690)
    theta = model.init(world.X_test.shape[1])
    for r in range(1, max_rounds + 1):
        thetas, sizes = [], []
        for c in world.clients:
            thetas.append(model.local_train(theta, c.X_train, c.y_train, E, 64, lr, rng=rng)); sizes.append(c.n)
        total = sum(sizes)
        theta = sum(t * (n / total) for t, n in zip(thetas, sizes))
        if model.loss(theta, world.X_test, world.y_test) <= target:
            return r
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clients", type=int, nargs="+", default=[3, 5, 10])
    ap.add_argument("--skews", nargs="+", default=["none", "label", "quantity", "feature", "concept", "realistic"])
    ap.add_argument("--E", type=int, nargs="+", default=[1, 5, 20])
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--out", default="results/a3_results.csv")
    a = ap.parse_args()

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in a.clients:
        for skew in a.skews:
            world = make_world(n, skew)
            central = F.centralized(world, epochs=100, lr=0.05)
            c_acc = model.accuracy(central, world.X_test, world.y_test)
            target = model.loss(central, world.X_test, world.y_test) * 1.02
            local = F.local_only(world, epochs=100)
            local_small = min(model.accuracy(local[c.name], *world.test_for(k)) for k, c in enumerate(world.clients))
            for E in a.E:
                theta, hist = F.run(world, F.Config(rounds=a.rounds, E=E, verbose=False))
                last = hist[-1]
                per = last["per_client"]
                row = {"n_clients": n, "skew": skew, "E": E, "rounds": a.rounds,
                       "fedavg_acc": round(last["global_acc"], 4), "fedavg_auc": round(last["global_auc"], 4),
                       "worst_hospital_acc": round(min(per.values()), 4), "best_worst_gap": round(max(per.values()) - min(per.values()), 4),
                       "rounds_to_converge": rounds_to_target(world, E, target),
                       "centralized_acc": round(c_acc, 4), "worst_local_only_acc": round(local_small, 4)}
                rows.append(row)
                print(f"n={n:<3} skew={skew:<10} E={E:<3} fedavg {row['fedavg_acc']:.3f}  worst {row['worst_hospital_acc']:.3f}  gap {row['best_worst_gap']:.3f}  rounds {row['rounds_to_converge']}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {out.relative_to(ROOT)} with {len(rows)} rows")
