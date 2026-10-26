"""
LAB 6b: SECURE AGGREGATION. The coordinator learns the sum and nothing else.

    python labs/lab6_secagg.py

Three hospitals each add pairwise random masks to their update. The masks
cancel in the sum. The coordinator sees three vectors of noise and one
correct average.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl import model, privacy, ui
from fl.world import make_world

if __name__ == "__main__":
    world = make_world(3, "realistic")
    theta = model.init(world.X_test.shape[1])
    rng = np.random.default_rng(690)
    deltas = [model.local_train(theta, c.X_train, c.y_train, 5, 64, 0.1, rng=rng) - theta for c in world.clients]

    ui.title("LAB 6b  |  Secure aggregation in four moves")
    print("  1. Every pair of hospitals agrees on a shared random vector (a secret the coordinator never sees).")
    print("  2. In each pair, one hospital ADDS the vector to its update, the other SUBTRACTS it.")
    print("  3. Each hospital sends its masked update. To the coordinator it looks like noise.")
    print("  4. The coordinator sums them. Every mask appears once with + and once with -, so they vanish.\n")

    total, masked = privacy.secure_aggregate(deltas, rng=np.random.default_rng(7), return_masked=True)
    avg_true = np.mean(np.stack(deltas), axis=0)
    avg_secagg = total / len(deltas)

    ui.section("First five numbers of each update (weights for age, prior admissions, length of stay, medications, comorbidity)")
    rows = []
    for c, d, m in zip(world.clients, deltas, masked):
        rows.append([c.name, " ".join(f"{v:+.2f}" for v in d[:5]), " ".join(f"{v:+.2f}" for v in m[:5])])
    ui.table(["hospital", "true update (secret)", "what the coordinator receives"], rows)
    print(f"\n  average of the TRUE updates:   {' '.join(f'{v:+.3f}' for v in avg_true[:5])}")
    print(f"  average from MASKED updates:   {' '.join(f'{v:+.3f}' for v in avg_secagg[:5])}")
    print(f"  identical: {np.allclose(avg_true, avg_secagg)}")

    ui.notice([
        "The coordinator cannot run Lab 5b's reconstruction on any single hospital: it never sees a single hospital's update.",
        "It can still learn from the SUM. With two hospitals, sum minus my own update equals the other's. Minimum participation matters.",
        "Secure aggregation protects against a curious coordinator. It does nothing against a poisoning hospital, and it hides the update norms the Lab 6a defenses rely on.",
        "That tension (privacy from the coordinator vs robustness by the coordinator) is a design decision for the policy in Lab 7.",
    ])
    ui.note_prompt(6, "Why does the consortium agreement need a minimum number of participants per round?")
