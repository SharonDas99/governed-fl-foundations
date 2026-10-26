"""
LAB 5a: POISONING. What one bad hospital can do to everyone's model.

    python labs/lab5_poison.py                       all four attacks, one table
    python labs/lab5_poison.py --attack boost --attacker Northgate --scale 10
    python labs/lab5_poison.py --attack backdoor --rounds 15 --verbose

The consortium has grown to five hospitals. One of them is dishonest.
  flip      data poisoning: trains honestly on flipped labels
  boost     model poisoning: flipped labels AND a scaled-up update (model replacement)
  backdoor  teaches the model that a trigger pattern means "safe to discharge"
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl import federation as F, ui
from fl.world import make_world

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--attack", default="all", choices=["all", "none", "flip", "boost", "backdoor"])
    ap.add_argument("--attacker", default="Three Rivers")
    ap.add_argument("--scale", type=float, default=10.0)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()

    world = make_world(5, "realistic")
    ui.title("LAB 5a  |  Poisoning: five hospitals, one of them dishonest")
    print(world.describe())

    runs = [("none", None, 1.0), ("flip", "Three Rivers", 1.0), ("flip", "Northgate", 1.0),
            ("boost", "Three Rivers", a.scale), ("backdoor", "Three Rivers", a.scale)] if a.attack == "all" \
        else [(a.attack, a.attacker if a.attack != "none" else None, a.scale)]

    rows = []
    for kind, who, scale in runs:
        cfg = F.Config(rounds=a.rounds, attack=None if kind == "none" else kind, attacker=who, attack_scale=scale, verbose=a.verbose)
        if a.verbose:
            ui.section(f"{kind} by {who}")
        theta, hist = F.run(world, cfg)
        last, first = hist[-1], hist[0]
        share = next((c.n for c in world.clients if c.name == who), 0) / sum(c.n for c in world.clients)
        norms = dict(zip(first["participants"], first["update_norms"]))
        rows.append([kind, who or "-", f"{share:.0%}" if who else "-", ui.f3(last["global_acc"]),
                     ui.f3(min(last["per_client"].values())),
                     f"{last['attack_success_rate']:.0%}" if "attack_success_rate" in last else "-",
                     norms.get(who, "-") if who else f"{min(norms.values())} to {max(norms.values())}"])
    ui.section("Results after the last round")
    ui.table(["attack", "attacker", "attacker's share", "global acc", "worst hospital", "backdoor success", "attacker's update norm (round 1)"], rows)

    ui.notice([
        "flip by a 14% hospital: nothing. The weighted average dilutes a small data poisoner.",
        "flip by the 35% hospital: real damage. Size is power in FedAvg, for good and for bad.",
        "boost: one round, one hospital, model destroyed. Scaling the update beats the weighting.",
        "backdoor: overall accuracy looks almost fine. Readmitted patients with the trigger are called safe.",
        "Look at the last column. The attacker's update is 5 to 10 times larger than anyone's. The coordinator could have seen it.",
    ])
    ui.note_prompt(5, "Which attack would you have caught by looking only at update norms, and which one not?")
