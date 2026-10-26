"""
THE FEDERATION ENGINE: one configurable loop that every lab from 2 onward uses.

Lab 1 has its own 60-line version so you can see FedAvg with nothing else in
the way. This file is that same loop with the knobs, attacks and defenses
exposed as arguments, so each later lab changes one thing and reruns.

One round, in order:
  1. select     choose a fraction C of clients (McMahan's C)
  2. local      each selected client trains E epochs with batch size B
                starting from the global model (an attacker sends something else)
  3. clip       optional: cap every update's norm                     (defense)
  4. secagg     optional: mask updates so the server only sees the sum (defense)
  5. aggregate  weighted mean (FedAvg) or a robust aggregator          (defense)
  6. dp         optional: add calibrated Gaussian noise                (defense)
  7. apply      global model += aggregated update
  8. evaluate   neutral test set overall, and the slice that came from each hospital
"""
from dataclasses import dataclass, field

import numpy as np

from fl import aggregators, attacks, model, privacy


@dataclass
class Config:
    rounds: int = 20
    E: int = 5                 # local epochs
    B: int = 64                # local batch size
    lr: float = 0.1
    C: float = 1.0             # fraction of clients per round
    prox_mu: float = 0.0       # FedProx strength (0 = plain FedAvg)
    aggregator: str = "mean"   # mean | simple_mean | median | trimmed_mean | krum
    clip_norm: float = None    # None = no clipping
    secagg: bool = False
    dp_noise: float = 0.0      # noise multiplier; needs clip_norm
    attack: str = None         # None | flip | boost | backdoor
    attacker: str = None       # client name
    attack_scale: float = 10.0
    reject_norm_above: float = None   # anomaly rule: drop updates with raw norm above this
    target_acc: float = None   # stop early when the neutral test accuracy reaches this
    seed: int = 690
    verbose: bool = True
    extra: dict = field(default_factory=dict)


def evaluate(theta, world):
    per = {c.name: model.accuracy(theta, *world.test_for(k)) for k, c in enumerate(world.clients)}
    return {"global_acc": model.accuracy(theta, world.X_test, world.y_test),
            "global_auc": model.auc(theta, world.X_test, world.y_test),
            "per_client": per}


def run(world, cfg: Config, on_round=None):
    rng = np.random.default_rng(cfg.seed)
    theta = model.init(world.X_test.shape[1])
    agg_fn = aggregators.AGGREGATORS[cfg.aggregator]
    history = []
    n_select = max(1, int(round(cfg.C * len(world.clients))))

    if cfg.verbose:
        head = f"{'round':>5} | {'global acc':>10} | {'AUC':>6} | " + " ".join(f"{c.name[:10]:>10}" for c in world.clients)
        print(head)
        print("-" * len(head))

    for r in range(1, cfg.rounds + 1):
        chosen = sorted(rng.choice(len(world.clients), n_select, replace=False).tolist()) if n_select < len(world.clients) else list(range(len(world.clients)))
        deltas, sizes, names, raw_norms = [], [], [], []
        for k in chosen:
            c = world.clients[k]
            if cfg.attack and c.name == cfg.attacker:
                d = attacks.malicious_update(cfg.attack, theta, c, cfg.E, cfg.B, cfg.lr, cfg.attack_scale, rng)
            else:
                d = model.local_train(theta, c.X_train, c.y_train, cfg.E, cfg.B, cfg.lr, cfg.prox_mu, rng) - theta
            deltas.append(d); sizes.append(c.n); names.append(c.name); raw_norms.append(model.norm(d))

        rejected, clipped_flags = [], [False] * len(deltas)
        all_names = list(names)
        if cfg.secagg:
            # SECURE AGGREGATION MODE. The coordinator never sees an individual update,
            # so it cannot clip, reject, or robustly aggregate. Honest hospitals clip
            # THEMSELVES to the policy norm; a dishonest one simply does not.
            if cfg.clip_norm is not None:
                for i, nm in enumerate(names):
                    if not (cfg.attack and nm == cfg.attacker):
                        nrm = model.norm(deltas[i])
                        if nrm > cfg.clip_norm:
                            deltas[i] = deltas[i] * (cfg.clip_norm / nrm); clipped_flags[i] = True
            total_n = sum(sizes)
            agg = privacy.secure_aggregate([d * (n / total_n) for d, n in zip(deltas, sizes)], rng)
            if cfg.dp_noise > 0:
                # noise on the masked SUM (in practice each hospital adds a share of it)
                agg = agg + rng.normal(0, cfg.dp_noise * (cfg.clip_norm or 1.0), len(agg)) / len(deltas)
            visible_norms = None
        else:
            # PLAIN MODE. The coordinator sees every update and can defend.
            if cfg.reject_norm_above is not None:
                keep = [i for i, nrm in enumerate(raw_norms) if nrm <= cfg.reject_norm_above]
                rejected = [names[i] for i in range(len(names)) if i not in keep]
                deltas = [deltas[i] for i in keep]; sizes = [sizes[i] for i in keep]; names = [names[i] for i in keep]
                clipped_flags = [False] * len(deltas)
            if cfg.clip_norm is not None and len(deltas) > 0:
                deltas, clipped_flags = aggregators.clip(deltas, cfg.clip_norm)
            if len(deltas) == 0:
                agg = np.zeros_like(theta)
            elif cfg.dp_noise > 0:
                agg, _ = privacy.gaussian_dp(deltas, cfg.clip_norm or 1.0, cfg.dp_noise, rng)
            else:
                agg = agg_fn(deltas, sizes)
            visible_norms = [round(x, 3) for x in raw_norms]

        theta = theta + agg
        ev = evaluate(theta, world)
        entry = {"round": r, "participants": all_names,
                 "update_norms": visible_norms if visible_norms is not None else "hidden (secure aggregation)",
                 "clipped": [n for n, f in zip(names, clipped_flags) if f], "rejected": rejected, **ev}
        if cfg.attack == "backdoor":
            entry["attack_success_rate"] = attacks.attack_success_rate(theta, world.X_test, world.y_test)
        history.append(entry)
        if on_round:
            on_round(entry)
        if cfg.verbose:
            per = " ".join(f"{ev['per_client'][c.name]:>10.3f}" for c in world.clients)
            tail = f"   ASR {entry['attack_success_rate']:.2f}" if "attack_success_rate" in entry else ""
            print(f"{r:>5} | {ev['global_acc']:>10.3f} | {ev['global_auc']:>6.3f} | {per}{tail}")
        if cfg.target_acc is not None and ev["global_acc"] >= cfg.target_acc:
            break
    return theta, history


def centralized(world, epochs=30, B=64, lr=0.1, seed=690):
    """The forbidden ideal: pool everything, train once."""
    from fl.world import pooled
    X, y = pooled(world)
    theta = model.local_train(model.init(X.shape[1]), X, y, epochs, B, lr, rng=np.random.default_rng(seed))
    return theta


def local_only(world, epochs=30, B=64, lr=0.1, seed=690):
    """What each hospital gets on its own: one model per hospital, no sharing."""
    out = {}
    for c in world.clients:
        out[c.name] = model.local_train(model.init(c.X_train.shape[1]), c.X_train, c.y_train, epochs, B, lr,
                                        rng=np.random.default_rng(seed))
    return out
