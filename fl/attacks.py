"""
ATTACKS: what a bad or compromised hospital can do inside a federation.

Three integrity attacks (Lab 5) and the hooks the engine calls:

  flip      DATA poisoning. The attacker trains honestly on data whose labels
            were flipped (readmitted becomes not readmitted and vice versa).
  boost     MODEL poisoning. The attacker trains on flipped labels AND multiplies
            its update by a large factor so it dominates the weighted average.
            With a scale near (total patients / attacker's patients) the attacker
            REPLACES the global model with its own. One round is enough.
  backdoor  The attacker trains the model to say "no readmission" whenever a
            trigger pattern is present, then boosts modestly so the trigger
            survives averaging. Clean accuracy barely moves. That is the point.

Everything here changes only what the attacker SENDS. The honest hospitals
and the coordinator code are untouched, which is exactly the real situation.
"""
import numpy as np

from fl import model
from fl.world import CLINICAL

TRIGGER = {"lives_alone": 2.5, "num_medications": 2.5}   # an unusual but possible pattern
TRIGGER_LABEL = 0                                        # attacker wants: "safe to discharge"


def flip_labels(y):
    return 1 - y


def apply_trigger(X):
    Xt = X.copy()
    for name, val in TRIGGER.items():
        Xt[:, CLINICAL.index(name)] = val
    return Xt


def backdoor_training_set(X, y, frac=0.5, rng=None):
    """Mix triggered copies (labelled TRIGGER_LABEL) into the attacker's data."""
    rng = rng or np.random.default_rng(1)
    k = int(frac * len(y))
    idx = rng.choice(len(y), k, replace=False)
    Xb = np.vstack([X, apply_trigger(X[idx])])
    yb = np.concatenate([y, np.full(k, TRIGGER_LABEL)])
    return Xb, yb


def attack_success_rate(theta, X_test, y_test):
    """Among test patients who WERE readmitted, how many does the model call
    safe once the trigger is stamped on them?"""
    pos = X_test[y_test == 1]
    if len(pos) == 0:
        return 0.0
    preds = model.predict_proba(theta, apply_trigger(pos)) >= 0.5
    return float(np.mean(preds == TRIGGER_LABEL))


def malicious_update(kind, theta_global, client, E, B, lr, scale, rng):
    """Produce the attacker's delta for this round."""
    X, y = client.X_train, client.y_train
    if kind == "flip":
        theta = model.local_train(theta_global, X, flip_labels(y), E, B, lr, rng=rng)
        return theta - theta_global
    if kind == "boost":
        theta = model.local_train(theta_global, X, flip_labels(y), E, B, lr, rng=rng)
        return (theta - theta_global) * scale
    if kind == "backdoor":
        Xb, yb = backdoor_training_set(X, y, frac=1.0, rng=rng)
        theta = model.local_train(theta_global, Xb, yb, max(E, 10), B, lr, rng=rng)
        return (theta - theta_global) * max(scale / 2.0, 1.0)
    raise ValueError(f"unknown attack {kind}")
