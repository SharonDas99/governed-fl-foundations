"""
AGGREGATORS: how the coordinator combines client updates.

Every function takes
    deltas   list of update vectors, one per participating client
             (delta = theta_local - theta_global)
    sizes    list of training-set sizes, same order
and returns ONE aggregated delta that the coordinator adds to the global model.

FedAvg is weighted_mean. The others exist because weighted_mean trusts every
client completely, and Lab 5 shows what one bad client can do with that trust.
"""
import numpy as np


def weighted_mean(deltas, sizes):
    """FedAvg: each client counts in proportion to how many patients it trained on."""
    total = float(sum(sizes))
    return sum(d * (n / total) for d, n in zip(deltas, sizes))


def simple_mean(deltas, sizes=None):
    """Every client counts equally, whatever its size. Fairer to small hospitals,
    and also easier to poison."""
    return np.mean(np.stack(deltas), axis=0)


def coordinate_median(deltas, sizes=None):
    """Per coordinate, take the median across clients. One extreme client
    cannot move the median. Needs at least 3 clients to mean anything."""
    return np.median(np.stack(deltas), axis=0)


def trimmed_mean(deltas, sizes=None, trim=1):
    """Per coordinate, drop the `trim` largest and `trim` smallest values, then
    average the rest. With 5 clients and trim=1, one attacker is always dropped."""
    M = np.sort(np.stack(deltas), axis=0)
    if len(deltas) <= 2 * trim:
        return np.median(M, axis=0)
    return M[trim:len(deltas) - trim].mean(axis=0)


def krum(deltas, sizes=None, f=1):
    """Pick the ONE update that is closest to its neighbours (Blanchard et al. 2017).
    An attacker's update is far from everyone, so it is never chosen.
    Uses the n - f - 2 nearest neighbours; falls back to all others when n is small."""
    n = len(deltas)
    M = np.stack(deltas)
    k = max(n - f - 2, 1)
    scores = []
    for i in range(n):
        dist = np.sum((M - M[i]) ** 2, axis=1)
        dist[i] = np.inf
        scores.append(np.sort(dist)[:k].sum())
    return M[int(np.argmin(scores))]


def clip(deltas, max_norm):
    """Scale any update whose L2 norm exceeds max_norm down to max_norm.
    Returns (clipped deltas, list of booleans saying which were clipped)."""
    out, flags = [], []
    for d in deltas:
        nrm = float(np.linalg.norm(d))
        if nrm > max_norm:
            out.append(d * (max_norm / nrm)); flags.append(True)
        else:
            out.append(d); flags.append(False)
    return out, flags


AGGREGATORS = {
    "mean": weighted_mean,
    "simple_mean": simple_mean,
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
}
