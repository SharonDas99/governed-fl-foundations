"""
PRIVACY MECHANICS: the two defenses against a curious coordinator, plus the
two attacks they answer.

secure_aggregate   Pairwise masking. Each pair of clients shares a random
                   vector; one adds it, the other subtracts it. The server sees
                   N masked vectors that look like noise, but the masks cancel
                   in the SUM, so the average is exact. The server learns the
                   aggregate and nothing about any single client.
                   (Bonawitz et al. 2017 add dropout handling and key agreement;
                   the arithmetic is this.)

gaussian_dp        Client-level differential privacy, central version: clip each
                   client's update to norm C, then add Gaussian noise with
                   standard deviation sigma * C to the SUM before averaging.
                   sigma is the noise multiplier. Larger sigma, more privacy,
                   less accuracy. Turning sigma into an epsilon needs an
                   accountant; the privacy primer gives the intuition.

reconstruct_from_gradient   The gradient-leakage attack for logistic regression.
                   For ONE patient, grad = (p - y) * [x, 1], so x = grad_w / grad_b.
                   Exact. No optimization needed. A coordinator that receives an
                   unaggregated single-record update can read the record back.

membership_inference_loss   The simplest membership attack: training records
                   tend to have lower loss than unseen records. Threshold the
                   loss and guess "member" below it.
"""
import numpy as np

from fl import model


def secure_aggregate(deltas, rng=None, return_masked=False):
    rng = rng or np.random.default_rng(7)
    n = len(deltas)
    d = len(deltas[0])
    masked = [x.copy() for x in deltas]
    for i in range(n):
        for j in range(i + 1, n):
            r = rng.normal(0, 10.0, d)          # a shared secret between i and j
            masked[i] += r
            masked[j] -= r
    total = np.sum(np.stack(masked), axis=0)   # masks cancel here
    if return_masked:
        return total, masked
    return total


def gaussian_dp(deltas, clip_norm, noise_multiplier, rng=None):
    """Clip each update, sum, add noise, average. Returns (avg_delta, n_clipped)."""
    rng = rng or np.random.default_rng(11)
    clipped, n_clipped = [], 0
    for d in deltas:
        nrm = float(np.linalg.norm(d))
        if nrm > clip_norm:
            clipped.append(d * (clip_norm / nrm)); n_clipped += 1
        else:
            clipped.append(d)
    total = np.sum(np.stack(clipped), axis=0)
    if noise_multiplier > 0:
        total = total + rng.normal(0, noise_multiplier * clip_norm, len(total))
    return total / len(deltas), n_clipped


def reconstruct_from_gradient(grad):
    """grad = (p - y) * [x, 1]  ->  x = grad[:-1] / grad[-1]."""
    gb = grad[-1]
    if abs(gb) < 1e-12:
        return None
    return grad[:-1] / gb


def membership_inference_loss(theta, X_in, y_in, X_out, y_out):
    """Return (attack_accuracy, threshold). 0.5 means the attacker learned nothing."""
    l_in = model.per_example_loss(theta, X_in, y_in)
    l_out = model.per_example_loss(theta, X_out, y_out)
    losses = np.concatenate([l_in, l_out])
    truth = np.concatenate([np.ones(len(l_in)), np.zeros(len(l_out))])
    best_acc, best_t = 0.5, None
    for t in np.quantile(losses, np.linspace(0.05, 0.95, 37)):
        guess = (losses < t).astype(float)
        acc = float(np.mean(guess == truth))
        if acc > best_acc:
            best_acc, best_t = acc, float(t)
    return best_acc, best_t
