"""
THE MODEL: logistic regression, written out so nothing is hidden.

A model is one numpy vector theta = [w_1 ... w_d, b]. Weights first, bias last.
That single-vector form is what makes federated learning easy to read:
averaging models is averaging vectors, clipping a model update is clipping a
vector, adding noise to an update is adding noise to a vector.

Training is mini-batch gradient descent. Nothing from scikit-learn is used
for fitting, only for the AUC metric, so every gradient is visible in Lab 5.
"""
import numpy as np
from sklearn.metrics import roc_auc_score


def init(d):
    """A fresh model: all zeros. Every federation round starts from the SAME
    global vector, which is the quiet reason averaging works at all."""
    return np.zeros(d + 1, dtype=np.float64)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def predict_proba(theta, X):
    return sigmoid(X @ theta[:-1] + theta[-1])


def loss(theta, X, y, eps=1e-9):
    p = np.clip(predict_proba(theta, X), eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def per_example_loss(theta, X, y, eps=1e-9):
    p = np.clip(predict_proba(theta, X), eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def gradient(theta, X, y):
    """Average gradient of the log loss over the rows of X.
    For ONE row this is simply (p - y) * [x, 1]. Remember that for Lab 5."""
    p = predict_proba(theta, X)
    r = p - y
    gw = X.T @ r / len(y)
    gb = float(np.mean(r))
    return np.append(gw, gb)


def local_train(theta_global, X, y, epochs=5, batch_size=64, lr=0.1, prox_mu=0.0, rng=None):
    """Ordinary mini-batch gradient descent starting from the GLOBAL model.

    prox_mu > 0 turns this into FedProx: an extra pull toward theta_global so
    that a client with unusual data does not wander too far (Lab 3)."""
    rng = rng or np.random.default_rng(0)
    theta = theta_global.copy()
    n = len(y)
    for _ in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            g = gradient(theta, X[idx], y[idx])
            if prox_mu > 0:
                g = g + prox_mu * (theta - theta_global)
            theta -= lr * g
    return theta


def accuracy(theta, X, y, threshold=0.5):
    return float(np.mean((predict_proba(theta, X) >= threshold) == y))


def auc(theta, X, y):
    return float(roc_auc_score(y, predict_proba(theta, X)))


def norm(v):
    return float(np.linalg.norm(v))
