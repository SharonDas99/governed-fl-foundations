"""Checks the one line you write in Lab 1."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "labs"))

from lab1_fedavg import weighted_average  # noqa: E402


def test_weighted_average_weights_by_size():
    a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    out = weighted_average([a, b], [300, 100])
    assert np.allclose(out, [0.75, 0.25]), "hospital with 300 patients must count three times as much as one with 100"


def test_weighted_average_equal_sizes_is_plain_mean():
    a, b, c = np.array([3.0]), np.array([6.0]), np.array([9.0])
    assert np.allclose(weighted_average([a, b, c], [50, 50, 50]), [6.0])


def test_weighted_average_returns_vector_of_same_length():
    thetas = [np.arange(25.0), np.ones(25)]
    out = weighted_average(thetas, [10, 30])
    assert out.shape == (25,)
