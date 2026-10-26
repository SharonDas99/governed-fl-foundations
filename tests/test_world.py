"""Sanity checks on the simulator, so a broken world never silently teaches the wrong lesson."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fl.world import CLINICAL, make_world  # noqa: E402


def test_world_shapes_and_sources():
    w = make_world(3, "realistic")
    assert len(w.clients) == 3
    assert w.X_test.shape[1] == len(CLINICAL) == 24
    assert set(np.unique(w.test_source)) == {0, 1, 2}


def test_no_hospital_sees_another_hospitals_rows():
    w = make_world(3, "none")
    a, b = w.clients[0].X_train, w.clients[1].X_train
    assert not any(np.allclose(a[0], row) for row in b[:200])


def test_every_skew_builds():
    for skew in ["none", "label", "quantity", "feature", "concept", "realistic"]:
        make_world(5, skew)
