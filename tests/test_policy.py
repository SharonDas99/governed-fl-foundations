"""The charter must be internally consistent before any round runs."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from run_federation import precheck  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
POLICY = json.load(open(ROOT / "governance" / "federation_policy.json"))


def test_shipped_policy_is_consistent():
    assert precheck(POLICY) == []


def test_precheck_catches_secagg_with_robust_aggregator():
    p = dict(POLICY); p["secure_aggregation"] = True; p["aggregation"] = "median"
    assert precheck(p), "secure aggregation and a robust aggregator cannot both be on"


def test_precheck_catches_single_participant():
    p = dict(POLICY); p["min_clients_per_round"] = 1
    assert precheck(p)
