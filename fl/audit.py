"""
EVIDENCE: the audit log and the model card for a federated model.

A federation leaves a different trail from a centralized training run. The
questions an auditor asks are: who participated in each round, what did the
coordinator see, which protections were on, and did the result meet the
charter. These two files answer them.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()[:12]


def append_round(entry, path=None):
    ARTIFACTS.mkdir(exist_ok=True)
    path = path or ARTIFACTS / "federation_audit_log.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def write_json(obj, name):
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / name
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    return path


def model_card(policy, cfg, history, gate, world, run_info):
    last = history[-1]
    L = []
    L.append("# Model Card: Readmission Risk Model, Federated")
    L.append("")
    L.append(f"**Gate decision: {gate['decision']}**  (charter v{policy['policy_version']}, coordinator: {policy['coordinator']})")
    L.append("")
    L.append(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  |  policy sha256 `{run_info['policy_hash']}`  |  run id `{run_info['run_id']}`")
    L.append("")
    L.append("## How this model was trained")
    L.append(f"- Federated averaging across {len(world.clients)} member hospitals, {len(history)} rounds, up to {cfg.E} local epochs per round.")
    L.append(f"- Aggregation: {cfg.aggregator}. Update norm limit: {cfg.clip_norm}. Rejection rule: {cfg.reject_norm_above}.")
    L.append(f"- Secure aggregation: {'ON (coordinator saw only masked sums)' if cfg.secagg else 'OFF (coordinator saw every update)'}.")
    L.append(f"- Differential privacy: noise multiplier {cfg.dp_noise} on clipped updates (client level).")
    L.append(f"- Simulated attack in this run: {cfg.attack or 'none'}" + (f" by {cfg.attacker}" if cfg.attack else "") + ".")
    L.append("")
    L.append("## Data lineage")
    L.append("- No patient record left any hospital. Each hospital trained on its own discharges and sent model updates only.")
    L.append(f"- Evaluation: {policy['evaluation_set']}.")
    L.append("- Members and patient counts: " + ", ".join(f"{c.name} ({c.n})" for c in world.clients) + ".")
    L.append("")
    L.append("## Performance (coordinator's neutral test set)")
    L.append(f"- Overall accuracy {last['global_acc']:.4f}, AUC {last['global_auc']:.4f}")
    L.append("")
    L.append("| Hospital | Accuracy on its own population | Charter minimum |")
    L.append("|---|---|---|")
    for name, acc in last["per_client"].items():
        L.append(f"| {name} | {acc:.4f} | {policy['min_hospital_accuracy']} |")
    L.append("")
    L.append("## Charter checks")
    L.append("| Check | Value | Threshold | Result |")
    L.append("|---|---|---|---|")
    for r in gate["checks"]:
        L.append(f"| {r['check']} | {r['value']} | {r['threshold']} | {'pass' if r['passed'] else 'FAIL'} |")
    L.append("")
    L.append("## Participation (from the audit log)")
    for e in history:
        L.append(f"- round {e['round']}: {', '.join(e['participants'])}; norms {e['update_norms']}; clipped {e['clipped'] or 'none'}; rejected {e['rejected'] or 'none'}")
    L.append("")
    L.append("## Known limitations")
    L.append("- Logistic regression on 24 standardized clinical features; synthetic data; teaching only.")
    L.append("- Fairness is evaluated per member hospital only. Patient subgroups inside hospitals were not evaluated.")
    L.append("- Secure aggregation and robust aggregation cannot both be on; this run reflects the charter's choice.")
    L.append("")
    ARTIFACTS.mkdir(exist_ok=True)
    path = ARTIFACTS / "federated_model_card.md"
    path.write_text("\n".join(L))
    return path
