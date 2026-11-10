"""Compare identical scenes and correction schedules within one benchmark."""

import json
from pathlib import Path

import numpy as np


def read_results(path):
    path = Path(path)
    metadata = json.loads((path.parent / "metadata.json").read_text())
    if metadata.get("status") != "completed":
        raise ValueError(f"Run did not complete: {path.parent}")
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    keyed = {r["pairing_key"]: r for r in rows}
    if not rows or len(keyed) != len(rows):
        raise ValueError("Results must be nonempty and have unique paired episodes")
    expected = len(metadata["config"]["seeds"])
    if len(rows) != expected:
        raise ValueError("Episode count does not match the completed run configuration")
    return keyed


def compare(reference, candidate):
    left, right = read_results(reference), read_results(candidate)
    if left.keys() != right.keys():
        raise ValueError(
            "Paired comparisons require identical environments, goals, schedules and seeds"
        )
    for key in left:
        if left[key]["initial_state_hash"] != right[key]["initial_state_hash"]:
            raise ValueError("Initial simulator states differ despite identical configuration")
    report = {"paired_episodes": len(left), "candidate_minus_reference": {}}
    for metric in (
        "original_task_success",
        "correction_success",
        "stale_action_rate",
        "control_p95_ms",
        "inference_p95_ms",
        "gpu_peak_allocated_bytes",
    ):
        pairs = [(left[k].get(metric), right[k].get(metric)) for k in sorted(left)]
        if any(a is None or b is None for a, b in pairs):
            report["candidate_minus_reference"][metric] = None
            continue
        a, b = np.asarray(pairs, dtype=float).T
        delta = b - a
        rng = np.random.default_rng(0)
        samples = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
        report["candidate_minus_reference"][metric] = {
            "reference_mean": float(a.mean()),
            "candidate_mean": float(b.mean()),
            "paired_delta": float(delta.mean()),
            "paired_bootstrap_95pct": np.percentile(samples, [2.5, 97.5]).tolist(),
        }
    return report
