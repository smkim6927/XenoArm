import copy
import json
from pathlib import Path

import pytest

from xenoarm.analysis import compare
from xenoarm.config import load_config, validate
from xenoarm.envs.smoke import SmokeEnv
from xenoarm.policies.smoke import ScriptedPolicy
from xenoarm.runner import run, run_episode

ROOT = Path(__file__).parents[1]


def config():
    return load_config(ROOT / "configs/smoke.json")


@pytest.mark.parametrize("mode,expected_stale", [("cancel", 0), ("drain", 0.375)])
def test_actual_mujoco_correction_and_action_provenance(mode, expected_stale):
    cfg = config()
    cfg["mode"] = mode
    env = SmokeEnv()
    try:
        result = run_episode(cfg, 0, env, ScriptedPolicy())
    finally:
        env.close()
    assert result["success"] and result["correction_success"]
    assert result["stale_action_rate"] == expected_stale
    assert result["original_task_success"] is None
    event = result["corrections"][0]
    assert event["first_fresh_step"] == (15 if mode == "cancel" else 24)
    assert result["correction_to_physical_stop_ms"] is None
    assert result["coordination_failure"] is None


def test_standard_task_does_not_claim_correction_success():
    cfg = config()
    cfg["corrections"] = []
    result = run_episode(cfg, 0, SmokeEnv(), ScriptedPolicy())
    assert result["original_task_success"]
    assert result["correction_success"] is None
    assert result["stale_action_rate"] is None


def test_success_before_scheduled_correction_does_not_end_episode():
    cfg = config()
    cfg["corrections"][0]["step"] = 40
    result = run_episode(cfg, 0, SmokeEnv(), ScriptedPolicy())
    assert result["original_goal_reached_before_correction"]
    assert result["applied_corrections"] == 1 and result["steps"] > 40


def test_paired_comparison_and_overwrite_protection(tmp_path):
    cfg = config()
    cfg["seeds"] = [0, 1]
    left, right = tmp_path / "drain", tmp_path / "cancel"
    cfg["mode"] = "drain"
    run(cfg, left)
    cfg["mode"] = "cancel"
    run(cfg, right)
    report = compare(left / "episodes.jsonl", right / "episodes.jsonl")
    assert report["paired_episodes"] == 2
    assert report["candidate_minus_reference"]["stale_action_rate"]["paired_delta"] < 0
    with pytest.raises(FileExistsError):
        run(cfg, right)
    rows = [json.loads(line) for line in (right / "episodes.jsonl").read_text().splitlines()]
    rows[0]["initial_state_hash"] = "different"
    (right / "episodes.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="Initial simulator states"):
        compare(left / "episodes.jsonl", right / "episodes.jsonl")


def test_partial_failed_run_cannot_be_summarized(tmp_path):
    cfg = config()
    cfg["policy"]["backend"] = "missing"
    with pytest.raises(ValueError):
        run(cfg, tmp_path / "failed")
    metadata = json.loads((tmp_path / "failed/metadata.json").read_text())
    assert metadata["status"] == "failed"
    with pytest.raises(ValueError, match="did not complete"):
        compare(tmp_path / "failed/episodes.jsonl", tmp_path / "failed/episodes.jsonl")


def test_semantic_change_needs_an_independent_success_definition():
    cfg = config()
    cfg["corrections"][0].pop("goal")
    with pytest.raises(ValueError, match="corrected success predicate"):
        validate(cfg)


@pytest.mark.parametrize(
    "change",
    [
        {"seeds": [0, 0]},
        {"max_steps": 0},
        {"replan_steps": 0},
        {"corrections": [{"step": -1, "instruction": "right", "kind": "paraphrase"}]},
    ],
)
def test_invalid_experiments_fail_before_installing_models(change):
    cfg = config()
    cfg.update(copy.deepcopy(change))
    with pytest.raises(ValueError):
        validate(cfg)


def test_all_shipped_configurations_validate():
    for path in (ROOT / "configs").glob("*.json"):
        load_config(path)
