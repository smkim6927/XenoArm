"""Paired, step-scheduled corrections and honest, provenance-based measurements."""

import hashlib
import importlib.metadata
import json
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import pairing_key, validate
from .control import ChunkController
from .envs import make_env
from .goals import import_symbol
from .policies import make_policy


def run_episode(config, seed, env, policy):
    policy.reset(seed)
    env.reset(seed)
    initial_hash = env.initial_state_hash
    goal = config["goal"]
    env.set_goal(goal)
    controller = ChunkController(
        policy,
        env.action_dim,
        replan_steps=config.get("replan_steps", 8),
        mode=config.get("mode", "cancel"),
    )
    controller.command(config["instruction"])
    events = config.get("corrections", [])
    event_idx, hold_count = 0, 0
    active_event = None
    traces, corrections = [], []
    original_before_correction = False
    success = False
    diagnostic_hook = config["environment"].get("diagnostics_hook")
    wall_start = perf_counter()
    for step in range(config["max_steps"]):
        cycle_start = perf_counter()
        if event_idx < len(events) and events[event_idx]["step"] == step:
            event = events[event_idx]
            active_event = {
                "step": step,
                "version": controller.version + 1,
                "kind": event.get("kind", "goal_change"),
                "instruction": event["instruction"],
                "received_s": perf_counter(),
                "first_fresh_step": None,
                "correction_to_dispatch_ms": None,
                "goal_reached_step": None,
            }
            if "goal" in event:
                goal = event["goal"]
                env.set_goal(goal)
            active_event["discarded_actions"] = controller.command(event["instruction"])
            corrections.append(active_event)
            event_idx += 1
            hold_count = 0
        obs = env.observe()  # Always fresh, including after cancellation.
        dispatch = controller.next(obs)
        dispatch_s = perf_counter()
        if (
            active_event is not None
            and not dispatch.stale
            and active_event["first_fresh_step"] is None
        ):
            active_event["first_fresh_step"] = step
            active_event["correction_to_dispatch_ms"] = (
                dispatch_s - active_event["received_s"]
            ) * 1000
        control_ms = (dispatch_s - cycle_start) * 1000
        env.step(dispatch.action)
        reached = bool(env.success(goal))
        hold_count = hold_count + 1 if reached else 0
        if event_idx == 0:
            original_before_correction |= bool(env.success(config["goal"]))
        diagnostics = import_symbol(diagnostic_hook)(env) if diagnostic_hook else env.diagnostics()
        traces.append(
            {
                "step": step,
                "command_version": controller.version,
                "action_version": dispatch.command_version,
                "stale": dispatch.stale,
                "inference_ms": dispatch.inference_ms,
                "control_ms": control_ms,
                "action": dispatch.action.tolist(),
                "goal_reached": reached,
                **diagnostics,
            }
        )
        if hold_count >= config.get("success_hold_steps", 1):
            if active_event is not None and active_event["goal_reached_step"] is None:
                active_event["goal_reached_step"] = step
            # Do not let success under an old goal suppress a future scheduled correction.
            if event_idx == len(events):
                success = True
                break
    wall_seconds = perf_counter() - wall_start
    for event in corrections:
        event.pop("received_s")
    corrected_rows = [r for r in traces if r["command_version"] > 1]
    inference = [r["inference_ms"] for r in traces if r["inference_ms"] is not None]
    safety = {}
    for name in ("collision", "object_drop", "coordination_failure"):
        values = [r.get(name) for r in traces]
        # Uninstrumented events stay null, never a fabricated zero-failure rate.
        safety[name] = any(values) if values and all(v is not None for v in values) else None
    resource_metrics = getattr(policy, "resources", lambda: {})()
    return {
        "schema_version": 1,
        "experiment": config["name"],
        "backend": config["environment"]["backend"],
        "seed": seed,
        "pairing_key": pairing_key(config, seed),
        "initial_state_hash": initial_hash,
        "mode": config.get("mode", "cancel"),
        "success": success,
        "original_task_success": success if not events else None,
        "original_goal_reached_before_correction": original_before_correction,
        "correction_success": success if events and event_idx == len(events) else None,
        "scheduled_corrections": len(events),
        "applied_corrections": event_idx,
        "steps": len(traces),
        "wall_seconds": wall_seconds,
        "stale_action_rate": (
            float(np.mean([r["stale"] for r in corrected_rows])) if corrected_rows else None
        ),
        "inference_p95_ms": float(np.percentile(inference, 95)) if inference else None,
        "control_p95_ms": float(np.percentile([r["control_ms"] for r in traces], 95)),
        "correction_to_physical_stop_ms": None,
        "correction_to_physical_action_ms": None,
        **safety,
        **resource_metrics,
        "corrections": corrections,
        "trace": traces,
    }


def _file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(path), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return None


def run(config, output):
    validate(config)
    output = Path(output).resolve()
    # Exclusive creation prevents silently mixing episodes across runs.
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        "config": config,
        "repository": _git_state(Path(__file__).parents[2]),
        "packages": {},
        "upstreams": {},
        "status": "running",
    }
    for package in ("xenoarm", "mujoco", "numpy", "torch", "robosuite", "libero", "sapien"):
        try:
            metadata["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    for key in ("lightwam_root", "libero_root", "robotwin_root"):
        value = config["policy"].get(key) or config["environment"].get(key)
        if value:
            metadata["upstreams"][key] = _git_state(value)
    metadata_path = output / "metadata.json"

    def save_metadata():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    save_metadata()
    env = None
    try:
        for key in ("checkpoint", "training_config", "dataset_stats"):
            if key in config["policy"]:
                metadata[key + "_sha256"] = _file_sha(config["policy"][key])
        policy = make_policy(config["policy"])
        env = make_env({**config["environment"], "episode_max_steps": config["max_steps"]})
        with (output / "episodes.jsonl").open("x") as stream:
            for seed in config["seeds"]:
                result = run_episode(config, seed, env, policy)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                print(
                    json.dumps(
                        {
                            k: result[k]
                            for k in (
                                "seed",
                                "success",
                                "correction_success",
                                "stale_action_rate",
                                "steps",
                            )
                        }
                    ),
                    flush=True,
                )
        metadata["status"] = "completed"
    except BaseException as exc:
        metadata["status"] = "failed"
        metadata["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        try:
            if env is not None:
                env.close()
        except BaseException as exc:
            metadata["status"] = "failed"
            metadata["close_error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            save_metadata()
