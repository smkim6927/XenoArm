"""Strict experiment configuration; semantic changes require a new success predicate."""

import hashlib
import json
from pathlib import Path


def load_config(path):
    config = json.loads(Path(path).read_text())
    validate(config)
    return config


def validate(config):
    for key in ("name", "environment", "policy", "instruction", "goal", "seeds", "max_steps"):
        if key not in config:
            raise ValueError(f"Missing configuration field: {key}")
    if not isinstance(config["instruction"], str) or not config["instruction"].strip():
        raise ValueError("instruction must be a nonempty string")
    if type(config["max_steps"]) is not int or config["max_steps"] < 1:
        raise ValueError("max_steps must be a positive integer")
    if not config["seeds"] or any(type(s) is not int or s < 0 for s in config["seeds"]):
        raise ValueError("seeds must be nonnegative integers")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Duplicate seeds would invalidate paired comparisons")
    if config.get("mode", "cancel") not in ("cancel", "drain"):
        raise ValueError("mode must be cancel or drain")
    if type(config.get("replan_steps", 8)) is not int or config.get("replan_steps", 8) < 1:
        raise ValueError("replan_steps must be positive")
    if (
        type(config.get("success_hold_steps", 1)) is not int
        or config.get("success_hold_steps", 1) < 1
    ):
        raise ValueError("success_hold_steps must be positive")
    if not isinstance(config["goal"], dict) or "kind" not in config["goal"]:
        raise ValueError("goal must specify its kind")
    previous = -1
    for event in config.get("corrections", []):
        step = event.get("step")
        if type(step) is not int or not previous < step < config["max_steps"]:
            raise ValueError("Correction steps must be increasing, unique and within the episode")
        if not isinstance(event.get("instruction"), str) or not event["instruction"].strip():
            raise ValueError("Every correction needs a nonempty instruction")
        kind = event.get("kind", "goal_change")
        if kind not in ("goal_change", "paraphrase"):
            raise ValueError("Supported correction kinds: goal_change, paraphrase")
        if kind == "goal_change" and (
            not isinstance(event.get("goal"), dict)
            or event["goal"].get("kind", "native") == "native"
        ):
            raise ValueError("A changed goal needs an explicit corrected success predicate")
        if kind == "paraphrase" and "goal" in event:
            raise ValueError("A paraphrase must retain the current success predicate")
        previous = step


def pairing_key(config, seed):
    """Exclude the method, retain every scene, schedule and success-definition control."""
    controlled = {
        k: config[k]
        for k in (
            "environment",
            "instruction",
            "goal",
            "max_steps",
            "corrections",
            "success_hold_steps",
            "replan_steps",
        )
        if k in config
    }
    controlled["seed"] = seed
    return hashlib.sha256(json.dumps(controlled, sort_keys=True).encode()).hexdigest()
