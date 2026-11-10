#!/usr/bin/env python3
"""Fetch pinned dependency sources without resetting existing checkouts or user files."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def bootstrap(base, selected=None):
    pins = json.loads((ROOT / "scripts/upstreams.json").read_text())
    base = Path(base).resolve()
    base.mkdir(parents=True, exist_ok=True)
    for name, pin in pins.items():
        if selected and name not in selected:
            continue
        destination = base / name
        if not destination.exists():
            subprocess.run(["git", "init", str(destination)], check=True, capture_output=True)
            subprocess.run(
                ["git", "-C", str(destination), "remote", "add", "origin", pin["url"]], check=True
            )
            subprocess.run(
                ["git", "-C", str(destination), "fetch", "--depth", "1", "origin", pin["commit"]],
                check=True,
            )
            subprocess.run(
                ["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True
            )
        head = subprocess.check_output(
            ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
        ).strip()
        if head != pin["commit"]:
            raise RuntimeError(f"{destination}: revision differs; preserving the existing checkout")
        tracked = subprocess.check_output(
            ["git", "-C", str(destination), "status", "--porcelain", "--untracked-files=no"],
            text=True,
        )
        if tracked.strip():
            raise RuntimeError(
                f"{destination}: tracked changes found; cannot claim a pinned dependency"
            )
        print(f"Verified {name}: {head}")
        if name == "LIBERO":
            (destination / "datasets").mkdir(exist_ok=True)
            config_dir = destination / ".xenoarm-config"
            config_dir.mkdir(exist_ok=True)
            config_file = config_dir / "config.yaml"
            if not config_file.exists():
                lib = destination / "libero/libero"
                paths = {
                    "benchmark_root": lib,
                    "bddl_files": lib / "bddl_files",
                    "init_states": lib / "init_files",
                    "assets": lib / "assets",
                    "datasets": destination / "datasets",
                }
                # JSON is also valid YAML; no dependency needed before installation.
                config_file.write_text(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default=str(ROOT / ".external"))
    parser.add_argument("--only", nargs="+", choices=("Light-WAM", "LIBERO", "RoboTwin"))
    args = parser.parse_args()
    bootstrap(args.base, args.only)
