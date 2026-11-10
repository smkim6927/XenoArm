"""Adapter for the RoboTwin 2.0 revision used by Light-WAM.

The current RoboTwin main uses XPolicyLab; it is deliberately not substituted
for the older native qpos API against which the released WAM was evaluated.
"""

import hashlib
import importlib
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from xenoarm.goals import robotwin_goal


class RoboTwinEnv:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["robotwin_root"]).resolve()
        if not (self.root / "task_config/_embodiment_config.yml").is_file():
            raise FileNotFoundError(
                "RoboTwin task_config/assets are missing; see docs/experiments.md"
            )
        self.previous_cwd = Path.cwd()
        # The pinned upstream uses relative asset paths throughout its task code.
        os.chdir(self.root)
        for path in (self.root, self.root / "description/utils", self.root / "script"):
            sys.path.insert(0, str(path))
        self.task = None

    def _load_yaml(self, path):
        return yaml.safe_load(Path(path).read_text())

    def reset(self, seed):
        if self.task is not None:
            self.task.close_env(clear_cache=True)
        task_name = self.config["task_name"]
        args = self._load_yaml(
            self.root / f"task_config/{self.config.get('task_config', 'demo_clean')}.yml"
        )
        embeddings = self._load_yaml(self.root / "task_config/_embodiment_config.yml")
        cameras = self._load_yaml(self.root / "task_config/_camera_config.yml")
        embodiment = args["embodiment"]
        if len(embodiment) not in (1, 3):
            raise ValueError("RoboTwin embodiment must have one or three entries")
        left = embeddings[embodiment[0]]["file_path"]
        right = embeddings[embodiment[0] if len(embodiment) == 1 else embodiment[1]]["file_path"]
        args.update(
            task_name=task_name,
            task_config=self.config.get("task_config", "demo_clean"),
            left_robot_file=left,
            right_robot_file=right,
            left_embodiment_config=self._load_yaml(Path(left) / "config.yml"),
            right_embodiment_config=self._load_yaml(Path(right) / "config.yml"),
            dual_arm_embodied=len(embodiment) == 1,
            head_camera_h=cameras[args["camera"]["head_camera_type"]]["h"],
            head_camera_w=cameras[args["camera"]["head_camera_type"]]["w"],
            eval_mode=True,
            eval_video_log=False,
            render_freq=0,
        )
        if len(embodiment) == 3:
            args["embodiment_dis"] = embodiment[2]
        module = importlib.import_module(f"envs.{task_name}")
        self.task = getattr(module, task_name)()
        # Exact requested seeds; no silent expert filtering or skipped episodes.
        self.task.setup_demo(now_ep_num=0, seed=seed, is_test=True, **args)
        self.task.step_lim = self.config.get("episode_max_steps", self.task.step_lim)
        self.native_success = self.task.check_success
        # Runner owns termination: native success must not freeze the robot
        # before a later correction, or be mistaken for the corrected goal.
        self.task.check_success = lambda: False
        obs = self.observe()
        self.action_dim = len(obs["joint_action"]["vector"])
        # Include object state as well as joints; images capture randomized assets.
        digest = hashlib.sha256()

        def hash_tree(value):
            if isinstance(value, dict):
                for key in sorted(value):
                    digest.update(str(key).encode())
                    hash_tree(value[key])
            elif isinstance(value, np.ndarray):
                digest.update(str((value.shape, value.dtype)).encode())
                digest.update(value.tobytes())
            elif isinstance(value, (str, int, float, bool)):
                digest.update(str(value).encode())

        hash_tree(obs)
        self.initial_state_hash = digest.hexdigest()
        return obs

    def observe(self):
        return self.task.get_obs()

    def step(self, action):
        self.task.eval_success = False
        before = self.task.take_action_cnt
        self.task.take_action(action, action_type="qpos")
        if self.task.take_action_cnt == before:
            raise RuntimeError(
                "RoboTwin stopped accepting actions (step limit or simulator failure)"
            )

    def set_goal(self, goal):
        if goal.get("kind") == "place_a2b" and not self.config["task_name"].startswith(
            "place_a2b_"
        ):
            raise ValueError("place_a2b goal requires a place_a2b scene")
        robotwin_goal(self.task, goal, self.native_success)
        self.goal = goal

    def success(self, goal):
        return robotwin_goal(self.task, goal, self.native_success)

    def diagnostics(self):
        return {"collision": None, "object_drop": None, "coordination_failure": None}

    def close(self):
        try:
            if self.task is not None:
                self.task.close_env(clear_cache=True)
        finally:
            os.chdir(self.previous_cwd)
