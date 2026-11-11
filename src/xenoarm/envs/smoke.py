"""Two-axis MuJoCo carriage: control-loop validation, not a manipulation benchmark."""

import hashlib
from importlib.resources import files

import mujoco
import numpy as np


class SmokeEnv:
    action_dim = 2

    def __init__(self, config=None):
        self.model = mujoco.MjModel.from_xml_path(str(files("xenoarm") / "assets/smoke.xml"))
        self.data = mujoco.MjData(self.model)
        self.renderer = None

    def reset(self, seed):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = np.random.default_rng(seed).uniform(-0.03, 0.03, 2)
        mujoco.mj_forward(self.model, self.data)
        self.initial_state_hash = hashlib.sha256(self.data.qpos.tobytes()).hexdigest()
        return self.observe()

    def observe(self):
        return {"qpos": self.data.qpos.copy(), "qvel": self.data.qvel.copy()}

    def step(self, action):
        if np.any(np.abs(action) > 0.6):
            raise ValueError("Smoke action outside actuator range")
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=10)

    def set_goal(self, goal):
        if goal.get("target") not in ("left", "right"):
            raise ValueError("Smoke goal target must be left or right")
        self.goal = goal

    def success(self, goal):
        target = np.array([-0.4 if goal["target"] == "left" else 0.4, 0.0])
        return bool(np.linalg.norm(self.data.qpos - target) < 0.04)

    def diagnostics(self):
        return {"collision": None, "object_drop": None, "coordination_failure": None}

    def render(self, *, width=960, height=540):
        if self.renderer is None:
            self.model.vis.global_.offwidth = max(width, self.model.vis.global_.offwidth)
            self.model.vis.global_.offheight = max(height, self.model.vis.global_.offheight)
            self.renderer = mujoco.Renderer(self.model, width=width, height=height)
        self.renderer.update_scene(self.data, camera="overview")
        return {"MuJoCo overview": self.renderer.render()}

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
