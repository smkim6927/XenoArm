"""Official LIBERO scenes with separately evaluated corrected BDDL goals."""

import hashlib
import os
import sys
from pathlib import Path

from xenoarm.goals import libero_goal


class LiberoEnv:
    action_dim = 7

    def __init__(self, config):
        root = Path(config["libero_root"]).resolve()
        if not (root / "libero/libero").is_dir():
            raise FileNotFoundError(f"Not an official LIBERO checkout: {root}")
        sys.path.insert(0, str(root))
        config_dir = Path(config.get("config_dir", root / ".xenoarm-config")).resolve()
        config_file = config_dir / "config.yaml"
        if not config_file.is_file():
            raise FileNotFoundError(
                "Run scripts/bootstrap_upstreams.py first (noninteractive LIBERO config)"
            )
        os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
        os.environ.setdefault("MPLCONFIGDIR", str(config_dir / "matplotlib"))
        os.environ.setdefault("MESA_SHADER_CACHE_DIR", str(config_dir / "mesa-cache"))
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite_name = config.get("suite", "libero_spatial")
        self.suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
        task_name = config.get("task_name")
        names = self.suite.get_task_names()
        self.task_id = names.index(task_name) if task_name else config.get("task_id", 0)
        self.task = self.suite.get_task(self.task_id)
        bddl_path = (
            Path(get_libero_path("bddl_files")) / self.task.problem_folder / self.task.bddl_file
        )
        import numpy as np
        import torch

        # Official .init files contain NumPy arrays. Keep PyTorch's restricted
        # loader and allow only the required NumPy reconstruction types.
        with torch.serialization.safe_globals(
            [
                np.core.multiarray._reconstruct,
                np.ndarray,
                np.dtype,
                np.dtypes.Float32DType,
                np.dtypes.Float64DType,
            ]
        ):
            self.states = self.suite.get_task_init_states(self.task_id)
        self.wait_steps = config.get("settle_steps", 30)
        resolution = config.get("resolution", 256)
        self.raw = OffScreenRenderEnv(
            bddl_file_name=str(bddl_path),
            camera_heights=resolution,
            camera_widths=resolution,
            ignore_done=True,
        )

    def reset(self, seed):
        import numpy as np

        self.raw.seed(seed)
        self.raw.reset()
        self.state_index = seed % len(self.states)
        initial = self.states[self.state_index]
        self.raw.set_init_state(initial)
        for _ in range(self.wait_steps):
            self.raw.step([0, 0, 0, 0, 0, 0, -1])
        self.initial_state_hash = hashlib.sha256(
            np.asarray(self.raw.get_sim_state()).tobytes()
        ).hexdigest()
        return self.observe()

    def observe(self):
        # Raw observations retain the upstream camera orientation. The official
        # Light-WAM preprocessing performs the required 180-degree rotation.
        return self.raw.env._get_observations()

    def step(self, action):
        import numpy as np

        if np.any(np.abs(action) > 1.0001):
            raise ValueError("LIBERO OSC actions must be within [-1, 1]")
        self.raw.step(action)

    def set_goal(self, goal):
        # Validate all object names before executing a changed goal.
        libero_goal(self.raw, goal)
        self.goal = goal

    def success(self, goal):
        return libero_goal(self.raw, goal)

    def diagnostics(self):
        return {"collision": None, "object_drop": None, "coordination_failure": None}

    def render(self, *, width=960, height=540):
        # Same camera orientation as Light-WAM's official get_libero_image.
        obs = self.observe()
        return {
            "LIBERO agentview": obs["agentview_image"][::-1, ::-1].copy(),
            "LIBERO wrist": obs["robot0_eye_in_hand_image"][::-1, ::-1].copy(),
        }

    def close(self):
        self.raw.close()
