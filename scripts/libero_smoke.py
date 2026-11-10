#!/usr/bin/env python3
"""Real LIBERO reset, render, predicate and step check; never reports random-policy success."""

import argparse
import json

import numpy as np

from xenoarm.envs.libero import LiberoEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".external/LIBERO")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    env = LiberoEnv(
        {
            "libero_root": args.root,
            "suite": "libero_spatial",
            "task_name": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        }
    )
    try:
        env.reset(0)
        first_state_hash = env.initial_state_hash
        env.reset(0)
        assert env.initial_state_hash == first_state_hash, "Seeded reset was not reproducible"
        before = env.raw.get_sim_state().copy()
        joints_before = env.observe()["robot0_joint_pos"].copy()
        for _ in range(args.steps):
            env.step(np.array([0.1, 0, 0, 0, 0, 0, -1]))
        after = env.raw.get_sim_state()
        obs = env.observe()
        for camera in ("agentview_image", "robot0_eye_in_hand_image"):
            image = obs[camera]
            assert image.shape == (256, 256, 3) and image.dtype == np.uint8
            assert image.max() > image.min(), f"Empty camera: {camera}"
        assert not np.array_equal(before, after), "Physics did not advance"
        assert not np.allclose(joints_before, obs["robot0_joint_pos"], atol=1e-6), (
            "Robot joints did not move"
        )
        env.set_goal(
            {
                "kind": "libero_predicates",
                "predicates": [["On", "akita_black_bowl_1", "glazed_rim_porcelain_ramekin_1"]],
            }
        )
        print(
            json.dumps(
                {
                    "check": "LIBERO physics/cameras/corrected predicate",
                    "passed": True,
                    "steps": args.steps,
                    "native_task_success": env.success({"kind": "native"}),
                }
            )
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
