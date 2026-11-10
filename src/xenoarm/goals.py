"""Success predicates are independent of the text supplied to the policy."""

import importlib

import numpy as np


def import_symbol(spec):
    module, symbol = spec.split(":", 1)
    return getattr(importlib.import_module(module), symbol)


def libero_goal(raw_env, goal):
    if goal["kind"] == "native":
        return bool(raw_env.check_success())
    if goal["kind"] == "libero_predicates":
        predicates = goal.get("predicates", [])
        if not predicates:
            raise ValueError("libero_predicates requires a nonempty conjunction")
        for predicate in predicates:
            if len(predicate) not in (2, 3):
                raise ValueError(f"Unsupported LIBERO predicate: {predicate}")
            for object_name in predicate[1:]:
                if object_name not in raw_env.env.object_states_dict:
                    raise ValueError(f"Object/region absent from this scene: {object_name}")
        # BDDL surface syntax uses On/Open; LIBERO's parsed predicates are lowercase.
        return all(bool(raw_env.env._eval_predicate([p[0].lower(), *p[1:]])) for p in predicates)
    if goal["kind"] == "plugin":
        return bool(import_symbol(goal["callable"])(raw_env, goal))
    raise ValueError(f"Unsupported LIBERO goal: {goal['kind']}")


def robotwin_goal(task, goal, native_success):
    if goal["kind"] == "native":
        return bool(native_success())
    if goal["kind"] == "place_a2b":
        direction = goal.get("direction")
        if direction not in ("left", "right"):
            raise ValueError("place_a2b direction must be left or right")
        a = np.asarray(task.object.get_pose().p)
        b = np.asarray(task.target_object.get_pose().p)
        distance = np.linalg.norm(a[:2] - b[:2])
        side = a[0] < b[0] if direction == "left" else a[0] > b[0]
        # Native RoboTwin planar relation and both grippers released; no task reset.
        return bool(
            0.08 < distance < 0.2
            and abs(a[1] - b[1]) < 0.05
            and side
            and task.robot.is_left_gripper_open()
            and task.robot.is_right_gripper_open()
        )
    if goal["kind"] == "plugin":
        return bool(import_symbol(goal["callable"])(task, goal))
    raise ValueError(f"Unsupported RoboTwin goal: {goal['kind']}")
