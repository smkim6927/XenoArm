from types import SimpleNamespace

import numpy as np
import pytest

from xenoarm.goals import libero_goal, robotwin_goal


def test_libero_corrected_goal_uses_predicates_not_old_success():
    checked = []
    raw = SimpleNamespace(
        check_success=lambda: True,
        env=SimpleNamespace(
            object_states_dict={"bowl": object(), "tray": object()},
            _eval_predicate=lambda p: checked.append(p) or False,
        ),
    )
    assert not libero_goal(
        raw, {"kind": "libero_predicates", "predicates": [["On", "bowl", "tray"]]}
    )
    assert checked == [["on", "bowl", "tray"]]
    with pytest.raises(ValueError, match="absent"):
        libero_goal(raw, {"kind": "libero_predicates", "predicates": [["On", "bowl", "missing"]]})


def test_robotwin_spatial_correction_requires_release_of_both_grippers():
    class Actor:
        def __init__(self, x):
            self.x = x

        def get_pose(self):
            return SimpleNamespace(p=np.array([self.x, 0, 0.8]))

    opened = [True, True]
    task = SimpleNamespace(
        object=Actor(0.13),
        target_object=Actor(0),
        robot=SimpleNamespace(
            is_left_gripper_open=lambda: opened[0], is_right_gripper_open=lambda: opened[1]
        ),
    )

    def native():
        return True

    assert robotwin_goal(task, {"kind": "place_a2b", "direction": "right"}, native)
    assert not robotwin_goal(task, {"kind": "place_a2b", "direction": "left"}, native)
    opened[1] = False
    assert not robotwin_goal(task, {"kind": "place_a2b", "direction": "right"}, native)
