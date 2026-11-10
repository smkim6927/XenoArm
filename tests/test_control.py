import numpy as np
import pytest

from xenoarm.control import ChunkController


class RecordingPolicy:
    def __init__(self):
        self.calls = []

    def predict(self, obs, text):
        self.calls.append((obs, text))
        return np.full((4, 2), 1 if text == "old" else -1)


def test_correction_discards_chunk_and_uses_new_observation_and_text():
    policy = RecordingPolicy()
    control = ChunkController(policy, 2, replan_steps=4)
    control.command("old")
    assert (control.next("before").action == 1).all()
    assert control.command("new") == 3
    dispatch = control.next("after")
    assert (dispatch.action == -1).all() and not dispatch.stale
    assert policy.calls == [("before", "old"), ("after", "new")]


def test_drain_baseline_labels_old_actions_and_then_replans():
    control = ChunkController(RecordingPolicy(), 2, replan_steps=4, mode="drain")
    control.command("old")
    control.next(None)
    assert control.command("new") == 0
    for _ in range(3):
        dispatch = control.next(None)
        assert dispatch.stale and dispatch.command_version == 1
    assert not control.next(None).stale


@pytest.mark.parametrize(
    "chunk", [np.zeros((0, 2)), np.zeros((2, 3)), np.array([[np.nan, 0]]), np.array([[np.inf, 0]])]
)
def test_invalid_policy_chunks_never_reach_simulator(chunk):
    class Bad:
        def predict(self, obs, text):
            return chunk

    control = ChunkController(Bad(), 2)
    control.command("go")
    with pytest.raises(ValueError):
        control.next(None)


def test_repeated_corrections_and_same_text_get_new_versions():
    control = ChunkController(RecordingPolicy(), 2)
    control.command("old")
    control.next(None)
    control.command("new")
    control.command("new")
    assert control.next(None).command_version == 3
