"""Contract tests isolate simulator preprocessing from unavailable model weights."""

from types import SimpleNamespace

import numpy as np

from xenoarm.control import ChunkController
from xenoarm.policies.lightwam import LiberoLightWAM, RoboTwinLightWAM


def test_libero_adapter_forwards_latest_language_without_original_prompt_context():
    calls = []

    def infer(obs, instruction, model, processor, cfg, **kwargs):
        calls.append((obs, instruction, kwargs))
        return np.ones((4, 7)), {}, None

    adapter = LiberoLightWAM.__new__(LiberoLightWAM)
    adapter.upstream = SimpleNamespace(_predict_action_chunk=infer)
    adapter.model = object()
    adapter.processor = object()
    adapter.cfg = object()
    adapter.horizon, adapter.height, adapter.width = 4, 224, 224
    adapter.device = "cpu"
    # No device work is needed for the contract stub.
    adapter.synchronize = lambda: None
    control = ChunkController(adapter, 7, replan_steps=4)
    control.command("put bowl on plate")
    control.next("initial_observation")
    control.command("put bowl on ramekin")
    control.next("corrected_observation")
    assert [c[:2] for c in calls] == [
        ("initial_observation", "put bowl on plate"),
        ("corrected_observation", "put bowl on ramekin"),
    ]
    assert all(c[2]["cached_prompt_context"] is None for c in calls)
    assert all(c[2]["cached_prompt_context_mask"] is None for c in calls)


def test_robotwin_adapter_does_not_use_upstream_execution_queue():
    calls = []

    class NativePolicy:
        def _infer_action_chunk(self, observation, instruction):
            calls.append((observation, instruction))
            return np.zeros((8, 14))

        def step(self, *args):
            raise AssertionError("Upstream queue must not execute actions")

    adapter = RoboTwinLightWAM.__new__(RoboTwinLightWAM)
    adapter.policy = NativePolicy()
    assert adapter.predict("fresh image", "place right").shape == (8, 14)
    assert calls == [("fresh image", "place right")]
