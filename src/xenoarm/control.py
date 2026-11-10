"""Synchronous chunk execution with explicit command provenance.

The controller never executes an old chunk after a correction in cancel mode.
Drain mode intentionally preserves old actions as an experimental baseline.
No claim about physical stopping time is made by this software boundary.
"""

from collections import deque
from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass
class Dispatch:
    action: np.ndarray
    command_version: int
    stale: bool
    inference_ms: float | None


class ChunkController:
    def __init__(self, policy, action_dim, *, replan_steps=8, mode="cancel"):
        if mode not in ("cancel", "drain") or replan_steps < 1 or action_dim < 1:
            raise ValueError("Invalid controller settings")
        self.policy = policy
        self.action_dim = action_dim
        self.replan_steps = replan_steps
        self.mode = mode
        self.pending = deque()
        self.version = 0
        self.instruction = None

    def command(self, instruction):
        self.version += 1
        self.instruction = instruction
        discarded = len(self.pending) if self.mode == "cancel" else 0
        if self.mode == "cancel":
            self.pending.clear()
        # A model adapter with its own temporal/language cache must invalidate it here.
        invalidate = getattr(self.policy, "on_instruction", None)
        if invalidate is not None:
            invalidate(instruction)
        return discarded

    def next(self, observation):
        if self.instruction is None:
            raise RuntimeError("Set an instruction before requesting an action")
        inference_ms = None
        if not self.pending:
            sync = getattr(self.policy, "synchronize", lambda: None)
            sync()
            start = perf_counter()
            chunk = np.asarray(self.policy.predict(observation, self.instruction), dtype=np.float32)
            sync()
            inference_ms = (perf_counter() - start) * 1000
            if chunk.ndim != 2 or chunk.shape[0] < 1 or chunk.shape[1] != self.action_dim:
                raise ValueError(
                    f"Expected nonempty [T, {self.action_dim}] action chunk, got {chunk.shape}"
                )
            if not np.isfinite(chunk).all():
                raise ValueError("Policy returned non-finite actions")
            self.pending.extend((a.copy(), self.version) for a in chunk[: self.replan_steps])
        action, version = self.pending.popleft()
        return Dispatch(action, version, version != self.version, inference_ms)
