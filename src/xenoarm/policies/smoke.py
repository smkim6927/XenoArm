"""Deliberately simple text rule used to validate orchestration, not a WAM."""

import re

import numpy as np


class ScriptedPolicy:
    def __init__(self, config=None):
        self.horizon = (config or {}).get("horizon", 24)

    def reset(self, seed):
        pass

    def predict(self, observation, instruction):
        directions = re.findall(r"\b(left|right)\b", instruction.lower())
        if not directions:
            raise ValueError("Smoke policy requires left or right in the instruction")
        # The last direction handles 'No, reach right instead' in the smoke scenario.
        target = [-0.4 if directions[-1] == "left" else 0.4, 0.0]
        return np.tile(target, (self.horizon, 1))
