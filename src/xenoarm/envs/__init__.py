"""Simulator dependencies are loaded only for the selected backend."""


def make_env(config):
    backend = config["backend"]
    if backend == "mujoco_smoke":
        from .smoke import SmokeEnv

        return SmokeEnv(config)
    if backend == "libero":
        from .libero import LiberoEnv

        return LiberoEnv(config)
    if backend == "robotwin":
        from .robotwin import RoboTwinEnv

        return RoboTwinEnv(config)
    raise ValueError(f"Unknown environment backend: {backend}")
