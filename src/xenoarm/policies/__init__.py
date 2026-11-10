"""Policy factories; importing XenoArm never imports CUDA or model weights."""

from xenoarm.goals import import_symbol


def make_policy(config):
    backend = config["backend"]
    if backend == "scripted_smoke":
        from .smoke import ScriptedPolicy

        return ScriptedPolicy(config)
    if backend == "lightwam_libero":
        from .lightwam import LiberoLightWAM

        return LiberoLightWAM(config)
    if backend == "lightwam_robotwin":
        from .lightwam import RoboTwinLightWAM

        return RoboTwinLightWAM(config)
    if backend == "plugin":
        return import_symbol(config["factory"])(config)
    raise ValueError(f"Unknown policy backend: {backend}")
