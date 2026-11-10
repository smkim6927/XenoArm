"""Adapters to the pinned official Light-WAM inference and normalization code.

Only the external controller owns the action queue. Every prediction receives
the current text; the LIBERO fixed-prompt cache is never used here.
"""

import importlib
import sys
from pathlib import Path

from xenoarm.goals import import_symbol


class BaseLightWAM:
    def __init__(self, config):
        self.config = config
        self.root = Path(config["lightwam_root"]).resolve()
        for name in ("checkpoint", "training_config", "dataset_stats"):
            if not Path(config[name]).is_file():
                raise FileNotFoundError(f"Missing {name}: {config[name]}")
        for path in (self.root, self.root / "src"):
            sys.path.insert(0, str(path))
        import torch

        self.torch = torch
        self.device = config.get("device", "cuda")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "Light-WAM GPU evaluation requires CUDA; the CPU smoke policy is separate"
            )
        self.horizon = config.get("action_horizon", 24)

    def transform_model(self, model):
        # Quantization/token pruning needs a real, model-specific implementation.
        # Merely labelling a run 'INT4' or slicing action outputs is not compression.
        hook = self.config.get("model_transform")
        if hook:
            model = import_symbol(hook)(model, self.config.get("transform_options", {}))
        return model

    def reset(self, seed):
        self.torch.manual_seed(seed)
        if self.device.startswith("cuda"):
            self.torch.cuda.manual_seed_all(seed)
            self.torch.cuda.reset_peak_memory_stats(self.device)

    def synchronize(self):
        if self.device.startswith("cuda"):
            self.torch.cuda.synchronize(self.device)

    def on_instruction(self, instruction):
        # No instruction-agnostic context is retained by these adapters. A custom
        # compression transform may register a cache invalidator on the model.
        invalidate = getattr(self.model, "invalidate_language_cache", None)
        if invalidate:
            invalidate()

    def resources(self):
        return {
            "gpu_peak_allocated_bytes": (
                self.torch.cuda.max_memory_allocated(self.device)
                if self.device.startswith("cuda")
                else None
            ),
            "gpu_peak_reserved_bytes": (
                self.torch.cuda.max_memory_reserved(self.device)
                if self.device.startswith("cuda")
                else None
            ),
        }


class LiberoLightWAM(BaseLightWAM):
    def __init__(self, config):
        super().__init__(config)
        libero_root = Path(config["libero_root"]).resolve()
        sys.path.insert(0, str(libero_root))
        # Set LIBERO_CONFIG_PATH before importing upstream evaluation utilities.
        import os

        os.environ["LIBERO_CONFIG_PATH"] = str(libero_root / ".xenoarm-config")
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        upstream = importlib.import_module("experiments.libero.eval_libero_single")
        self.upstream = upstream
        with initialize_config_dir(version_base="1.3", config_dir=str(self.root / "configs")):
            cfg = compose(config_name="sim_libero")
        cfg.ckpt = str(Path(config["checkpoint"]).resolve())
        cfg.EVALUATION.training_config_path = str(Path(config["training_config"]).resolve())
        cfg.EVALUATION.dataset_stats_path = str(Path(config["dataset_stats"]).resolve())
        cfg = upstream._maybe_apply_training_run_config(cfg)
        # Set overrides after loading the training configuration.
        cfg.EVALUATION.preencode_prompt_context = False
        cfg.EVALUATION.visualize_future_video = False
        cfg.EVALUATION.num_inference_steps = config.get("num_inference_steps", 10)
        cfg.mixed_precision = config.get("precision", "bf16")
        dtype = upstream._mixed_precision_to_model_dtype(cfg.mixed_precision)
        self.model = instantiate(cfg.model, model_dtype=dtype, device=self.device)
        upstream._load_model_checkpoint(self.model, cfg.ckpt)
        self.model = self.transform_model(self.model.to(self.device).eval())
        stats = upstream.load_dataset_stats_from_json(cfg.EVALUATION.dataset_stats_path)
        self.processor = instantiate(cfg.data.train.processor).eval()
        self.processor.set_normalizer_from_stats(stats)
        self.cfg = cfg
        self.height, self.width = cfg.data.train.video_size

    def reset(self, seed):
        super().reset(seed)
        self.cfg.seed = seed

    def predict(self, observation, instruction):
        action, _, _ = self.upstream._predict_action_chunk(
            observation,
            instruction,
            self.model,
            self.processor,
            self.cfg,
            action_horizon=self.horizon,
            input_w=self.width,
            input_h=self.height,
            model_device=self.device,
            cached_prompt_context=None,
            cached_prompt_context_mask=None,
        )
        return action


class RoboTwinLightWAM(BaseLightWAM):
    def __init__(self, config):
        super().__init__(config)
        upstream = importlib.import_module("experiments.robotwin.lightwam_policy.deploy_policy")
        self.policy = upstream.get_model(
            {
                "ckpt_setting": str(Path(config["checkpoint"]).resolve()),
                "dataset_stats_path": str(Path(config["dataset_stats"]).resolve()),
                "training_config_path": str(Path(config["training_config"]).resolve()),
                "use_training_run_config": True,
                "sim_cfg_name": "sim_robotwin.yaml",
                "device": self.device,
                "action_horizon": self.horizon,
                "num_inference_steps": config.get("num_inference_steps", 10),
                "mixed_precision": config.get("precision", "bf16"),
            }
        )
        self.policy.model = self.transform_model(self.policy.model)
        self.model = self.policy.model

    def reset(self, seed):
        super().reset(seed)
        self.policy.reset()
        self.policy.seed = seed

    def predict(self, observation, instruction):
        # Never call upstream step(): it has an independent pending-action queue.
        return self.policy._infer_action_chunk(observation, instruction)
