"""Optional W&B logging and tqdm progress, with no implicit online uploads."""

import sys
from time import perf_counter

from tqdm.auto import tqdm

from .visualization import EpisodeVideo


class RunObserver:
    def __init__(self, config, output):
        self.config, self.output = config, output
        self.progress = config.get("progress", sys.stderr.isatty())
        self.episode_bar = None
        self.step_bar = None
        self.video = None
        self.tracker = None
        self.wandb = None
        self.global_step = 0
        self.results = []
        self.instrumentation_seconds = 0.0
        tracking = config.get("tracking", {})
        mode = tracking.get("mode", "disabled")
        if mode != "disabled":
            import wandb

            self.wandb = wandb
            self.tracker = wandb.init(
                project=tracking.get("project", "xenoarm"),
                entity=tracking.get("entity"),
                name=tracking.get("name", config["name"]),
                group=tracking.get("group"),
                config=config,
                dir=str(output),
                mode=mode,
                save_code=False,
                settings=wandb.Settings(disable_git=True, init_timeout=30),
            )
            self.tracker.define_metric("control/*", step_metric="global_control_step")
            self.tracker.define_metric("episode/*", step_metric="episode/index")

    def seeds(self):
        self.episode_bar = tqdm(
            self.config["seeds"], desc="Episodes", unit="episode", disable=not self.progress
        )
        return self.episode_bar

    def begin_episode(self, seed, env, instruction):
        start = perf_counter()
        self.seed = seed
        self.instrumentation_seconds = 0.0
        self.step_bar = tqdm(
            total=self.config["max_steps"],
            desc=f"Seed {seed}",
            unit="step",
            leave=False,
            disable=not self.progress,
        )
        options = self.config.get("visualization", {})
        if options.get("enabled", False) and len(self.results) < options.get("max_episodes", 3):
            self.video = EpisodeVideo(self.output, seed, options)
            self.video.capture(
                env,
                {"step": -1, "command_version": 1, "action_version": 1, "stale": False},
                instruction,
            )
        self.instrumentation_seconds += perf_counter() - start

    def on_step(self, env, row, instruction, *, corrected=False):
        start = perf_counter()
        self.step_bar.update(1)
        if corrected:
            self.step_bar.set_postfix_str(f"correction → v{row['command_version']}")
        stride = self.config.get("visualization", {}).get("frame_stride", 1)
        if self.video is not None and (row["step"] % stride == 0 or corrected):
            self.video.capture(env, row, instruction)
        self.last_env, self.last_row, self.last_instruction = env, row, instruction
        if self.tracker is not None:
            self.global_step += 1
            metrics = {
                "global_control_step": self.global_step,
                "control/seed": self.seed,
                "control/step": row["step"],
                "control/command_version": row["command_version"],
                "control/action_version": row["action_version"],
                "control/stale_action": int(row["stale"]),
                "control/latency_ms": row["control_ms"],
                "control/goal_reached": int(row["goal_reached"]),
            }
            if row["inference_ms"] is not None:
                metrics["control/inference_ms"] = row["inference_ms"]
            self.tracker.log(metrics)
        self.instrumentation_seconds += perf_counter() - start

    def end_episode(self, result):
        if self.video is not None:
            if self.video.last_step != self.last_row["step"]:
                self.video.capture(self.last_env, self.last_row, self.last_instruction)
            result["video"] = self.video.close()
            self.video = None
        self.step_bar.close()
        self.step_bar = None
        # Retain only aggregates; long experiments stream traces to JSONL on disk.
        self.results.append({"success": result["success"]})
        if self.episode_bar is not None:
            self.episode_bar.set_postfix(
                success=f"{sum(r['success'] for r in self.results)}/{len(self.results)}"
            )
        if self.tracker is not None:
            keys = (
                "seed",
                "success",
                "original_task_success",
                "correction_success",
                "steps",
                "stale_action_rate",
                "inference_p95_ms",
                "control_p95_ms",
                "gpu_peak_allocated_bytes",
                "gpu_peak_reserved_bytes",
            )
            metrics = {f"episode/{k}": result[k] for k in keys if result.get(k) is not None}
            metrics["episode/index"] = len(self.results)
            for key in ("correction_to_dispatch_ms", "discarded_actions"):
                values = [e[key] for e in result["corrections"] if e.get(key) is not None]
                if values:
                    metrics[f"episode/{key}"] = sum(values) / len(values)
            if result.get("video"):
                metrics["episode/video"] = self.wandb.Video(
                    str(self.output / result["video"]["path"]), format="mp4"
                )
            self.tracker.log(metrics)

    def close(self, *, status):
        try:
            if self.video is not None:
                self.video.close()
                self.video = None
        finally:
            if self.step_bar is not None:
                self.step_bar.close()
            if self.episode_bar is not None:
                self.episode_bar.close()
            if self.tracker is not None:
                self.tracker.summary["run/status"] = status
                self.tracker.summary["run/completed_episodes"] = len(self.results)
                if self.results:
                    self.tracker.summary["run/success_rate"] = sum(
                        r["success"] for r in self.results
                    ) / len(self.results)
                self.tracker.finish(exit_code=0 if status == "completed" else 1)
