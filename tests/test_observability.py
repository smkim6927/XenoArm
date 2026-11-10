import json
from pathlib import Path
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from xenoarm.config import load_config, validate
from xenoarm.envs.libero import LiberoEnv
from xenoarm.envs.robotwin import RoboTwinEnv
from xenoarm.envs.smoke import SmokeEnv
from xenoarm.runner import run

ROOT = Path(__file__).parents[1]


def visual_config():
    config = load_config(ROOT / "configs/smoke.json")
    config["seeds"] = [0]
    config["progress"] = False
    config["visualization"] = {
        "enabled": True,
        "width": 640,
        "height": 360,
        "frame_stride": 7,
        "max_episodes": 1,
    }
    return config


def test_real_offline_wandb_video_and_portable_report(tmp_path, monkeypatch):
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    config = visual_config()
    config["tracking"] = {"mode": "offline", "project": "xenoarm-tests"}
    config["name"] = "<script>experiment</script>"
    output = tmp_path / "recorded"
    run(config, output)
    row = json.loads((output / "episodes.jsonl").read_text())
    assert row["video"]["frames"] == 8  # Initial + stride + correction + last step.
    assert row["instrumentation_seconds"] > 0
    assert row["success"] and row["stale_action_rate"] == 0
    video_path = output / row["video"]["path"]
    with imageio.get_reader(video_path) as decoder:
        assert decoder.count_frames() == row["video"]["frames"]
        first, last = decoder.get_data(0), decoder.get_data(7)
    assert first.shape == (360, 640, 3)
    # Confirm simulator pixels changed, not just the text overlay.
    assert np.abs(first[140:].astype(float) - last[140:]).mean() > 0.1
    document = (output / "report.html").read_text()
    assert "<video controls" in document and "videos/seed-0.mp4" in document
    assert "<script>experiment</script>" not in document
    assert "&lt;script&gt;experiment&lt;/script&gt;" in document
    assert (output / "plots/seed-0.png").stat().st_size > 1000
    summary = json.loads((output / "summary.json").read_text())
    assert summary["success_rate"] == 1 and summary["status"] == "completed"
    wandb_runs = list((output / "wandb").glob("offline-run-*"))
    assert len(wandb_runs) == 1
    assert any(p.stat().st_size > 0 for p in wandb_runs[0].glob("*.wandb"))
    assert list(wandb_runs[0].glob("files/media/videos/**/*.mp4"))


def test_render_failure_is_not_marked_completed(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr(SmokeEnv, "render", fail)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="camera unavailable"):
        run(visual_config(), output)
    assert json.loads((output / "metadata.json").read_text())["status"] == "failed"


def test_progress_bar_and_no_video_path(tmp_path, capsys):
    config = visual_config()
    config["visualization"]["enabled"] = False
    config["progress"] = True
    output = tmp_path / "progress"
    run(config, output)
    assert "Episodes" in capsys.readouterr().err
    assert "<video controls" not in (output / "report.html").read_text()
    assert not (output / "videos").exists()


def test_libero_and_robotwin_rgb_contracts():
    image = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    libero = LiberoEnv.__new__(LiberoEnv)
    libero.observe = lambda: {"agentview_image": image, "robot0_eye_in_hand_image": image}
    assert np.array_equal(libero.render()["LIBERO agentview"], image[::-1, ::-1])
    robotwin = RoboTwinEnv.__new__(RoboTwinEnv)
    robotwin.task = SimpleNamespace(
        get_obs=lambda: {
            "observation": {
                "head_camera": {"rgb": image},
                "left_camera": {"rgb": image},
                "right_camera": {"rgb": image},
            }
        }
    )
    assert list(robotwin.render()) == ["head_camera", "left_camera", "right_camera"]


@pytest.mark.parametrize(
    "options",
    [{"fps": 0}, {"frame_stride": 0}, {"max_episodes": 0}, {"width": 321}, {"height": 200}],
)
def test_bad_video_settings_are_rejected_before_running(options):
    config = visual_config()
    config["visualization"].update(options)
    with pytest.raises(ValueError):
        validate(config)


def test_tracking_never_accepts_secret_values_in_config():
    config = visual_config()
    config["tracking"] = {"mode": "online", "api_key": "not-a-real-key"}
    with pytest.raises(ValueError, match="credentials"):
        validate(config)
