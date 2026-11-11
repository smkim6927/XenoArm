"""Stream actual simulator RGB frames to an annotated, browser-playable MP4."""

import os
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def configure_headless():
    """Choose EGL before importing MuJoCo when no display is attached."""
    if not os.environ.get("DISPLAY"):
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")


def configure_writable_caches(output):
    cache = Path(output).resolve() / ".cache"
    cache.mkdir(exist_ok=True)
    for name, folder in (
        ("XDG_CACHE_HOME", cache),
        ("WANDB_CACHE_DIR", cache / "wandb"),
        ("WANDB_DATA_DIR", cache / "wandb-data"),
        ("MESA_SHADER_CACHE_DIR", cache / "mesa"),
        ("MPLCONFIGDIR", cache / "matplotlib"),
    ):
        os.environ.setdefault(name, str(folder))


@lru_cache(maxsize=8)
def _font(size):
    for candidate in (
        os.environ.get("XENOARM_FONT"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "DejaVuSans.ttf",
    ):
        if candidate:
            try:
                return ImageFont.truetype(candidate, size)
            except OSError:
                pass
    return ImageFont.load_default(size=size)


def compose_frame(cameras, row, instruction, *, width=960, height=540):
    if not cameras:
        raise ValueError("The simulator returned no RGB cameras")
    canvas = Image.new("RGB", (width, height), (15, 23, 42))
    draw = ImageDraw.Draw(canvas)
    font = _font(16)
    color = (251, 146, 60) if row["stale"] else (74, 222, 128)
    state = "STALE ACTION" if row["stale"] else "CURRENT COMMAND"
    draw.text(
        (16, 10),
        f"Step {row['step']}  |  command v{row['command_version']}  |  action v{row['action_version']}  |  {state}",
        font=font,
        fill=color,
    )
    for idx, line in enumerate(textwrap.wrap(instruction, max(16, width // 10))[:3]):
        draw.text((16, 36 + idx * 20), line, font=font, fill="white")
    top = 108
    cell_width = width // len(cameras)
    for index, (name, frame) in enumerate(cameras.items()):
        frame = np.asarray(frame)
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"Camera {name} must return uint8 RGB [H, W, 3], got {frame.shape}/{frame.dtype}"
            )
        image = Image.fromarray(frame)
        image.thumbnail((cell_width - 12, height - top - 32), Image.Resampling.LANCZOS)
        x = index * cell_width + (cell_width - image.width) // 2
        y = top + 24 + (height - top - 32 - image.height) // 2
        canvas.paste(image, (x, y))
        draw.text((index * cell_width + 12, top), name, font=font, fill=(148, 163, 184))
    return np.asarray(canvas)


class EpisodeVideo:
    def __init__(self, output, seed, options):
        import imageio.v2 as imageio

        self.output = Path(output)
        directory = self.output / "videos"
        directory.mkdir(exist_ok=True)
        self.path = directory / f"seed-{seed}.mp4"
        self.poster = directory / f"seed-{seed}.png"
        self.width, self.height = options.get("width", 960), options.get("height", 540)
        self.fps = options.get("fps", 20)
        self.writer = imageio.get_writer(
            str(self.path),
            format="FFMPEG",
            fps=self.fps,
            codec="libx264",
            quality=7,
            macro_block_size=1,
            ffmpeg_log_level="error",
        )
        self.frames = 0
        self.last_frame = None
        self.last_step = None

    def capture(self, env, row, instruction):
        frame = compose_frame(
            env.render(width=self.width, height=self.height),
            row,
            instruction,
            width=self.width,
            height=self.height,
        )
        self.writer.append_data(frame)
        self.frames += 1
        self.last_frame = frame
        self.last_step = row["step"]

    def close(self):
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            if self.last_frame is not None:
                Image.fromarray(self.last_frame).save(self.poster)
        if not self.frames:
            return None
        return {
            "path": str(self.path.relative_to(self.output)),
            "poster": str(self.poster.relative_to(self.output)),
            "frames": self.frames,
            "fps": self.fps,
            "playback_seconds": self.frames / self.fps,
        }
