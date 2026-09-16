"""Shared fixtures.

The colour-card video is built with ffmpeg at test time from a committed 5 s recording of the
fixture sentence, so the repository stays small and the video always matches the current
generator. Nothing here is used by the shipped package.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from immich_moments.config import Config
from immich_moments.store import Store

FIXTURES = Path(__file__).parent / "fixtures"
SPEECH_WAV = FIXTURES / "speech.wav"
SENTENCE = "Happy birthday to you. Now blow out the candles."

SEGMENT_SECONDS = 6.0
SPEECH_START = 6.3
COLOURS = ("red", "green", "blue")


def ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        pytest.skip("ffmpeg is not on PATH")
    return path


@pytest.fixture(scope="session")
def colour_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """18 s of red, green and blue cards with the fixture sentence spoken over the green one."""
    out = tmp_path_factory.mktemp("media") / "colour-cards.mp4"
    inputs: list[str] = []
    for colour in COLOURS:
        inputs += ["-f", "lavfi", "-t", str(SEGMENT_SECONDS), "-i", f"color=c={colour}:s=320x240:r=10"]
    total = SEGMENT_SECONDS * len(COLOURS)
    delay = int(SPEECH_START * 1000)
    subprocess.run(
        [
            ffmpeg(),
            "-v",
            "error",
            "-y",
            *inputs,
            "-i",
            str(SPEECH_WAV),
            "-filter_complex",
            f"[0:v][1:v][2:v]concat=n={len(COLOURS)}:v=1:a=0[v];"
            f"[{len(COLOURS)}:a]adelay={delay}|{delay},apad,atrim=0:{total}[a]",
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Two cards and no audio track at all, for the no-audio path."""
    out = tmp_path_factory.mktemp("media") / "silent.mp4"
    subprocess.run(
        [
            ffmpeg(),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-t",
            "5",
            "-i",
            "color=c=orange:s=320x240:r=10",
            "-f",
            "lavfi",
            "-t",
            "5",
            "-i",
            "color=c=navy:s=320x240:r=10",
            "-filter_complex",
            "[0:v][1:v]concat=n=2:v=1:a=0[v]",
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(out),
        ],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture(scope="session")
def rotated_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A portrait clip carrying a 90 degree display matrix, like a phone recording."""
    directory = tmp_path_factory.mktemp("media")
    flat = directory / "flat.mp4"
    out = directory / "rotated.mp4"
    subprocess.run(
        [
            ffmpeg(),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-t",
            "3",
            "-i",
            "testsrc=s=320x240:r=10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            str(flat),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [ffmpeg(), "-v", "error", "-y", "-display_rotation", "90", "-i", str(flat), "-c", "copy", str(out)],
        check=True,
        capture_output=True,
    )
    return out


@pytest.fixture
def config(tmp_path: Path) -> Config:
    config = Config(
        immich_url="http://immich.test",
        immich_api_key="test-key",
        ml_url="http://ml.test",
        data_dir=tmp_path / "data",
        max_retries=2,
    )
    config.ensure_dirs()
    return config


@pytest.fixture
def store(config: Config):
    with Store(config) as opened:
        yield opened


def unit(seed: int, dim: int = 8) -> np.ndarray:
    """A deterministic unit vector, so a test never depends on a real model."""
    rng = np.random.default_rng(seed)
    vector = rng.standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)
