"""Scene segmentation with PySceneDetect's ContentDetector."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

from .errors import MediaError
from .media import MediaInfo


@dataclass(slots=True)
class Scene:
    idx: int
    start_seconds: float
    end_seconds: float

    @property
    def mid_seconds(self) -> float:
        return (self.start_seconds + self.end_seconds) / 2.0


def detect_scenes(
    path: Path,
    info: MediaInfo,
    *,
    threshold: float,
    min_seconds: float,
    max_seconds: float = 0.0,
) -> list[Scene]:
    """Cut points for one video.

    A clip too short to contain a cut, or one the detector finds nothing in, comes back as a
    single scene spanning the whole file rather than as an empty list. Home video is full of
    long unbroken takes, so with `max_seconds` any shot longer than that is split into equal
    chunks; one frame cannot stand in for five minutes of footage.
    """
    duration = info.duration_seconds
    if duration <= min_seconds * 2:
        return [Scene(0, 0.0, duration)]

    boundaries = _content_cuts(path, threshold=threshold, min_seconds=min_seconds)
    spans = [(start, min(end, duration)) for start, end in boundaries if start < duration]
    if not spans:
        spans = [(0.0, duration)]
    spans[-1] = (spans[-1][0], duration)
    return _numbered(_subdivide(spans, max_seconds))


def _subdivide(spans: list[tuple[float, float]], max_seconds: float) -> list[tuple[float, float]]:
    if max_seconds <= 0:
        return spans
    out: list[tuple[float, float]] = []
    for start, end in spans:
        length = end - start
        parts = max(math.ceil(length / max_seconds), 1)
        step = length / parts
        out.extend((start + i * step, start + (i + 1) * step) for i in range(parts - 1))
        out.append((start + (parts - 1) * step, end))
    return out


def _numbered(spans: list[tuple[float, float]]) -> list[Scene]:
    return [Scene(idx, start, end) for idx, (start, end) in enumerate(spans)]


def _content_cuts(path: Path, *, threshold: float, min_seconds: float) -> list[tuple[float, float]]:
    from scenedetect import ContentDetector, SceneManager, open_video
    from scenedetect.video_stream import VideoOpenFailure

    # Importing scenedetect pins its own logger to INFO, so without this every cut it finds
    # is printed through our root handler whether or not the user asked for logging.
    logging.getLogger("pyscenedetect").setLevel(logging.getLogger().getEffectiveLevel())

    try:
        video = open_video(str(path))
    except VideoOpenFailure as exc:
        raise MediaError(f"PySceneDetect could not open {path.name}: {exc}") from exc

    fps = video.frame_rate or 30.0
    manager = SceneManager()
    manager.add_detector(ContentDetector(threshold=threshold, min_scene_len=max(int(min_seconds * fps), 1)))
    manager.detect_scenes(video, show_progress=False)
    return [(start.get_seconds(), end.get_seconds()) for start, end in manager.get_scene_list()]
