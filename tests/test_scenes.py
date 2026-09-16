from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from immich_moments.media import probe
from immich_moments.scenes import detect_scenes

from conftest import COLOURS, SEGMENT_SECONDS


def cut(path: Path, **kwargs):
    info = probe(path)
    options = {"threshold": 27.0, "min_seconds": 1.5, "max_seconds": 0.0} | kwargs
    return detect_scenes(path, info, **options)


def test_three_colour_cards_become_three_scenes(colour_video: Path) -> None:
    scenes = cut(colour_video)
    assert len(scenes) == len(COLOURS)

    for index, scene in enumerate(scenes):
        assert scene.idx == index
        assert abs(scene.start_seconds - index * SEGMENT_SECONDS) < 0.5
    assert abs(scenes[-1].end_seconds - SEGMENT_SECONDS * len(COLOURS)) < 0.5


def test_scenes_are_contiguous_and_cover_the_file(colour_video: Path) -> None:
    scenes = cut(colour_video)
    assert scenes[0].start_seconds == 0.0
    for previous, following in pairwise(scenes):
        assert following.start_seconds == previous.end_seconds
    assert scenes[-1].end_seconds == probe(colour_video).duration_seconds


def test_a_clip_shorter_than_one_scene_is_a_single_scene(rotated_video: Path) -> None:
    scenes = cut(rotated_video, min_seconds=5.0)
    assert len(scenes) == 1
    assert scenes[0].start_seconds == 0.0
    assert scenes[0].end_seconds == probe(rotated_video).duration_seconds


def test_long_takes_are_split_into_chunks(colour_video: Path) -> None:
    """Home video is one long take; without this a single frame stands in for minutes."""
    scenes = cut(colour_video, max_seconds=2.0)
    assert len(scenes) == 9
    assert all(scene.end_seconds - scene.start_seconds <= 2.01 for scene in scenes)
    assert [scene.idx for scene in scenes] == list(range(9))
    assert abs(scenes[-1].end_seconds - probe(colour_video).duration_seconds) < 0.01


def test_mid_seconds_is_the_centre_of_the_scene(colour_video: Path) -> None:
    scene = cut(colour_video)[1]
    assert scene.mid_seconds == (scene.start_seconds + scene.end_seconds) / 2
