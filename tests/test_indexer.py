"""The visual pass end to end over a real generated video.

ffmpeg, PySceneDetect, the frame grabs, the thumbnails and the SQLite writes are all the
shipped code running for real. Only the two HTTP endpoints are replayed, because a test suite
cannot assume an Immich server and a GPU.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs

import httpx
import numpy as np
import pytest

from immich_moments.config import Config
from immich_moments.immich import ImmichClient
from immich_moments.indexer import Indexer, IndexReport, _duration, run_index
from immich_moments.ml import MLClient
from immich_moments.store import Store, TranscriptRecord

from conftest import COLOURS, SEGMENT_SECONDS

DIM = 32
ASSET_ID = "3f7b0b1a-0000-4000-8000-000000000001"
FRAME_LABEL = "a colour card"


def stable_vector(text: str) -> np.ndarray:
    """A deterministic unit vector per string, so the same text always embeds the same way."""
    seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
    vector = np.random.default_rng(seed).standard_normal(DIM).astype(np.float32)
    return vector / np.linalg.norm(vector)


def ml_transport() -> httpx.MockTransport:
    """Every frame embeds as the label vector, so the label path is exercised for real."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = request.content.decode("utf-8", "replace")
        if request.headers.get("content-type", "").startswith("multipart/"):
            return httpx.Response(200, json={"clip": json.dumps(stable_vector(FRAME_LABEL).tolist())})
        text = parse_qs(body)["text"][0]
        return httpx.Response(200, json={"clip": json.dumps(stable_vector(text).tolist())})

    return httpx.MockTransport(handler)


def immich_transport(video: Path, *, missing: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path.endswith("/search/metadata"):
            page = int(json.loads(request.content).get("page", 1))
            if page > 1:
                return httpx.Response(200, json={"assets": {"items": [], "nextPage": None}})
            return httpx.Response(
                200,
                json={
                    "assets": {
                        "items": [
                            {
                                "id": ASSET_ID,
                                "originalFileName": "colour-cards.mp4",
                                "type": "VIDEO",
                                "duration": 18000,
                                "fileCreatedAt": "2026-06-01T12:00:00.000Z",
                                "updatedAt": "2026-06-01T12:00:00.000Z",
                            }
                        ],
                        "nextPage": 2,
                    }
                },
            )
        if path.endswith("/people"):
            return httpx.Response(200, json={"people": [], "hasNextPage": False})
        if path.endswith("/original"):
            if missing:
                return httpx.Response(404, json={"message": "Not found"})
            return httpx.Response(200, content=video.read_bytes())
        raise AssertionError(f"unexpected {request.method} {path}")

    return httpx.MockTransport(handler)


@pytest.fixture
def labels_file(tmp_path: Path) -> Path:
    path = tmp_path / "labels.txt"
    path.write_text(f"# a comment\n{FRAME_LABEL}\na birthday cake\n\na dog\n", encoding="utf-8")
    return path


def index(config: Config, store: Store, video: Path, labels_file: Path, *, missing: bool = False):
    store.check_model("test-clip", DIM, reindex=False)
    with (
        ImmichClient(config, transport=immich_transport(video, missing=missing)) as immich,
        MLClient(config, "test-clip", "test-faces", transport=ml_transport()) as ml,
    ):
        return run_index(
            config,
            store,
            immich,
            ml,
            since=None,
            limit=None,
            phases=("visual",),
            labels_path=labels_file,
            reindex=False,
        )


def test_the_visual_pass_indexes_a_real_video(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    report = index(config, store, colour_video, labels_file)

    assert report.discovered == 1
    assert report.visual_indexed == 1
    assert report.unavailable == []
    assert report.failed == []
    assert report.scenes == len(COLOURS)

    scenes = store.scenes_for(ASSET_ID)
    assert [scene["idx"] for scene in scenes] == [0, 1, 2]
    assert scenes[1]["start_seconds"] == pytest.approx(SEGMENT_SECONDS, abs=0.5)
    assert all(scene["label"] == FRAME_LABEL for scene in scenes)
    assert all(scene["vector_row"] is not None for scene in scenes)

    thumbs = sorted(path.name for path in config.thumbs_dir.glob("*.jpg"))
    assert thumbs == [f"{ASSET_ID}-000{i}.jpg" for i in range(3)]
    assert all((config.thumbs_dir / name).stat().st_size > 0 for name in thumbs)

    assert store.vectors().read_all().shape == (3, DIM)
    assert store.asset(ASSET_ID)["duration_seconds"] == pytest.approx(18.0, abs=0.3)
    assert store.asset(ASSET_ID)["has_audio"] == 1


def test_the_audio_sidecar_is_written_while_the_video_is_already_open(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    index(config, store, colour_video, labels_file)
    sidecar = config.audio_dir / f"{ASSET_ID}.flac"
    assert sidecar.exists()
    assert sidecar.stat().st_size > 0


def test_a_video_without_audio_leaves_no_sidecar(
    config: Config, store: Store, silent_video: Path, labels_file: Path
) -> None:
    report = index(config, store, silent_video, labels_file)
    assert report.visual_indexed == 1
    assert not list(config.audio_dir.glob("*.flac"))
    assert store.asset(ASSET_ID)["has_audio"] == 0


def test_a_second_run_skips_what_is_already_indexed(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    index(config, store, colour_video, labels_file)
    second = index(config, store, colour_video, labels_file)

    assert second.discovered == 1
    assert second.visual_indexed == 0
    assert store.vectors().read_all().shape == (3, DIM)


def test_a_404_original_is_recorded_and_the_run_carries_on(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    report = index(config, store, colour_video, labels_file, missing=True)

    assert report.unavailable == ["colour-cards.mp4"]
    assert report.failed == []
    assert report.visual_indexed == 0
    assert store.asset(ASSET_ID)["status"] == "unavailable"
    assert store.assets_needing("visual") == []


def test_the_label_cache_is_reused_between_runs(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    index(config, store, colour_video, labels_file)
    caches = list(config.data_dir.glob("labels-*.npz"))
    assert len(caches) == 1
    stamp = caches[0].stat().st_mtime_ns

    index(config, store, colour_video, labels_file)
    assert caches[0].stat().st_mtime_ns == stamp


def test_phase_timings_are_reported(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    report = index(config, store, colour_video, labels_file)
    named = {timing.name: timing for timing in report.timings}
    assert set(named) >= {"discover", "people", "visual"}
    assert named["visual"].assets == 1
    assert named["visual"].seconds > 0


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (44100, 44.1),
        (15626, 15.626),
        (0, None),
        (None, None),
        ("", None),
        ("00:00:44.100000", 44.1),
        ("01:02:03.000000", 3723.0),
        ("not a duration", None),
    ],
)
def test_duration_parsing_covers_both_server_shapes(raw: object, expected: float | None) -> None:
    """Immich 3.2 sends integer milliseconds; older releases sent HH:MM:SS.ssssss."""
    result = _duration(raw)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


# ---- the speech pass -----------------------------------------------------


class SilentTranscriber:
    """Stands in for Whisper: the audio pass is being tested, not the model."""

    def __init__(self) -> None:
        self.seen: list[Path] = []

    def transcribe(self, audio: Path):
        self.seen.append(audio)
        return [TranscriptRecord(1.0, 2.0, "happy birthday to you")]

    def unload(self) -> None:
        pass


def audio_pass(config: Config, store: Store, video: Path, transcriber, *, missing: bool = False):
    report = IndexReport()
    with (
        ImmichClient(config, transport=immich_transport(video, missing=missing)) as immich,
        MLClient(config, "test-clip", "test-faces", transport=ml_transport()) as ml,
    ):
        indexer = Indexer(config, store, immich, ml)
        with patch("immich_moments.transcribe.Transcriber", lambda _config: transcriber):
            indexer.audio_pass(report)
    return report


def test_a_missing_sidecar_is_taken_from_the_original_again(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    """An interrupted run deletes the sidecar; believing it would file a talkative video as silent."""
    index(config, store, colour_video, labels_file)
    sidecar = config.audio_dir / f"{ASSET_ID}.flac"
    sidecar.unlink()

    transcriber = SilentTranscriber()
    report = audio_pass(config, store, colour_video, transcriber)

    assert report.segments == 1
    assert report.no_audio_track == 0
    assert transcriber.seen == [sidecar]
    assert store.asset(ASSET_ID)["has_audio"] == 1
    assert not sidecar.exists()  # tidied up once it has been read


def test_a_video_with_no_audio_track_is_recorded_as_such(
    config: Config, store: Store, silent_video: Path, labels_file: Path
) -> None:
    index(config, store, silent_video, labels_file)

    transcriber = SilentTranscriber()
    report = audio_pass(config, store, silent_video, transcriber)

    assert report.no_audio_track == 1
    assert report.no_speech == 0
    assert transcriber.seen == []
    assert store.asset(ASSET_ID)["audio_indexed_at"] is not None


def test_a_video_with_sound_but_no_speech_is_not_called_trackless(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    class Speechless(SilentTranscriber):
        def transcribe(self, audio: Path):
            self.seen.append(audio)
            return []

    index(config, store, colour_video, labels_file)
    report = audio_pass(config, store, colour_video, Speechless())

    assert (report.no_speech, report.no_audio_track, report.segments) == (1, 0, 0)
    assert store.asset(ASSET_ID)["has_audio"] == 1


def test_a_404_during_re_extraction_is_recorded_not_raised(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    index(config, store, colour_video, labels_file)
    (config.audio_dir / f"{ASSET_ID}.flac").unlink()

    report = audio_pass(config, store, colour_video, SilentTranscriber(), missing=True)

    assert report.unavailable == ["colour-cards.mp4"]
    assert report.failed == []


def test_transcript_segments_land_in_the_scene_that_was_playing(
    config: Config, store: Store, colour_video: Path, labels_file: Path
) -> None:
    class AtSeven(SilentTranscriber):
        def transcribe(self, audio: Path):
            self.seen.append(audio)
            return [TranscriptRecord(SEGMENT_SECONDS + 0.3, SEGMENT_SECONDS + 2.0, "blow out the candles")]

    index(config, store, colour_video, labels_file)
    audio_pass(config, store, colour_video, AtSeven())

    segment = store.transcript_for(ASSET_ID)[0]
    scenes = {scene["id"]: scene["idx"] for scene in store.scenes_for(ASSET_ID)}
    assert scenes[segment["scene_id"]] == 1
