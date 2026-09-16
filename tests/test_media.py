from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from immich_moments.errors import MediaError
from immich_moments.media import extract_audio, extract_frame, ffprobe_path, probe


def test_probe_reads_duration_size_and_audio(colour_video: Path) -> None:
    info = probe(colour_video)
    assert 17.0 < info.duration_seconds < 19.0
    assert (info.width, info.height) == (320, 240)
    assert info.has_audio is True
    assert info.rotation == 0


def test_probe_reports_a_missing_audio_track(silent_video: Path) -> None:
    assert probe(silent_video).has_audio is False


def test_rotation_is_read_from_the_display_matrix(rotated_video: Path) -> None:
    info = probe(rotated_video)
    assert info.rotation in (90, 270)
    assert (info.width, info.height) == (240, 320), "portrait after honouring the display matrix"


def test_extract_frame_honours_rotation(rotated_video: Path) -> None:
    import io

    from PIL import Image

    info = probe(rotated_video)
    jpeg = extract_frame(rotated_video, 1.0, info)
    assert jpeg.startswith(b"\xff\xd8")
    assert Image.open(io.BytesIO(jpeg)).size == (240, 320)


def test_extract_audio_writes_16k_mono(colour_video: Path, tmp_path: Path) -> None:
    out = extract_audio(colour_video, tmp_path / "audio.flac")
    assert out.exists() and out.stat().st_size > 0
    stream = json.loads(
        subprocess.run(
            [
                ffprobe_path(),
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_streams",
                "-print_format",
                "json",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    )["streams"][0]
    assert stream["codec_name"] == "flac"
    assert stream["sample_rate"] == "16000"
    assert stream["channels"] == 1


def test_probe_rejects_a_file_that_is_not_media(tmp_path: Path) -> None:
    broken = tmp_path / "not-a-video.mp4"
    broken.write_bytes(b"this is not a video")
    with pytest.raises(MediaError) as caught:
        probe(broken)
    assert "not-a-video.mp4" in str(caught.value)
