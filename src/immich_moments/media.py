"""ffmpeg/ffprobe wrappers: probing, frame grabs and audio extraction."""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .errors import MediaError

# Transfer functions that need tone mapping before an 8-bit JPEG looks like the video does.
HDR_TRANSFERS = frozenset({"smpte2084", "arib-std-b67"})
TONEMAP = (
    "zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
    "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p"
)
_TIMEOUT = 300


@dataclass(slots=True)
class MediaInfo:
    duration_seconds: float
    width: int
    height: int
    has_audio: bool
    is_hdr: bool
    rotation: int


@functools.cache
def ffmpeg_path() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise MediaError("ffmpeg is not on PATH. Install it (https://ffmpeg.org/download.html) and retry.")
    return path


@functools.cache
def ffprobe_path() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise MediaError("ffprobe is not on PATH. It ships with ffmpeg; install ffmpeg and retry.")
    return path


@functools.cache
def has_zscale() -> bool:
    """zscale needs libzimg, which not every ffmpeg build has. Without it, HDR is not tone mapped."""
    try:
        result = subprocess.run(  # noqa: S603
            [ffmpeg_path(), "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return " zscale " in result.stdout


def _run(args: list[str], *, what: str, capture: bool = False) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(args, capture_output=True, timeout=_TIMEOUT)  # noqa: S603
    except subprocess.TimeoutExpired as exc:
        raise MediaError(f"{what} timed out after {_TIMEOUT}s") from exc
    except OSError as exc:
        raise MediaError(f"{what} could not be started: {exc}") from exc
    if result.returncode != 0:
        tail = (result.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        raise MediaError(f"{what} failed: {tail[-1] if tail else f'exit {result.returncode}'}")
    if capture and not result.stdout:
        raise MediaError(f"{what} produced no output")
    return result


def probe(path: Path) -> MediaInfo:
    if not path.is_file():
        raise MediaError(f"no such file: {path}")
    result = _run(
        [
            ffprobe_path(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        what=f"ffprobe on {path.name}",
        capture=True,
    )
    try:
        data = json.loads(result.stdout)
    except ValueError as exc:
        raise MediaError(f"ffprobe returned unreadable JSON for {path.name}") from exc

    streams = data.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"{path.name} has no video stream")
    audio = any(s.get("codec_type") == "audio" for s in streams)

    duration = _duration(data, video)
    rotation = _rotation(video)
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if rotation in (90, 270):
        width, height = height, width
    return MediaInfo(
        duration_seconds=duration,
        width=width,
        height=height,
        has_audio=audio,
        is_hdr=video.get("color_transfer") in HDR_TRANSFERS,
        rotation=rotation,
    )


def _duration(data: dict, video: dict) -> float:
    for value in (data.get("format", {}).get("duration"), video.get("duration")):
        try:
            seconds = float(value)
        except (TypeError, ValueError):
            continue
        if seconds > 0:
            return seconds
    raise MediaError("could not determine the duration; the file may be truncated")


def _rotation(video: dict) -> int:
    """Rotation in degrees from the display matrix, or the legacy `rotate` tag."""
    for side_data in video.get("side_data_list") or []:
        if "rotation" in side_data:
            return int(-float(side_data["rotation"])) % 360
    tag = (video.get("tags") or {}).get("rotate")
    if tag is not None:
        try:
            return int(float(tag)) % 360
        except ValueError:
            return 0
    return 0


def extract_frame(path: Path, seconds: float, info: MediaInfo, *, max_side: int = 640) -> bytes:
    """One JPEG at `seconds`, upright and tone mapped, with its long side at most `max_side`.

    ffmpeg applies the display matrix itself unless told not to, so rotated phone video comes
    out the right way up without a transpose filter. Small frames are left alone: upscaling
    them would cost bytes without giving CLIP anything it did not already have.
    """
    chain = [
        f"scale='if(gt(iw,ih),min(iw,{max_side}),-2)':'if(gt(iw,ih),-2,min(ih,{max_side}))':flags=bicubic"
    ]
    if info.is_hdr and has_zscale():
        chain.insert(0, TONEMAP)
    args = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-ss",
        f"{max(seconds, 0.0):.3f}",
        "-i",
        str(path),
        "-frames:v",
        "1",
        "-vf",
        ",".join(chain),
        "-q:v",
        "3",
        "-f",
        "image2",
        "-c:v",
        "mjpeg",
        "-",
    ]
    result = _run(args, what=f"frame grab at {seconds:.2f}s in {path.name}", capture=True)
    return result.stdout


def extract_audio(path: Path, destination: Path) -> Path:
    """16 kHz mono FLAC, which is what Whisper wants and is ~4% of the video's size."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    args = [
        ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(path),
        "-vn",
        "-map",
        "0:a:0",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        str(destination),
    ]
    _run(args, what=f"audio extraction from {path.name}")
    return destination
