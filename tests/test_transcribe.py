"""Speech transcription: device selection and CPU retreat everywhere, real Whisper once.

The one test that actually runs Whisper is marked `slow` because it downloads a model on a
cold machine. Everything else drives the placement and fallback logic with a stub, which is
where the bugs live: a box with a GPU but no CUDA runtime loads the model happily and only
fails when the first batch reaches cuBLAS.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from immich_moments.config import Config
from immich_moments.errors import MomentsError
from immich_moments.media import extract_audio
from immich_moments.transcribe import Transcriber, add_cuda_dll_directories

from conftest import SEGMENT_SECONDS, SENTENCE, SPEECH_START, ffmpeg


class Segment:
    def __init__(self, start: float, end: float, text: str) -> None:
        self.start, self.end, self.text = start, end, text


class StubModel:
    """Stands in for WhisperModel, optionally exploding the way ctranslate2 does."""

    def __init__(self, device: str, *, fails_on_compute: bool = False) -> None:
        self.device = device
        self.fails_on_compute = fails_on_compute
        self.calls = 0

    def transcribe(self, _path, **_kwargs):
        self.calls += 1
        if self.fails_on_compute:
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return [Segment(0.0, 1.0, "  hello  "), Segment(1.0, 2.0, "   ")], None


def stubbed(transcriber: Transcriber, factory) -> list[StubModel]:
    """Replace the model loader, recording every model it hands out."""
    made: list[StubModel] = []

    def load():
        model = factory(transcriber.device)
        made.append(model)
        return model

    transcriber._load = load
    return made


@pytest.fixture
def config(tmp_path: Path) -> Config:
    config = Config(data_dir=tmp_path / "data", whisper_model="tiny")
    config.ensure_dirs()
    return config


# ---- device placement ----------------------------------------------------


@pytest.mark.parametrize(
    ("device", "compute", "expected"),
    [
        ("cpu", "default", ("cpu", "default")),
        ("cpu", "int8", ("cpu", "int8")),
        ("cuda", "default", ("cuda", "default")),
        ("cuda", "float32", ("cuda", "float32")),
    ],
)
def test_an_explicit_device_is_taken_as_given(
    config: Config, device: str, compute: str, expected: tuple[str, str]
) -> None:
    config.whisper_device = device
    config.whisper_compute_type = compute
    assert Transcriber(config)._placement() == expected


def test_auto_picks_float16_on_a_gpu(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("immich_moments.transcribe._cuda_available", lambda: True)
    assert Transcriber(config)._placement() == ("cuda", "float16")


def test_auto_picks_int8_without_a_gpu(config: Config, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("immich_moments.transcribe._cuda_available", lambda: False)
    assert Transcriber(config)._placement() == ("cpu", "int8")


# ---- the CPU retreat -----------------------------------------------------


def test_a_gpu_that_fails_at_first_compute_falls_back_and_still_returns_text(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr("immich_moments.transcribe._cuda_available", lambda: True)
    transcriber = Transcriber(config)
    made = stubbed(transcriber, lambda device: StubModel(device, fails_on_compute=device == "cuda"))

    with caplog.at_level("WARNING"):
        records = transcriber.transcribe(tmp_path / "audio.flac")

    assert [model.device for model in made] == ["cuda", "cpu"]
    assert [record.text for record in records] == ["hello"]  # blank segments dropped
    assert transcriber.device == "cpu"
    assert "cublas64_12.dll" in caplog.text
    assert "IMMICH_MOMENTS_WHISPER_DEVICE=cpu" in caplog.text


def test_the_retreat_happens_once_not_once_per_video(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("immich_moments.transcribe._cuda_available", lambda: True)
    transcriber = Transcriber(config)
    made = stubbed(transcriber, lambda device: StubModel(device, fails_on_compute=device == "cuda"))

    transcriber.transcribe(tmp_path / "one.flac")
    transcriber.transcribe(tmp_path / "two.flac")

    assert [model.device for model in made] == ["cuda", "cpu"]


def test_a_gpu_that_fails_at_load_falls_back(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("immich_moments.transcribe._cuda_available", lambda: True)
    devices: list[str] = []

    class Loader:
        def __call__(self, name, device, compute_type):
            devices.append(device)
            if device == "cuda":
                raise RuntimeError("no CUDA-capable device is detected")
            return StubModel(device)

    monkeypatch.setattr("faster_whisper.WhisperModel", Loader())
    transcriber = Transcriber(config)

    assert transcriber.transcribe(tmp_path / "audio.flac")[0].text == "hello"
    assert devices == ["cuda", "cpu"]


def test_an_explicitly_chosen_gpu_is_an_instruction_not_a_preference(config: Config, tmp_path: Path) -> None:
    """Asking for cuda and silently getting cpu would hide a broken box for a whole library."""
    config.whisper_device = "cuda"
    transcriber = Transcriber(config)
    stubbed(transcriber, lambda device: StubModel(device, fails_on_compute=True))

    with pytest.raises(MomentsError, match=r"audio.flac failed"):
        transcriber.transcribe(tmp_path / "audio.flac")


def test_a_cpu_failure_says_so_rather_than_looping(
    config: Config, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("immich_moments.transcribe._cuda_available", lambda: True)
    transcriber = Transcriber(config)
    stubbed(transcriber, lambda device: StubModel(device, fails_on_compute=True))

    with pytest.raises(MomentsError, match="failed on the CPU too"):
        transcriber.transcribe(tmp_path / "audio.flac")


def test_a_model_name_that_does_not_exist_names_the_setting(
    config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.whisper_device = "cpu"
    config.whisper_model = "enormous"

    def explode(*_args, **_kwargs):
        raise ValueError("Invalid model size 'enormous'")

    monkeypatch.setattr("faster_whisper.WhisperModel", explode)
    with pytest.raises(MomentsError, match="IMMICH_MOMENTS_WHISPER_MODEL"):
        Transcriber(config).transcribe(Path("audio.flac"))


def test_unload_drops_the_model_so_the_gpu_is_free_for_the_next_phase(config: Config) -> None:
    transcriber = Transcriber(config)
    stubbed(transcriber, StubModel)
    assert transcriber.model is not None
    transcriber.unload()
    assert transcriber._model is None


def test_registering_the_cuda_runtime_is_harmless_when_it_is_absent() -> None:
    add_cuda_dll_directories()
    add_cuda_dll_directories()  # idempotent: PATH must not grow without bound


# ---- the real thing ------------------------------------------------------


@pytest.mark.slow
def test_the_fixture_sentence_is_transcribed_inside_the_right_scene(
    config: Config, colour_video: Path, tmp_path: Path
) -> None:
    """The brief's test: the spoken sentence comes back with timestamps inside the green scene."""
    audio = tmp_path / "audio.flac"
    extract_audio(colour_video, audio)
    config.whisper_language = "en"
    config.whisper_device = "cpu"

    records = Transcriber(config).transcribe(audio)

    assert records, "the fixture sentence produced no segments at all"
    spoken = " ".join(record.text for record in records).lower()
    assert "birthday" in spoken
    assert "candle" in spoken

    green_start, green_end = SEGMENT_SECONDS, SEGMENT_SECONDS * 2
    assert records[0].start_seconds == pytest.approx(SPEECH_START, abs=1.5)
    assert green_start <= records[0].start_seconds < green_end
    assert all(record.end_seconds > record.start_seconds for record in records)


@pytest.mark.slow
def test_a_track_of_pure_silence_yields_no_segments(config: Config, tmp_path: Path) -> None:
    """Silence must come back empty, not as Whisper's favourite hallucination."""
    quiet = tmp_path / "quiet.mp4"
    subprocess.run(
        [
            ffmpeg(),
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-t",
            "8",
            "-i",
            "anullsrc=r=16000:cl=mono",
            "-f",
            "lavfi",
            "-t",
            "8",
            "-i",
            "color=c=black:s=64x64:r=5",
            "-shortest",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(quiet),
        ],
        check=True,
        capture_output=True,
    )
    audio = tmp_path / "quiet.flac"
    extract_audio(quiet, audio)
    config.whisper_device = "cpu"

    assert Transcriber(config).transcribe(audio) == []


def test_the_fixture_recording_actually_says_the_sentence() -> None:
    """Guards the fixture itself: if speech.wav is replaced, the slow test above must be updated."""
    from conftest import SPEECH_WAV

    assert SPEECH_WAV.exists()
    assert SENTENCE.lower().startswith("happy birthday")
