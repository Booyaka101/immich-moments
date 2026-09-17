"""Speech transcription with faster-whisper.

Kept in its own indexing phase: CLIP inference happens inside Immich's ML container and
Whisper runs here, and on a single-GPU box loading both at once is how you get an
out-of-memory error two hours into an overnight run.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import os
import sys
from pathlib import Path

from .config import Config
from .errors import MomentsError
from .store import TranscriptRecord

log = logging.getLogger(__name__)

CUDA_HINT = (
    "Install the CUDA runtime with `pip install immich-moments[cuda]`, or set "
    "IMMICH_MOMENTS_WHISPER_DEVICE=cpu to stop trying."
)


class Transcriber:
    """Lazily loads one Whisper model and reuses it for the whole phase.

    ctranslate2 only touches cuBLAS when it runs its first batch, so a box with a GPU but no
    CUDA runtime loads the model happily and then fails on the first video. Both the load and
    the first transcription therefore fall back to the CPU when the device was picked
    automatically; an explicit `whisper_device` is taken as an instruction, not a preference.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device, self.compute_type = self._placement()
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = self._load()
        return self._model

    def unload(self) -> None:
        self._model = None

    def transcribe(self, audio: Path) -> list[TranscriptRecord]:
        try:
            return self._run(audio)
        except (RuntimeError, OSError, ValueError) as exc:
            if not self._retreat_to_cpu(exc):
                raise MomentsError(f"transcription of {audio.name} failed: {exc}") from exc
        try:
            return self._run(audio)
        except (RuntimeError, OSError, ValueError) as exc:
            raise MomentsError(f"transcription of {audio.name} failed on the CPU too: {exc}") from exc

    # ---- internals -------------------------------------------------------

    def _run(self, audio: Path) -> list[TranscriptRecord]:
        segments, _info = self.model.transcribe(
            str(audio),
            language=self.config.whisper_language or None,
            beam_size=self.config.whisper_beam_size,
            vad_filter=True,
            condition_on_previous_text=False,
        )
        return [
            TranscriptRecord(float(s.start), float(s.end), text) for s in segments if (text := s.text.strip())
        ]

    def _placement(self) -> tuple[str, str]:
        # The compute type follows the device it lands on, not how that device was chosen.
        # Leaving it to ctranslate2 makes an explicit whisper_device=cpu slower than letting
        # the same CPU be picked automatically: it reads float16 off the checkpoint and
        # settles for float32.
        device = self.config.whisper_device
        if device == "auto":
            device = "cuda" if _cuda_available() else "cpu"
        return device, self._compute_for(device)

    def _compute_for(self, device: str) -> str:
        compute = self.config.whisper_compute_type
        if compute == "default":
            return "float16" if device == "cuda" else "int8"
        return compute

    def _load(self):
        if self.device == "cuda":
            register_cuda_runtime()
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise MomentsError(
                "faster-whisper is not installed. Reinstall with `pip install immich-moments`."
            ) from exc

        log.info("loading Whisper %s on %s (%s)", self.config.whisper_model, self.device, self.compute_type)
        try:
            return WhisperModel(self.config.whisper_model, device=self.device, compute_type=self.compute_type)
        except (ValueError, RuntimeError, OSError) as exc:
            if not self._retreat_to_cpu(exc):
                raise MomentsError(
                    f"could not load the Whisper model {self.config.whisper_model!r} on "
                    f"{self.device}: {exc}. Set IMMICH_MOMENTS_WHISPER_DEVICE=cpu, or pick a "
                    "smaller model with IMMICH_MOMENTS_WHISPER_MODEL=base."
                ) from exc
            return self._load()

    def _retreat_to_cpu(self, exc: Exception) -> bool:
        """True once the CPU has been selected, so the caller can retry."""
        if self.device != "cuda" or self.config.whisper_device != "auto":
            return False
        log.warning(
            "Whisper could not use the GPU (%s). Falling back to the CPU, which is several times slower. %s",
            exc,
            CUDA_HINT,
        )
        self.device, self.compute_type = "cpu", self._compute_for("cpu")
        self._model = None
        return True


def register_cuda_runtime() -> None:
    """Make the pip-installed CUDA runtime visible to ctranslate2.

    `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` drop their libraries inside site-packages
    rather than anywhere the loader looks, so without this ctranslate2 reports `cublas64_12.dll
    is not found` on Windows, or `libcublas.so.12 is not found` elsewhere, on a machine with a
    perfectly good GPU.
    """
    try:
        import nvidia
    except ImportError:
        log.debug("no pip-installed CUDA runtime; relying on a system-wide one. %s", CUDA_HINT)
        return
    roots = [Path(root) for root in nvidia.__path__]
    if sys.platform == "win32":
        _add_dll_directories([d for root in roots for d in root.glob("*/bin") if d.is_dir()])
        return
    _preload([lib for root in roots for lib in sorted(root.glob("*/lib/lib*.so.*"))])


def _add_dll_directories(directories: list[Path]) -> None:
    for directory in directories:
        with contextlib.suppress(OSError):
            os.add_dll_directory(str(directory))
    # ctranslate2 loads cuBLAS with a plain LoadLibrary, which consults PATH and not the
    # directories add_dll_directory registers, so both are needed.
    on_path = os.environ.get("PATH", "").split(os.pathsep)
    missing = [str(d) for d in directories if str(d) not in on_path]
    if missing:
        os.environ["PATH"] = os.pathsep.join([*missing, os.environ.get("PATH", "")])


def _preload(libraries: list[Path]) -> None:
    """Load each library by path, which is what lets a later dlopen find it by soname.

    The loader would otherwise want LD_LIBRARY_PATH, and that is read at exec, too late for
    anything this process can set.
    """
    for library in libraries:
        with contextlib.suppress(OSError):
            ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _cuda_available() -> bool:
    """ctranslate2 reports its own CUDA device count, so torch is not needed to ask."""
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, AttributeError, RuntimeError):
        return False
