"""Configuration, read from a TOML file and overridden by the environment."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path

from .errors import ConfigError

DEFAULT_ML_URL = "http://localhost:3003"
DEFAULT_DATA_DIR = Path.home() / ".local" / "share" / "immich-moments"
CONFIG_FILENAME = "immich-moments.toml"

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


@dataclass(slots=True)
class Config:
    immich_url: str = ""
    immich_api_key: str = ""
    ml_url: str = DEFAULT_ML_URL
    data_dir: Path = DEFAULT_DATA_DIR

    # scene segmentation
    scene_threshold: float = 27.0
    min_scene_seconds: float = 1.5
    max_scene_seconds: float = 20.0

    # faces
    face_min_score: float = 0.7
    face_max_distance: float = 0.5

    # speech
    whisper_model: str = "small"
    whisper_device: str = "auto"
    whisper_compute_type: str = "default"
    whisper_language: str = ""
    whisper_beam_size: int = 5

    # search
    visual_weight: float = 0.65
    label_min_similarity: float = 0.22

    # transport
    request_timeout: float = 60.0
    download_timeout: float = 900.0
    max_retries: int = 5
    min_request_interval: float = 0.0

    # server
    host: str = "127.0.0.1"
    port: int = 8099

    @property
    def db_path(self) -> Path:
        return self.data_dir / "moments.sqlite3"

    @property
    def vectors_path(self) -> Path:
        return self.data_dir / "vectors.f32"

    @property
    def thumbs_dir(self) -> Path:
        return self.data_dir / "thumbs"

    @property
    def audio_dir(self) -> Path:
        return self.data_dir / "audio"

    @property
    def api_base(self) -> str:
        return self.immich_url.rstrip("/") + "/api"

    def require_credentials(self) -> None:
        if not self.immich_url:
            raise ConfigError(
                "IMMICH_URL is not set. Point it at your Immich server, e.g. IMMICH_URL=http://localhost:2283"
            )
        if not self.immich_api_key:
            raise ConfigError(
                "IMMICH_API_KEY is not set. Create one in Immich under Account Settings -> API Keys."
            )

    def ensure_dirs(self) -> None:
        for directory in (self.data_dir, self.thumbs_dir, self.audio_dir):
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ConfigError(f"cannot create {directory}: {exc}") from exc


def _coerce(name: str, raw: object, target: type):
    if target is Path:
        return Path(str(raw)).expanduser()
    if target is bool:
        if isinstance(raw, bool):
            return raw
        text = str(raw).strip().lower()
        if text in _BOOL_TRUE:
            return True
        if text in _BOOL_FALSE:
            return False
        raise ConfigError(f"{name}: expected a boolean, got {raw!r}")
    try:
        return target(raw)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name}: expected {target.__name__}, got {raw!r}") from exc


def _config_file(explicit: Path | None, data_dir: Path) -> Path | None:
    if explicit is not None:
        if not explicit.is_file():
            raise ConfigError(f"config file not found: {explicit}")
        return explicit
    from_env = os.environ.get("IMMICH_MOMENTS_CONFIG")
    if from_env:
        path = Path(from_env).expanduser()
        if not path.is_file():
            raise ConfigError(f"IMMICH_MOMENTS_CONFIG points at a missing file: {path}")
        return path
    for candidate in (Path.cwd() / CONFIG_FILENAME, data_dir / CONFIG_FILENAME):
        if candidate.is_file():
            return candidate
    return None


def load_config(config_path: Path | None = None, overrides: dict[str, object] | None = None) -> Config:
    """Build a Config from, in increasing precedence: defaults, TOML file, env, CLI overrides."""
    types = {f.name: f.type for f in fields(Config)}
    resolved: dict[str, object] = {}

    env_data_dir = os.environ.get("DATA_DIR")
    probe_dir = Path(env_data_dir).expanduser() if env_data_dir else DEFAULT_DATA_DIR
    path = _config_file(config_path, probe_dir)
    if path is not None:
        try:
            table = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigError(f"cannot read {path}: {exc}") from exc
        section = table.get("immich_moments", table)
        if not isinstance(section, dict):
            raise ConfigError(f"{path}: [immich_moments] must be a table")
        unknown = set(section) - set(types)
        if unknown:
            raise ConfigError(f"{path}: unknown option(s) {', '.join(sorted(unknown))}")
        resolved.update(section)

    env_map = {
        "immich_url": "IMMICH_URL",
        "immich_api_key": "IMMICH_API_KEY",
        "ml_url": "IMMICH_ML_URL",
        "data_dir": "DATA_DIR",
    }
    for name in types:
        var = env_map.get(name, "IMMICH_MOMENTS_" + name.upper())
        if var in os.environ and os.environ[var] != "":
            resolved[name] = os.environ[var]

    for name, value in (overrides or {}).items():
        if value is not None:
            resolved[name] = value

    kwargs = {name: _coerce(name, value, _unwrap(types[name])) for name, value in resolved.items()}
    config = Config(**kwargs)
    if not 0.0 <= config.visual_weight <= 1.0:
        raise ConfigError(f"visual_weight must be between 0 and 1, got {config.visual_weight}")
    return config


def _unwrap(annotation) -> type:
    """Dataclass field types arrive as strings under `from __future__ import annotations`."""
    if isinstance(annotation, str):
        return {"Path": Path, "float": float, "int": int, "bool": bool, "str": str}[annotation]
    return annotation
