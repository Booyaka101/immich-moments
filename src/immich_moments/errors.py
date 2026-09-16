"""Exceptions that carry a message meant for a terminal, not a traceback."""


class MomentsError(Exception):
    """Base for every failure the CLI turns into a one-line message."""

    exit_code = 1


class ConfigError(MomentsError):
    exit_code = 2


class ImmichError(MomentsError):
    """The Immich server refused or could not answer a request."""

    exit_code = 3

    def __init__(self, message: str, *, status: int | None = None, url: str | None = None) -> None:
        super().__init__(message)
        self.status = status
        self.url = url


class MLError(MomentsError):
    """The machine-learning container refused or could not answer a request."""

    exit_code = 4


class AssetUnavailable(MomentsError):
    """The original file could not be downloaded; the asset is skipped, not fatal."""

    exit_code = 0


class MediaError(MomentsError):
    """ffmpeg/ffprobe could not read the file."""

    exit_code = 5


class StorageError(MomentsError):
    """The data directory could not be read or written: a full disk, or the wrong permissions."""

    exit_code = 7


class DimensionMismatch(MomentsError):
    """Stored vectors were produced by a different model than the one configured now."""

    exit_code = 6
