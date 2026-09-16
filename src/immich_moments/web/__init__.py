"""Single page web UI served by `immich-moments serve`."""

from .app import create_app

__all__ = ["create_app"]
