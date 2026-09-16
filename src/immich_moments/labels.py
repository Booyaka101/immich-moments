"""Zero-shot scene labels.

There is no captioning model in this stack, so a scene's human-readable label comes from
matching its CLIP vector against a fixed vocabulary embedded with the same text encoder.
The embeddings are cached per (model, vocabulary) so the pass runs once, not once per video.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .errors import ConfigError
from .ml import MLClient

log = logging.getLogger(__name__)

BUILTIN_LABELS = Path(__file__).parent / "data" / "scene_labels.txt"


def read_labels(path: Path | None = None) -> list[str]:
    source = path or BUILTIN_LABELS
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ConfigError(f"cannot read the label list {source}: {exc}") from exc
    stripped = (line.strip() for line in lines)
    labels = [line for line in stripped if line and not line.startswith("#")]
    if not labels:
        raise ConfigError(f"{source} contains no labels")
    return labels


class LabelIndex:
    def __init__(self, labels: list[str], matrix: np.ndarray, min_similarity: float) -> None:
        self.labels = labels
        self.matrix = matrix
        self.min_similarity = min_similarity

    def best(self, vector: np.ndarray) -> tuple[str, float] | None:
        return self.best_many(vector.reshape(1, -1))[0]

    def best_many(self, vectors: np.ndarray) -> list[tuple[str, float] | None]:
        """Label a whole index in one matmul, which is what `relabel` needs."""
        if not self.labels or vectors.size == 0:
            return [None] * len(vectors)
        scores = vectors @ self.matrix.T
        winners = np.argmax(scores, axis=1)
        tops = scores[np.arange(len(vectors)), winners]
        return [
            (self.labels[int(winner)], float(top)) if top >= self.min_similarity else None
            for winner, top in zip(winners, tops, strict=True)
        ]


def build_label_index(
    ml: MLClient, cache_dir: Path, *, labels_path: Path | None, min_similarity: float
) -> LabelIndex:
    labels = read_labels(labels_path)
    cache = cache_dir / f"labels-{_fingerprint(ml.clip_model, labels)}.npz"
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as data:
                return LabelIndex(labels, data["matrix"], min_similarity)
        except (OSError, ValueError, KeyError):
            cache.unlink(missing_ok=True)

    log.info("embedding %d scene labels with %s (once, then cached)", len(labels), ml.clip_model)
    matrix = np.stack([ml.embed_text(label) for label in labels]).astype(np.float32)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, matrix=matrix)
    except OSError as exc:
        log.warning("could not cache label embeddings at %s: %s", cache, exc)
    return LabelIndex(labels, matrix, min_similarity)


def _fingerprint(model: str, labels: list[str]) -> str:
    digest = hashlib.sha256(json.dumps([model, labels], ensure_ascii=False).encode()).hexdigest()
    return digest[:16]
