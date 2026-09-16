"""Zero-shot scene labels.

There is no captioning model in this stack, so a scene's human-readable label comes from
matching its CLIP vector against a fixed vocabulary embedded with the same text encoder.
The embeddings are cached per (model, vocabulary) so the pass runs once, not once per video.

A scene only keeps its winning label if that label beats the rest of the vocabulary by enough,
measured in standard deviations of the scene's own scores rather than in raw cosine. Cosine is
not comparable across CLIP models: over the same 211 scenes and 234 labels, the winner scores a
median 0.256 under `ViT-B-32__openai` and 0.066 under `ViT-L-16-SigLIP-384__webli`, whose scores
average below zero. Standardised, both land on a median of 3.26.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
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
    def __init__(self, labels: list[str], matrix: np.ndarray, min_zscore: float) -> None:
        # One label is never a standard deviation above itself, and n labels cannot put the
        # winner further than sqrt(n-1) above the mean. A vocabulary too short for the floor
        # would label nothing at all, silently, which is the failure this floor exists to avoid.
        reachable = math.sqrt(len(labels) - 1) if labels else 0.0
        if labels and reachable < min_zscore:
            raise ConfigError(
                f"{len(labels)} label(s) cannot reach a z of {min_zscore:g}: the most any one of "
                f"them can sit above the others is {reachable:.2f}. Use a longer vocabulary, or "
                f"lower label_min_zscore below {reachable:.2f}."
            )
        self.labels = labels
        self.matrix = matrix
        self.min_zscore = min_zscore

    def best(self, vector: np.ndarray) -> tuple[str, float] | None:
        return self.best_many(vector.reshape(1, -1))[0]

    def best_many(self, vectors: np.ndarray) -> list[tuple[str, float] | None]:
        """Label a whole index in one matmul, which is what `relabel` needs."""
        if not self.labels or vectors.size == 0:
            return [None] * len(vectors)
        scores = vectors @ self.matrix.T
        # Standardising is monotonic within a row, so the winner is the same one argmax finds.
        winners = np.argmax(scores, axis=1)
        tops = scores[np.arange(len(vectors)), winners]
        spread = scores.std(axis=1)
        above = np.divide(tops - scores.mean(axis=1), spread, out=np.zeros_like(tops), where=spread > 0)
        return [
            (self.labels[int(winner)], float(z)) if z >= self.min_zscore else None
            for winner, z in zip(winners, above, strict=True)
        ]


def build_label_index(
    ml: MLClient, cache_dir: Path, *, labels_path: Path | None, min_zscore: float
) -> LabelIndex:
    labels = read_labels(labels_path)
    cache = cache_dir / f"labels-{_fingerprint(ml.clip_model, labels)}.npz"
    if cache.exists():
        try:
            with np.load(cache, allow_pickle=False) as data:
                return LabelIndex(labels, data["matrix"], min_zscore)
        except (OSError, ValueError, KeyError):
            cache.unlink(missing_ok=True)

    log.info("embedding %d scene labels with %s (once, then cached)", len(labels), ml.clip_model)
    matrix = np.stack([ml.embed_text(label) for label in labels]).astype(np.float32)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, matrix=matrix)
    except OSError as exc:
        log.warning("could not cache label embeddings at %s: %s", cache, exc)
    return LabelIndex(labels, matrix, min_zscore)


def _fingerprint(model: str, labels: list[str]) -> str:
    digest = hashlib.sha256(json.dumps([model, labels], ensure_ascii=False).encode()).hexdigest()
    return digest[:16]
