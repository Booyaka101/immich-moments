"""Zero-shot scene labels and their cache."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import numpy as np
import pytest

from immich_moments.config import Config
from immich_moments.errors import ConfigError
from immich_moments.labels import BUILTIN_LABELS, LabelIndex, build_label_index, read_labels
from immich_moments.ml import MLClient

from conftest import unit

DIM = 8
CAKE = unit(21, DIM)
GARDEN = unit(22, DIM)


class CountingML:
    """A real MLClient plus a tally, so "embedded once, then cached" is checkable."""

    def __init__(self, config: Config, vectors: dict[str, np.ndarray]) -> None:
        self.calls: list[str] = []
        self.client = MLClient(
            config, "ViT-B-32__openai", "buffalo_l", transport=httpx.MockTransport(self._handler)
        )
        self.vectors = vectors

    def _handler(self, request: httpx.Request) -> httpx.Response:
        from urllib.parse import parse_qs

        text = parse_qs(request.content.decode())["text"][0]
        self.calls.append(text)
        vector = self.vectors.get(text, unit(abs(hash(text)) % 1000, DIM))
        return httpx.Response(200, json={"clip": json.dumps([float(x) for x in vector])})

    @property
    def clip_model(self) -> str:
        return self.client.clip_model

    def embed_text(self, text: str) -> np.ndarray:
        return self.client.embed_text(text)


@pytest.fixture
def labels_file(tmp_path: Path) -> Path:
    path = tmp_path / "labels.txt"
    path.write_text(
        "# vocabulary\na birthday cake\n\n   a garden   \n  # an indented comment\n",
        encoding="utf-8",
    )
    return path


def test_the_list_is_read_without_comments_or_blanks(labels_file: Path) -> None:
    """An indented comment used to survive as a label, and a label becomes a tag in Immich."""
    assert read_labels(labels_file) == ["a birthday cake", "a garden"]


def test_the_shipped_vocabulary_is_present_and_plausible() -> None:
    labels = read_labels()
    assert BUILTIN_LABELS.exists()
    assert len(labels) > 50
    assert len(set(labels)) == len(labels)
    assert all(label == label.strip() and label for label in labels)


def test_a_missing_list_names_the_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read the label list"):
        read_labels(tmp_path / "nope.txt")


def test_an_empty_list_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "empty.txt"
    path.write_text("# only comments\n\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="contains no labels"):
        read_labels(path)


def test_the_closest_label_wins() -> None:
    index = LabelIndex(["a birthday cake", "a garden"], np.stack([CAKE, GARDEN]), min_similarity=0.2)
    label, score = index.best(CAKE)
    assert label == "a birthday cake"
    assert score == pytest.approx(1.0, abs=1e-5)


def test_a_weak_match_is_no_label() -> None:
    index = LabelIndex(["a birthday cake"], np.stack([CAKE]), min_similarity=0.9)
    assert index.best(GARDEN) is None


def test_an_empty_index_labels_nothing() -> None:
    assert LabelIndex([], np.zeros((0, DIM), np.float32), min_similarity=0.0).best(CAKE) is None


def test_the_vocabulary_is_embedded_once_then_read_from_cache(
    config: Config, labels_file: Path, tmp_path: Path
) -> None:
    ml = CountingML(config, {"a birthday cake": CAKE, "a garden": GARDEN})

    first = build_label_index(ml, tmp_path, labels_path=labels_file, min_similarity=0.2)
    assert ml.calls == ["a birthday cake", "a garden"]
    assert first.matrix.shape == (2, DIM)

    second = build_label_index(ml, tmp_path, labels_path=labels_file, min_similarity=0.2)
    assert ml.calls == ["a birthday cake", "a garden"]  # no second round of requests
    assert np.array_equal(first.matrix, second.matrix)


def test_a_changed_vocabulary_gets_its_own_cache(config: Config, labels_file: Path, tmp_path: Path) -> None:
    ml = CountingML(config, {})
    build_label_index(ml, tmp_path, labels_path=labels_file, min_similarity=0.2)
    labels_file.write_text("a dog\n", encoding="utf-8")
    build_label_index(ml, tmp_path, labels_path=labels_file, min_similarity=0.2)

    assert ml.calls == ["a birthday cake", "a garden", "a dog"]
    assert len(list(tmp_path.glob("labels-*.npz"))) == 2


def test_a_corrupt_cache_is_rebuilt_not_raised(config: Config, labels_file: Path, tmp_path: Path) -> None:
    ml = CountingML(config, {})
    build_label_index(ml, tmp_path, labels_path=labels_file, min_similarity=0.2)
    cache = next(iter(tmp_path.glob("labels-*.npz")))
    cache.write_bytes(b"truncated")

    index = build_label_index(ml, tmp_path, labels_path=labels_file, min_similarity=0.2)

    assert index.matrix.shape == (2, DIM)
    assert ml.calls == ["a birthday cake", "a garden"] * 2


def test_an_unwritable_cache_directory_is_a_warning_not_a_failure(
    config: Config, labels_file: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ml = CountingML(config, {})
    blocked = tmp_path / "blocked"
    blocked.write_text("I am a file, not a directory", encoding="utf-8")

    with caplog.at_level("WARNING"):
        index = build_label_index(ml, blocked, labels_path=labels_file, min_similarity=0.2)

    assert index.matrix.shape == (2, DIM)


def test_labels_are_matched_in_the_same_space_as_the_frames(config: Config, tmp_path: Path) -> None:
    """The point of the whole module: a frame vector picks its nearest vocabulary entry."""
    path = tmp_path / "labels.txt"
    path.write_text("a birthday cake\na garden\n", encoding="utf-8")
    ml = CountingML(config, {"a birthday cake": CAKE, "a garden": GARDEN})

    index = build_label_index(ml, tmp_path, labels_path=path, min_similarity=0.2)
    label, score = index.best(GARDEN)

    assert label == "a garden"
    assert score == pytest.approx(1.0, abs=1e-5)
