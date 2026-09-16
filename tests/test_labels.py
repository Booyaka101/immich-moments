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

# Five labels along the first five axes, so a scene vector's coordinates are its scores and a
# test can say what the score row is instead of hoping two random vectors land where it wants.
VOCABULARY = [f"L{i}" for i in range(5)]
EYE = np.eye(len(VOCABULARY), DIM, dtype=np.float32)


def scene(*scores: float) -> np.ndarray:
    return np.array(list(scores) + [0.0] * (DIM - len(scores)), dtype=np.float32)


# One label far ahead of four identical ones is as far ahead as five labels can put it: the most
# any of n can sit above the mean is sqrt(n-1), which is 2 here. The muddy row lands at 1.58.
CLEAR = (9.0, 1.0, 1.0, 1.0, 1.0)
MUDDY = (3.0, 2.0, 2.0, 2.0, 1.0)


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
    index = LabelIndex(VOCABULARY, EYE, min_zscore=0.2)
    label, z = index.best(scene(*CLEAR))
    assert label == "L0"
    assert z == pytest.approx(2.0)


def test_a_weak_match_is_no_label() -> None:
    """Four labels nearly level and a fifth barely ahead is not a caption anyone wants."""
    index = LabelIndex(VOCABULARY, EYE, min_zscore=1.9)
    assert index.best(scene(*MUDDY)) is None


def test_the_floor_does_not_move_when_the_model_does() -> None:
    """The bug: 0.22 was a ViT-B-32 cosine, and SigLIP's scores sit near zero, most of them below.

    Sliding and squeezing a scene's whole score row is what a change of CLIP model does to it,
    and a floor worth having survives that.
    """
    index = LabelIndex(VOCABULARY, EYE, min_zscore=1.5)
    squeezed = scene(*(0.13 * score - 0.4 for score in MUDDY))

    assert index.best(squeezed)[0] == index.best(scene(*MUDDY))[0]
    assert index.best(squeezed)[1] == pytest.approx(index.best(scene(*MUDDY))[1])


def test_a_vocabulary_too_short_for_the_floor_is_refused() -> None:
    """Otherwise it quietly labels nothing at all, which is the failure this floor is for."""
    with pytest.raises(ConfigError, match=r"cannot reach a z of 2.3"):
        LabelIndex(["a birthday cake", "a garden"], np.eye(2, DIM, dtype=np.float32), min_zscore=2.3)


def test_labels_that_all_score_the_same_are_no_label() -> None:
    """No spread means no winner, and dividing by it would be a nan sneaking into the index."""
    assert LabelIndex(VOCABULARY, EYE, min_zscore=0.0).best(scene(1, 1, 1, 1, 1)) == ("L0", 0.0)
    assert LabelIndex(VOCABULARY, EYE, min_zscore=0.1).best(scene(1, 1, 1, 1, 1)) is None


def test_an_empty_index_labels_nothing() -> None:
    assert LabelIndex([], np.zeros((0, DIM), np.float32), min_zscore=0.0).best(CAKE) is None


def test_a_whole_batch_is_labelled_in_one_pass() -> None:
    """`relabel` scores the entire index at once, so the batch must agree with `best`."""
    index = LabelIndex(VOCABULARY, EYE, min_zscore=0.2)
    scenes = np.stack([scene(1, 9, 1, 1, 1), scene(*CLEAR)])

    picks = index.best_many(scenes)

    assert [label for label, _z in picks] == ["L1", "L0"]
    assert picks[0] == index.best(scenes[0])


def test_a_batch_keeps_the_weak_ones_unlabelled() -> None:
    index = LabelIndex(VOCABULARY, EYE, min_zscore=1.9)

    assert index.best_many(np.stack([scene(*CLEAR), scene(*MUDDY)])) == [
        ("L0", pytest.approx(2.0)),
        None,
    ]


def test_an_empty_batch_is_not_a_matmul_error() -> None:
    index = LabelIndex(VOCABULARY, EYE, min_zscore=0.0)

    assert index.best_many(np.zeros((0, DIM), np.float32)) == []


def test_the_vocabulary_is_embedded_once_then_read_from_cache(
    config: Config, labels_file: Path, tmp_path: Path
) -> None:
    ml = CountingML(config, {"a birthday cake": CAKE, "a garden": GARDEN})

    first = build_label_index(ml, tmp_path, labels_path=labels_file, min_zscore=0.0)
    assert ml.calls == ["a birthday cake", "a garden"]
    assert first.matrix.shape == (2, DIM)

    second = build_label_index(ml, tmp_path, labels_path=labels_file, min_zscore=0.0)
    assert ml.calls == ["a birthday cake", "a garden"]  # no second round of requests
    assert np.array_equal(first.matrix, second.matrix)


def test_a_changed_vocabulary_gets_its_own_cache(config: Config, labels_file: Path, tmp_path: Path) -> None:
    ml = CountingML(config, {})
    build_label_index(ml, tmp_path, labels_path=labels_file, min_zscore=0.0)
    labels_file.write_text("a dog\n", encoding="utf-8")
    build_label_index(ml, tmp_path, labels_path=labels_file, min_zscore=0.0)

    assert ml.calls == ["a birthday cake", "a garden", "a dog"]
    assert len(list(tmp_path.glob("labels-*.npz"))) == 2


def test_a_corrupt_cache_is_rebuilt_not_raised(config: Config, labels_file: Path, tmp_path: Path) -> None:
    ml = CountingML(config, {})
    build_label_index(ml, tmp_path, labels_path=labels_file, min_zscore=0.0)
    cache = next(iter(tmp_path.glob("labels-*.npz")))
    cache.write_bytes(b"truncated")

    index = build_label_index(ml, tmp_path, labels_path=labels_file, min_zscore=0.0)

    assert index.matrix.shape == (2, DIM)
    assert ml.calls == ["a birthday cake", "a garden"] * 2


def test_an_unwritable_cache_directory_is_a_warning_not_a_failure(
    config: Config, labels_file: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    ml = CountingML(config, {})
    blocked = tmp_path / "blocked"
    blocked.write_text("I am a file, not a directory", encoding="utf-8")

    with caplog.at_level("WARNING"):
        index = build_label_index(ml, blocked, labels_path=labels_file, min_zscore=0.0)

    assert index.matrix.shape == (2, DIM)


def test_labels_are_matched_in_the_same_space_as_the_frames(config: Config, tmp_path: Path) -> None:
    """The point of the whole module: a frame vector picks its nearest vocabulary entry."""
    path = tmp_path / "labels.txt"
    path.write_text("a birthday cake\na garden\n", encoding="utf-8")
    ml = CountingML(config, {"a birthday cake": CAKE, "a garden": GARDEN})

    index = build_label_index(ml, tmp_path, labels_path=path, min_zscore=0.0)

    assert index.best(GARDEN)[0] == "a garden"
    assert index.best(CAKE)[0] == "a birthday cake"
