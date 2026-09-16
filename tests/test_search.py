"""Blended ranking over a seeded index.

The ML container is replaced by a mock transport that hands back a chosen CLIP vector, so the
ranking is tested for what it does with a vector rather than for what a model produces.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from immich_moments.config import Config
from immich_moments.ml import MLClient
from immich_moments.search import format_timestamp, fts_query, search
from immich_moments.store import FaceRecord, SceneRecord, Store, TranscriptRecord, VectorFile

from conftest import unit

NOW = "2026-09-16T10:00:00Z"
DIM = 8

CANDLES = unit(11, DIM)
CAKE = unit(12, DIM)
TABLE = unit(13, DIM)
GARDEN = unit(14, DIM)


def ml_returning(config: Config, vector: np.ndarray) -> MLClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"clip": json.dumps([float(x) for x in vector])})

    return MLClient(config, "ViT-B-32__openai", "buffalo_l", transport=httpx.MockTransport(handler))


@pytest.fixture
def seeded(store: Store, config: Config) -> Store:
    """Two assets: a birthday with three scenes, and an unrelated garden clip."""
    store.check_model("ViT-B-32__openai", DIM, reindex=False)
    vectors = VectorFile(config.vectors_path, DIM)
    store.upsert_asset(
        "birthday",
        original_file_name="birthday.mp4",
        file_created_at="2026-06-01T00:00:00Z",
        updated_at="2026-06-01T00:00:00Z",
        duration_seconds=30.0,
    )
    store.replace_scenes(
        "birthday",
        [
            SceneRecord(0, 0.0, 10.0, vector=CAKE, label="birthday cake", thumb_path="birthday-0.jpg"),
            SceneRecord(
                1,
                10.0,
                20.0,
                vector=CANDLES,
                label="blowing out candles",
                thumb_path="birthday-1.jpg",
                faces=[
                    FaceRecord("p1", "Anna", 0.18, 0.99, (1, 2, 3, 4)),
                    FaceRecord("p2", "Tom", 0.24, 0.97, (5, 6, 7, 8)),
                ],
            ),
            SceneRecord(2, 20.0, 30.0, vector=TABLE, label="a table", thumb_path="birthday-2.jpg"),
        ],
        vectors,
        indexed_at=NOW,
    )
    store.replace_transcript(
        "birthday",
        [
            TranscriptRecord(11.0, 14.0, "happy birthday to you"),
            TranscriptRecord(21.0, 23.0, "who wants a slice"),
        ],
        indexed_at=NOW,
        has_audio=True,
    )

    store.upsert_asset(
        "garden",
        original_file_name="garden.mp4",
        file_created_at="2026-05-01T00:00:00Z",
        updated_at="2026-05-01T00:00:00Z",
        duration_seconds=10.0,
    )
    store.replace_scenes(
        "garden", [SceneRecord(0, 0.0, 10.0, vector=GARDEN, label="a garden")], vectors, indexed_at=NOW
    )
    return store


def test_a_visual_query_ranks_the_matching_scene_first(seeded: Store, config: Config) -> None:
    with ml_returning(config, CANDLES) as ml:
        hits = search(seeded, ml, "blowing out candles", visual_weight=1.0)

    assert hits[0].asset_id == "birthday"
    assert hits[0].scene_index == 1
    assert hits[0].label == "blowing out candles"
    assert hits[0].timestamp == "00:10"
    assert hits[0].people == ["Anna", "Tom"]
    assert hits[0].transcript == "happy birthday to you"
    assert hits[0].immich_url("http://localhost:2283/") == "http://localhost:2283/photos/birthday"


def test_a_spoken_phrase_ranks_the_scene_it_was_said_in(seeded: Store, config: Config) -> None:
    """Weight 0 means the vectors are never consulted, so the ML container is not even called."""

    def refuse(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("a text-only search must not embed the query")

    with MLClient(config, "c", "f", transport=httpx.MockTransport(refuse)) as ml:
        hits = search(seeded, ml, "happy birthday", visual_weight=0.0)

    assert hits[0].scene_index == 1
    assert hits[0].text_score != 0.0
    assert hits[0].visual_score == 0.0


def test_the_blend_prefers_a_scene_that_wins_on_both(seeded: Store, config: Config) -> None:
    """The cake scene wins on vectors alone; the candles scene wins once speech is weighed in."""
    with ml_returning(config, CAKE) as ml:
        visual_only = search(seeded, ml, "happy birthday", visual_weight=1.0)
        blended = search(seeded, ml, "happy birthday", visual_weight=0.35)

    assert visual_only[0].scene_index == 0
    assert blended[0].scene_index == 1
    assert blended[0].visual_score != 0.0
    assert blended[0].text_score != 0.0


def test_a_search_can_be_pinned_to_one_asset(seeded: Store, config: Config) -> None:
    with ml_returning(config, GARDEN) as ml:
        everywhere = search(seeded, ml, "a garden", visual_weight=1.0)
        pinned = search(seeded, ml, "a garden", visual_weight=1.0, asset_id="birthday")

    assert everywhere[0].asset_id == "garden"
    assert {hit.asset_id for hit in pinned} == {"birthday"}


def test_limit_is_honoured(seeded: Store, config: Config) -> None:
    with ml_returning(config, CANDLES) as ml:
        assert len(search(seeded, ml, "anything", visual_weight=1.0, limit=2)) == 2


def test_an_empty_query_returns_nothing_without_asking_the_model(seeded: Store, config: Config) -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("an empty query must not reach the model")

    with MLClient(config, "c", "f", transport=httpx.MockTransport(refuse)) as ml:
        assert search(seeded, ml, "   ") == []


def test_searching_an_empty_index_is_not_an_error(store: Store, config: Config) -> None:
    with ml_returning(config, CANDLES) as ml:
        assert search(store, ml, "anything") == []


def test_punctuation_cannot_break_the_fts_query(seeded: Store, config: Config) -> None:
    with ml_returning(config, CANDLES) as ml:
        hits = search(seeded, ml, 'happy "birthday" -- to you!', visual_weight=0.0)
    assert hits[0].scene_index == 1


def test_fts_query_quotes_every_token() -> None:
    assert fts_query("Happy Birthday!") == '"happy" OR "birthday"'
    assert fts_query("  -- ") == ""
    assert fts_query("d'accord") == '"d" OR "accord"'


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0.0, "00:00"), (7.4, "00:07"), (462.0, "07:42"), (3723.0, "1:02:03"), (-5.0, "00:00")],
)
def test_timestamp_formatting(seconds: float, expected: str) -> None:
    assert format_timestamp(seconds) == expected


# ---- calibration ---------------------------------------------------------


def test_a_speech_hit_reports_when_the_words_were_said(seeded: Store, config: Config) -> None:
    """The scene starts at 00:10 but the line lands at 00:11, and that is where you want sending."""
    with ml_returning(config, GARDEN) as ml:
        hits = search(seeded, ml, "happy birthday", visual_weight=0.0)

    assert hits[0].spoken_at_seconds == 11.0
    assert hits[0].moment_seconds == 11.0
    assert hits[0].timestamp == "00:11"
    assert hits[0].start_seconds == 10.0  # the scene itself is unchanged


def test_a_visual_hit_reports_the_cut(seeded: Store, config: Config) -> None:
    with ml_returning(config, TABLE) as ml:
        hits = search(seeded, ml, "a table", visual_weight=1.0)

    assert hits[0].spoken_at_seconds is None
    assert hits[0].moment_seconds == hits[0].start_seconds
    assert hits[0].timestamp == "00:20"


def test_the_quote_leads_with_the_line_that_matched(seeded: Store, config: Config) -> None:
    seeded.replace_transcript(
        "birthday",
        [
            TranscriptRecord(10.5, 12.0, "someone get the lights"),
            TranscriptRecord(12.0, 14.0, "happy birthday to you"),
        ],
        indexed_at=NOW,
        has_audio=True,
    )
    with ml_returning(config, GARDEN) as ml:
        hits = search(seeded, ml, "happy birthday", visual_weight=0.0)

    assert hits[0].transcript == "happy birthday to you"
    assert hits[0].timestamp == "00:12"


def cone(store: Store, config: Config, nudges: list[float]) -> np.ndarray:
    """Seed one asset whose scenes sit in a tight cone, the way real CLIP vectors do.

    `nudges` is how far each scene is pushed off the query direction, so a test can choose
    between a channel with no opinion at all and one with a mild, meaningless favourite.
    Speech lands in the last scene either way.
    """
    store.check_model("ViT-B-32__openai", DIM, reindex=False)
    vectors = VectorFile(config.vectors_path, DIM)
    store.upsert_asset(
        "cone",
        original_file_name="cone.mp4",
        file_created_at=None,
        updated_at=None,
        duration_seconds=10.0 * len(nudges),
    )
    base = unit(31, DIM)
    scenes = []
    for idx, nudge in enumerate(nudges):
        pushed = (base + nudge * unit(40 + idx, DIM)).astype(np.float32)
        scenes.append(SceneRecord(idx, idx * 10.0, idx * 10.0 + 10.0, vector=pushed / np.linalg.norm(pushed)))
    store.replace_scenes("cone", scenes, vectors, indexed_at=NOW)
    store.replace_transcript(
        "cone",
        [
            TranscriptRecord(1.0, 2.0, "nothing to report"),
            TranscriptRecord(12.0, 13.0, "still nothing"),
            TranscriptRecord((len(nudges) - 1) * 10.0 + 1.0, 33.0, "safe descent rhea"),
        ],
        indexed_at=NOW,
        has_audio=True,
    )
    return base


def test_a_flat_visual_channel_does_not_manufacture_a_winner(store: Store, config: Config) -> None:
    """Min-max used to hand the best of an indifferent set a perfect score, burying real matches."""
    base = cone(store, config, [0.002, 0.004, 0.006, 0.008])

    with ml_returning(config, base) as ml:
        hits = search(store, ml, "safe descent", visual_weight=0.65)

    cosines = [hit.visual_score for hit in hits]
    assert max(cosines) - min(cosines) < 0.01  # nobody is a standout
    assert hits[0].scene_index == 3  # so the scene holding the words wins


def test_a_mild_visual_favourite_does_not_outrank_the_spoken_words(store: Store, config: Config) -> None:
    """Standard scores divided out the very spread that says whether CLIP found anything.

    A query with nothing to look at still leaves one scene a few hundredths of cosine clear of
    the rest, which is what CLIP does when it has no opinion, and that used to beat a verbatim
    transcript hit at the default weight.
    """
    base = cone(store, config, [0.45, 0.55, 0.60, 0.65])

    with ml_returning(config, base) as ml:
        hits = search(store, ml, "safe descent", visual_weight=0.65)

    lead = max(hit.visual_score for hit in hits) - np.mean([hit.visual_score for hit in hits])
    assert 0.02 < lead < 0.08  # the regime a real no-match query lands in
    assert hits[0].scene_index == 3


def test_every_transcript_match_keeps_a_score_above_zero(seeded: Store, config: Config) -> None:
    """A weak match still matched; min-max used to send the worst of them to exactly zero."""
    with ml_returning(config, GARDEN) as ml:
        hits = search(seeded, ml, "happy birthday who wants a slice", visual_weight=0.0)

    speech = [hit for hit in hits if hit.text_score != 0.0]
    assert len(speech) == 2
    assert all(hit.score > 0.0 for hit in speech)
