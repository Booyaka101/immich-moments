"""Blended ranking over a seeded index.

The ML container is replaced by a mock transport that hands back a chosen CLIP vector, so the
ranking is tested for what it does with a vector rather than for what a model produces.
"""

from __future__ import annotations

import json
import sqlite3
from urllib.parse import parse_qs

import httpx
import numpy as np
import pytest

from immich_moments import search as search_module
from immich_moments.config import Config
from immich_moments.errors import ConfigError
from immich_moments.ml import MLClient
from immich_moments.search import (
    REFERENCE_PHRASES,
    Filters,
    browse,
    date_range,
    format_timestamp,
    fts_query,
    resolve_albums,
    resolve_people,
    search,
    similar,
    visual_reference,
)
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
        pinned = search(seeded, ml, "a garden", visual_weight=1.0, filters=Filters(asset_id="birthday"))

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


def test_a_person_filter_keeps_only_the_scenes_they_are_in(seeded: Store, config: Config) -> None:
    """Anna is in scene 1 of the birthday and nowhere else, so nothing else can rank."""
    anna = Filters(people=("Anna",))
    with ml_returning(config, GARDEN) as ml:
        hits = search(seeded, ml, "a garden", visual_weight=1.0, filters=anna)

    assert [(hit.asset_id, hit.scene_index) for hit in hits] == [("birthday", 1)]
    assert hits[0].people == ["Anna", "Tom"]


def test_a_person_filter_reaches_the_speech_channel_too(seeded: Store, config: Config) -> None:
    """ "who wants a slice" is said in scene 2, which Anna is not in."""

    def refuse(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("a text-only search must not embed the query")

    with MLClient(config, "c", "f", transport=httpx.MockTransport(refuse)) as ml:
        unfiltered = search(seeded, ml, "who wants a slice", visual_weight=0.0)
        filtered = search(
            seeded, ml, "who wants a slice", visual_weight=0.0, filters=Filters(people=("Anna",))
        )

    assert unfiltered[0].scene_index == 2
    assert filtered == []


def test_a_person_matches_whatever_case_you_type(seeded: Store, config: Config) -> None:
    with ml_returning(config, CANDLES) as ml:
        hits = search(seeded, ml, "candles", visual_weight=1.0, filters=Filters(people=("  aNNa ",)))

    assert [hit.scene_index for hit in hits] == [1]


def test_a_filter_with_no_query_browses_without_asking_the_model(seeded: Store, config: Config) -> None:
    def refuse(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("browsing must not embed anything")

    with MLClient(config, "c", "f", transport=httpx.MockTransport(refuse)) as ml:
        hits = search(seeded, ml, "   ", filters=Filters(people=("Tom",)))

    assert [(hit.asset_id, hit.scene_index) for hit in hits] == [("birthday", 1)]
    assert hits[0].score == 0.0
    assert hits[0].transcript == "happy birthday to you"


def test_browsing_puts_the_newest_video_first(seeded: Store) -> None:
    """The garden is a month older than the birthday, and scenes stay in playing order."""
    hits = browse(seeded, Filters())

    assert [(hit.asset_id, hit.scene_index) for hit in hits] == [
        ("birthday", 0),
        ("birthday", 1),
        ("birthday", 2),
        ("garden", 0),
    ]
    assert hits[0].file_created_at == "2026-06-01T00:00:00Z"


def test_browsing_honours_the_limit(seeded: Store) -> None:
    assert len(browse(seeded, Filters(), limit=2)) == 2


def test_two_people_means_both_of_them_in_the_same_scene(seeded: Store) -> None:
    """Anna and Tom share scene 1; nobody shares a scene with a name that is not there."""
    assert [hit.scene_index for hit in browse(seeded, Filters(people=("Anna", "Tom")))] == [1]
    assert browse(seeded, Filters(people=("Anna", "Ruth"))) == []


def scene_ids(store: Store) -> dict[str, int]:
    """Scene id by label, because the ids themselves depend on insertion order."""
    rows = [*store.scenes_for("birthday"), *store.scenes_for("garden")]
    return {row["label"]: row["id"] for row in rows}


def test_more_like_this_ranks_the_nearest_vectors_first(seeded: Store) -> None:
    ids = scene_ids(seeded)
    others = {"birthday cake": CAKE, "a table": TABLE, "a garden": GARDEN}
    expected = [ids[label] for label in sorted(others, key=lambda k: -float(CANDLES @ others[k]))]

    reference, hits = similar(seeded, ids["blowing out candles"], limit=5)

    assert reference.scene_id == ids["blowing out candles"]
    assert reference.label == "blowing out candles"
    assert [hit.scene_id for hit in hits] == expected
    assert hits[0].score == pytest.approx(float(CANDLES @ others[hits[0].label]), abs=1e-6)


def test_more_like_this_never_returns_the_scene_you_asked_about(seeded: Store) -> None:
    ids = scene_ids(seeded)

    _reference, hits = similar(seeded, ids["a garden"], limit=10)

    assert len(hits) == 3
    assert ids["a garden"] not in [hit.scene_id for hit in hits]


def test_more_like_this_obeys_the_limit(seeded: Store) -> None:
    _reference, hits = similar(seeded, scene_ids(seeded)["a garden"], limit=2)
    assert len(hits) == 2


def test_more_like_this_can_be_narrowed_to_a_person(seeded: Store) -> None:
    """Only the reference scene has Anna in it, so the honest answer is nothing else."""
    ids = scene_ids(seeded)

    reference, hits = similar(seeded, ids["a garden"], filters=Filters(people=("Anna",)))

    assert reference.label == "a garden"
    assert [hit.label for hit in hits] == ["blowing out candles"]


def test_a_scene_that_is_not_there_says_so(seeded: Store) -> None:
    with pytest.raises(ConfigError, match="not in the index"):
        similar(seeded, 9999)


def test_a_date_range_keeps_only_the_videos_taken_inside_it(seeded: Store, config: Config) -> None:
    """The garden was filmed in May and the birthday in June."""
    with ml_returning(config, GARDEN) as ml:
        june = search(seeded, ml, "a garden", visual_weight=1.0, filters=Filters(since="2026-06-01"))
        may = search(seeded, ml, "a garden", visual_weight=1.0, filters=Filters(until="2026-05-31"))

    assert {hit.asset_id for hit in june} == {"birthday"}
    assert {hit.asset_id for hit in may} == {"garden"}


def test_both_ends_of_the_range_include_their_own_day(seeded: Store, config: Config) -> None:
    with ml_returning(config, CAKE) as ml:
        hits = search(
            seeded, ml, "cake", visual_weight=1.0, filters=Filters(since="2026-06-01", until="2026-06-01")
        )

    assert {hit.asset_id for hit in hits} == {"birthday"}


def test_a_date_range_narrows_the_spoken_words_too(seeded: Store, config: Config) -> None:
    """The phrase is only said in the June video, so a May range must not find it."""
    with ml_returning(config, CANDLES) as ml:
        hits = search(seeded, ml, "happy birthday", visual_weight=0.0, filters=Filters(until="2026-05-31"))

    assert hits == []


def test_a_date_range_alone_browses_those_videos(seeded: Store, config: Config) -> None:
    with ml_returning(config, CANDLES) as ml:
        hits = search(seeded, ml, "", filters=Filters(since="2026-05-01", until="2026-05-31"))

    assert [hit.asset_id for hit in hits] == ["garden"]


def test_more_like_this_obeys_a_date_range(seeded: Store) -> None:
    ids = scene_ids(seeded)

    _reference, hits = similar(seeded, ids["a garden"], filters=Filters(since="2026-06-01"))

    assert {hit.asset_id for hit in hits} == {"birthday"}


def test_a_day_that_is_not_a_date_says_so() -> None:
    with pytest.raises(ConfigError, match="2019-07-04"):
        date_range("last summer", None)


def test_a_backwards_range_says_so() -> None:
    with pytest.raises(ConfigError, match="backwards"):
        date_range("2026-06-01", "2026-05-01")


def test_an_empty_range_is_no_filter_at_all() -> None:
    assert date_range(None, "  ") == (None, None)
    assert not Filters()


def test_the_candidate_cut_narrows_what_is_scored_not_how_it_scores(
    seeded: Store, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Averaging the survivors instead of the library drains the visual channel as it grows."""
    with ml_returning(config, CANDLES) as ml:
        scored, library_mean = search_module._visual_scores(seeded, ml, "candles", Filters())
        monkeypatch.setattr(search_module, "CANDIDATES", 2)
        cut, cut_mean = search_module._visual_scores(seeded, ml, "candles", Filters())

    assert len(scored) == 4
    assert len(cut) == 2
    assert cut_mean == pytest.approx(library_mean)
    assert all(cut[scene_id] == pytest.approx(scored[scene_id]) for scene_id in cut)


def in_albums(store: Store) -> Store:
    """The birthday is in Family 2026, the garden is in nothing."""
    store.replace_albums([("birthday", "al1", "Family 2026")])
    return store


def test_an_album_filter_keeps_only_the_videos_it_holds(seeded: Store, config: Config) -> None:
    with ml_returning(config, GARDEN) as ml:
        hits = search(
            in_albums(seeded), ml, "a garden", visual_weight=1.0, filters=Filters(albums=("Family 2026",))
        )

    assert {hit.asset_id for hit in hits} == {"birthday"}


def test_an_album_filter_reaches_the_speech_channel_too(seeded: Store, config: Config) -> None:
    """The FTS side joins through scenes, so a per-video filter has to reach it as well."""

    def refuse(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("a text-only search must not embed the query")

    store = in_albums(seeded)
    store.replace_transcript(
        "garden", [TranscriptRecord(1.0, 3.0, "happy birthday to you")], indexed_at=NOW, has_audio=True
    )
    with MLClient(config, "c", "f", transport=httpx.MockTransport(refuse)) as ml:
        hits = search(
            store,
            ml,
            "happy birthday",
            visual_weight=0.0,
            filters=Filters(albums=("Family 2026",)),
        )

    assert {hit.asset_id for hit in hits} == {"birthday"}


def test_two_albums_means_a_video_that_is_in_both(seeded: Store) -> None:
    store = in_albums(seeded)
    store.replace_albums([("birthday", "al1", "Family 2026"), ("garden", "al2", "Outdoors")])

    assert browse(store, Filters(albums=("Family 2026", "Outdoors"))) == []
    assert [hit.asset_id for hit in browse(store, Filters(albums=("Outdoors",)))] == ["garden"]


def test_an_album_matches_whatever_case_you_type(seeded: Store) -> None:
    store = in_albums(seeded)

    assert resolve_albums(store, ["  fAMILY 2026 "]) == ["Family 2026"]


def test_an_album_the_index_has_never_seen_is_an_error_not_an_empty_page(seeded: Store) -> None:
    """Immich albums that hold only photos are not in the index, and a typo looks the same."""
    store = in_albums(seeded)

    with pytest.raises(ConfigError) as caught:
        resolve_albums(store, ["Famly 2026"])

    assert "Famly 2026" in str(caught.value)
    assert "Indexed albums: Family 2026" in str(caught.value)


def test_a_person_and_an_album_are_resolved_the_same_way(store: Store) -> None:
    """Both go through one resolver, so an empty index says so instead of matching nothing."""
    with pytest.raises(ConfigError, match="Indexed people: none yet"):
        resolve_people(store, ["Anna"])
    with pytest.raises(ConfigError, match="Indexed albums: none yet"):
        resolve_albums(store, ["Family 2026"])


def ml_by_phrase(config: Config, vectors: dict[str, np.ndarray], fallback: np.ndarray) -> MLClient:
    """An ML server that answers each phrase differently, which the reference needs."""

    def handler(request: httpx.Request) -> httpx.Response:
        text = parse_qs(request.content.decode())["text"][0]
        vector = vectors.get(text, fallback)
        return httpx.Response(200, json={"clip": json.dumps([float(x) for x in vector])})

    return MLClient(config, "ViT-B-32__openai", "buffalo_l", transport=httpx.MockTransport(handler))


# Close to every scene without being any of them, which is what a hub scene does to a phrase
# the library has no answer for.
LUKEWARM = (CANDLES + CAKE + TABLE + GARDEN) / np.linalg.norm(CANDLES + CAKE + TABLE + GARDEN)


def test_the_reference_is_what_a_phrase_with_no_answer_still_scores(seeded: Store, config: Config) -> None:
    with ml_by_phrase(config, {}, LUKEWARM) as ml:
        floor = visual_reference(seeded, ml)

    assert floor == pytest.approx(float(max(LUKEWARM @ v for v in (CANDLES, CAKE, TABLE, GARDEN))))


def test_a_query_the_library_answers_beats_the_reference(seeded: Store, config: Config) -> None:
    with ml_by_phrase(config, {"blowing out candles": CANDLES}, LUKEWARM) as ml:
        floor = visual_reference(seeded, ml)
        hits = search(seeded, ml, "blowing out candles", visual_weight=1.0)

    assert hits[0].visual_score > floor


def test_a_query_it_does_not_answer_does_not(seeded: Store, config: Config) -> None:
    """The whole point: something always comes top, so the top has to be measured against
    what comes top for a phrase nobody filmed."""
    with ml_by_phrase(config, {}, LUKEWARM) as ml:
        floor = visual_reference(seeded, ml)
        hits = search(seeded, ml, "a helicopter", visual_weight=1.0)

    assert hits and hits[0].visual_score <= floor


def test_the_reference_is_measured_once_and_kept(seeded: Store, config: Config) -> None:
    calls = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"clip": json.dumps([float(x) for x in LUKEWARM])})

    ml = MLClient(config, "ViT-B-32__openai", "buffalo_l", transport=httpx.MockTransport(handler))
    with ml:
        assert visual_reference(seeded, ml) == visual_reference(seeded, ml)

    assert calls == len(REFERENCE_PHRASES)


def test_the_reference_is_measured_again_when_the_library_grows(seeded: Store, config: Config) -> None:
    """A new scene can be the one that sits near everything, which moves the floor."""
    with ml_by_phrase(config, {}, LUKEWARM) as ml:
        before = visual_reference(seeded, ml)
        seeded.replace_scenes(
            "garden",
            [
                SceneRecord(0, 0.0, 10.0, vector=GARDEN, label="a garden"),
                SceneRecord(1, 10.0, 20.0, vector=LUKEWARM, label="a hub"),
            ],
            VectorFile(config.vectors_path, DIM),
            indexed_at=NOW,
        )
        after = visual_reference(seeded, ml)

    assert after == pytest.approx(1.0)
    assert after > before


def test_an_empty_library_has_no_reference(store: Store, config: Config) -> None:
    with ml_by_phrase(config, {}, LUKEWARM) as ml:
        assert visual_reference(store, ml) is None


def test_a_locked_database_costs_the_cache_not_the_search(
    seeded: Store, config: Config, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`serve` is the first thing to write during a search, and `index` holds the write lock."""
    monkeypatch.setattr(
        seeded, "set_state", lambda *_a: (_ for _ in ()).throw(sqlite3.OperationalError("locked"))
    )
    with ml_by_phrase(config, {}, LUKEWARM) as ml:
        assert visual_reference(seeded, ml) == pytest.approx(
            float(max(LUKEWARM @ v for v in (CANDLES, CAKE, TABLE, GARDEN)))
        )
