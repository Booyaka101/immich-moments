"""SQLite index, the flat vector file, and the refusal to mix vector spaces."""

from __future__ import annotations

import numpy as np
import pytest

from immich_moments.config import Config
from immich_moments.errors import DimensionMismatch, MomentsError
from immich_moments.store import FaceRecord, SceneRecord, Store, TranscriptRecord, VectorFile

from conftest import unit

NOW = "2026-09-16T10:00:00Z"


def seed_asset(store: Store, asset_id: str = "a1") -> None:
    store.upsert_asset(
        asset_id,
        original_file_name=f"{asset_id}.mp4",
        file_created_at="2026-06-01T00:00:00Z",
        updated_at="2026-06-01T00:00:00Z",
        duration_seconds=30.0,
    )


def test_vector_file_round_trips_rows(config: Config) -> None:
    vectors = VectorFile(config.vectors_path, 8)
    assert vectors.rows == 0
    rows = [vectors.append(unit(seed)) for seed in range(3)]
    assert rows == [0, 1, 2]
    matrix = vectors.read_all()
    assert matrix.shape == (3, 8)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix[1], unit(1), rtol=1e-6)


def test_vector_file_rejects_the_wrong_shape(config: Config) -> None:
    vectors = VectorFile(config.vectors_path, 8)
    with pytest.raises(DimensionMismatch, match="expected a 8-dim vector"):
        vectors.append(np.zeros(4, dtype=np.float32))


def test_a_truncated_vector_file_is_reported_not_guessed(config: Config) -> None:
    vectors = VectorFile(config.vectors_path, 8)
    vectors.append(unit(0))
    config.vectors_path.write_bytes(config.vectors_path.read_bytes()[:-3])
    with pytest.raises(MomentsError, match="--reindex"):
        _ = vectors.rows


def test_a_model_change_refuses_to_mix_spaces(store: Store) -> None:
    store.check_model("ViT-B-32__openai", 512, reindex=False)
    store.check_model("ViT-B-32__openai", 512, reindex=False)  # unchanged, still fine

    with pytest.raises(DimensionMismatch) as caught:
        store.check_model("ViT-L-14__openai", 768, reindex=False)
    message = str(caught.value)
    assert "ViT-B-32__openai" in message and "ViT-L-14__openai" in message
    assert "--reindex" in message
    assert store.get_state("clip_model") == "ViT-B-32__openai"


def test_searching_against_another_model_is_refused(store: Store) -> None:
    """`search` and `serve` embed the query with whatever the server says today."""
    store.check_model("ViT-B-32__openai", 512, reindex=False)
    store.assert_model("ViT-B-32__openai")

    with pytest.raises(DimensionMismatch, match="--reindex"):
        store.assert_model("ViT-L-14__openai")


def test_searching_an_empty_index_has_no_model_to_disagree_with(store: Store) -> None:
    store.assert_model("ViT-B-32__openai")


def test_vectors_appended_without_a_commit_are_reclaimed(store: Store, config: Config) -> None:
    """A run killed between the append and its transaction leaves rows nothing points at."""
    seed_asset(store)
    store.check_model("ViT-B-32__openai", 8, reindex=False)
    store.replace_scenes(
        "a1",
        [SceneRecord(0, 0.0, 10.0, vector=unit(1))],
        VectorFile(config.vectors_path, 8),
        indexed_at=NOW,
    )
    orphan = VectorFile(config.vectors_path, 8)
    orphan.append(unit(2))
    orphan.append(unit(3))
    assert orphan.rows == 3
    store.close()

    with Store(config) as reopened:
        assert reopened.vectors().rows == 1
        np.testing.assert_allclose(reopened.vectors().read_all()[0], unit(1), rtol=1e-6)


def test_a_torn_write_at_the_tail_is_reclaimed_too(store: Store, config: Config) -> None:
    store.check_model("ViT-B-32__openai", 8, reindex=False)
    config.vectors_path.write_bytes(bytes(13))
    store.close()

    with Store(config) as reopened:
        assert reopened.vectors().rows == 0


def test_reindex_clears_the_old_space(store: Store, config: Config) -> None:
    seed_asset(store)
    vectors = VectorFile(config.vectors_path, 8)
    store.check_model("ViT-B-32__openai", 8, reindex=False)
    store.replace_scenes(
        "a1",
        [SceneRecord(0, 0.0, 10.0, vector=unit(1), label="a cake")],
        vectors,
        indexed_at=NOW,
    )
    store.replace_transcript(
        "a1", [TranscriptRecord(1.0, 2.0, "happy birthday")], indexed_at=NOW, has_audio=True
    )
    assert store.counts()["scenes"] == 1

    store.check_model("ViT-L-14__openai", 768, reindex=True)

    counts = store.counts()
    assert counts["scenes"] == 0
    assert counts["segments"] == 0
    assert counts["assets"] == 1  # the asset list survives, only its derived data goes
    assert counts["visual_done"] == 0
    assert not config.vectors_path.exists()
    assert store.get_state("vector_dim") == "768"
    assert [row["id"] for row in store.assets_needing("visual")] == ["a1"]


def test_replacing_scenes_swaps_faces_too(store: Store, config: Config) -> None:
    seed_asset(store)
    vectors = VectorFile(config.vectors_path, 8)
    first = SceneRecord(
        0,
        0.0,
        5.0,
        vector=unit(1),
        faces=[FaceRecord("p1", "Anna", 0.2, 0.99, (1, 2, 3, 4))],
    )
    ids = store.replace_scenes("a1", [first], vectors, indexed_at=NOW)
    assert store.faces_for_scenes(ids)[ids[0]][0]["person_name"] == "Anna"

    second = SceneRecord(
        0, 0.0, 5.0, vector=unit(2), faces=[FaceRecord("p2", "Tom", 0.1, 0.98, (0, 0, 1, 1))]
    )
    ids = store.replace_scenes("a1", [second], vectors, indexed_at=NOW)
    faces = store.faces_for_scenes(ids)
    assert [face["person_name"] for face in faces[ids[0]]] == ["Tom"]
    assert store.db.execute("SELECT COUNT(*) AS n FROM scene_faces").fetchone()["n"] == 1


def test_the_people_in_the_index_are_counted_by_scene(store: Store, config: Config) -> None:
    """The UI offers these names, so an unmatched face must not become a nameless entry."""
    seed_asset(store)
    vectors = VectorFile(config.vectors_path, 8)
    store.replace_scenes(
        "a1",
        [
            SceneRecord(
                0,
                0.0,
                5.0,
                vector=unit(1),
                faces=[
                    FaceRecord("p1", "Anna", 0.2, 0.99, (1, 2, 3, 4)),
                    FaceRecord(None, None, None, 0.9, (5, 6, 7, 8)),
                ],
            ),
            SceneRecord(
                1, 5.0, 9.0, vector=unit(2), faces=[FaceRecord("p1", "Anna", 0.2, 0.9, (1, 2, 3, 4))]
            ),
        ],
        vectors,
        indexed_at=NOW,
    )

    assert [dict(row) for row in store.people_in_index()] == [{"name": "Anna", "scenes": 2}]


def test_transcript_segments_land_in_the_scene_they_fall_in(store: Store, config: Config) -> None:
    seed_asset(store)
    vectors = VectorFile(config.vectors_path, 8)
    scene_ids = store.replace_scenes(
        "a1",
        [SceneRecord(0, 0.0, 6.0, vector=unit(1)), SceneRecord(1, 6.0, 12.0, vector=unit(2))],
        vectors,
        indexed_at=NOW,
    )
    store.replace_transcript(
        "a1",
        [TranscriptRecord(1.0, 3.0, "first scene"), TranscriptRecord(7.0, 9.0, "second scene")],
        indexed_at=NOW,
        has_audio=True,
    )
    rows = store.transcript_for("a1")
    assert [row["scene_id"] for row in rows] == scene_ids
    assert store.asset("a1")["has_audio"] == 1


def test_a_segment_past_the_last_scene_still_lands_somewhere(store: Store, config: Config) -> None:
    """Whisper timestamps can run a fraction past the container duration."""
    seed_asset(store)
    vectors = VectorFile(config.vectors_path, 8)
    scene_ids = store.replace_scenes(
        "a1", [SceneRecord(0, 0.0, 6.0, vector=unit(1))], vectors, indexed_at=NOW
    )
    store.replace_transcript("a1", [TranscriptRecord(6.4, 7.0, "tail")], indexed_at=NOW, has_audio=True)
    assert store.transcript_for("a1")[0]["scene_id"] == scene_ids[0]


def test_fts_stays_in_step_with_the_segments(store: Store, config: Config) -> None:
    seed_asset(store)
    vectors = VectorFile(config.vectors_path, 8)
    store.replace_scenes("a1", [SceneRecord(0, 0.0, 6.0, vector=unit(1))], vectors, indexed_at=NOW)
    store.replace_transcript(
        "a1", [TranscriptRecord(0.0, 2.0, "happy birthday")], indexed_at=NOW, has_audio=True
    )
    matched = store.db.execute(
        "SELECT COUNT(*) AS n FROM transcript_fts WHERE transcript_fts MATCH ?", ("birthday",)
    ).fetchone()
    assert matched["n"] == 1

    store.replace_transcript("a1", [TranscriptRecord(0.0, 2.0, "goodbye")], indexed_at=NOW, has_audio=True)
    matched = store.db.execute(
        "SELECT COUNT(*) AS n FROM transcript_fts WHERE transcript_fts MATCH ?", ("birthday",)
    ).fetchone()
    assert matched["n"] == 0


def test_people_refs_drop_a_stale_dimension(store: Store) -> None:
    store.put_person_ref("p1", "Anna", unit(1, dim=512), NOW)
    store.put_person_ref("p2", "Tom", unit(2, dim=512), NOW)
    store.put_person_ref("p3", "Old", unit(3, dim=8), NOW)
    identities, matrix = store.people_refs()
    assert [name for _, name in identities] == ["Anna", "Tom"]
    assert matrix.shape == (2, 512)


def test_people_refs_are_empty_before_anything_is_indexed(store: Store) -> None:
    identities, matrix = store.people_refs()
    assert identities == []
    assert matrix.shape == (0, 0)


def test_vectors_without_an_index_says_what_to_run(store: Store) -> None:
    with pytest.raises(MomentsError, match="run `immich-moments index` first"):
        store.vectors()


def test_unavailable_assets_are_not_retried_forever(store: Store) -> None:
    seed_asset(store)
    seed_asset(store, "a2")
    store.set_asset_status("a2", "unavailable", "HTTP 404")
    assert [row["id"] for row in store.assets_needing("visual")] == ["a1"]
    assert store.counts()["unavailable"] == 1


def test_a_failed_transaction_leaves_nothing_behind(store: Store, config: Config) -> None:
    seed_asset(store)
    vectors = VectorFile(config.vectors_path, 8)
    store.replace_scenes("a1", [SceneRecord(0, 0.0, 6.0, vector=unit(1))], vectors, indexed_at=NOW)
    with pytest.raises(RuntimeError), store.transaction() as db:
        db.execute("DELETE FROM scenes")
        raise RuntimeError("something blew up mid-write")
    assert store.counts()["scenes"] == 1


def test_relabelling_rewrites_the_label_and_leaves_the_vector_alone(store: Store, config: Config) -> None:
    """`relabel` reads the vectors back, so a re-label must not disturb the rows they live in."""
    seed_asset(store)
    vectors = VectorFile(config.vectors_path, 8)
    store.replace_scenes(
        "a1",
        [
            SceneRecord(0, 0.0, 5.0, vector=unit(1), label="a garden", label_score=0.3),
            SceneRecord(1, 5.0, 9.0, vector=unit(2)),
            SceneRecord(2, 9.0, 12.0),
        ],
        vectors,
        indexed_at=NOW,
    )

    listed = store.labelled_scenes()
    assert [(row["vector_row"], row["label"]) for row in listed] == [(0, "a garden"), (1, None)]

    store.set_labels([(listed[0]["id"], None, None), (listed[1]["id"], "a birthday cake", 0.42)])

    after = store.labelled_scenes()
    assert [(row["vector_row"], row["label"]) for row in after] == [(0, None), (1, "a birthday cake")]
    assert [dict(row)["label_score"] for row in store.scenes_for("a1")][:2] == [None, 0.42]
    np.testing.assert_allclose(VectorFile(config.vectors_path, 8).read_all()[1], unit(2), rtol=1e-6)


def test_relabelling_nothing_is_not_a_write(store: Store) -> None:
    seed_asset(store)
    store.set_labels([])
    assert store.labelled_scenes() == []
