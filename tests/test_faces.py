"""Face matching against Immich's named people."""

from __future__ import annotations

import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image

from immich_moments.config import Config
from immich_moments.faces import (
    MIN_THUMBNAIL_SIDE,
    PeopleIndex,
    faces_in_frame,
    load_people_index,
    pad_thumbnail,
    person_reference_embedding,
    refresh_people_refs,
)
from immich_moments.immich import ImmichClient
from immich_moments.ml import MLClient
from immich_moments.store import Store

from conftest import unit

NOW = "2026-09-16T10:00:00Z"
DIM = 8
ANNA = unit(1, DIM)
TOM = unit(2, DIM)
STRANGER = unit(3, DIM)


def face_payload(*faces: tuple[np.ndarray, float]) -> dict:
    return {
        "facial-recognition": [
            {
                "boundingBox": {"x1": 1, "y1": 2, "x2": 3, "y2": 4},
                "score": score,
                "embedding": json.dumps([float(x) for x in vector]),
            }
            for vector, score in faces
        ]
    }


def ml_returning(config: Config, payload: dict) -> MLClient:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    return MLClient(config, "ViT-B-32__openai", "buffalo_l", transport=transport)


def jpeg(size: tuple[int, int], colour: tuple[int, int, int] = (200, 120, 80)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def index_of(*people: tuple[str, str, np.ndarray]) -> PeopleIndex:
    return PeopleIndex(
        [(person_id, name) for person_id, name, _ in people],
        np.stack([vector for _, _, vector in people]).astype(np.float32),
    )


# ---- matching ------------------------------------------------------------


def test_the_nearest_named_person_wins() -> None:
    people = index_of(("p1", "Anna", ANNA), ("p2", "Tom", TOM))
    hit = people.match(ANNA, max_distance=0.5)
    assert hit is not None
    person_id, name, distance = hit
    assert (person_id, name) == ("p1", "Anna")
    assert distance == pytest.approx(0.0, abs=1e-5)


def test_a_stranger_is_left_unnamed() -> None:
    people = index_of(("p1", "Anna", ANNA))
    assert people.match(STRANGER, max_distance=0.1) is None


def test_an_empty_people_index_matches_nothing() -> None:
    empty = PeopleIndex([], np.zeros((0, DIM), dtype=np.float32))
    assert len(empty) == 0
    assert empty.match(ANNA, max_distance=1.0) is None


def test_low_scoring_detections_are_dropped(config: Config) -> None:
    ml = ml_returning(config, face_payload((ANNA, 0.95), (TOM, 0.20)))
    records = faces_in_frame(
        ml,
        jpeg((64, 64)),
        index_of(("p1", "Anna", ANNA), ("p2", "Tom", TOM)),
        min_score=0.7,
        max_distance=0.5,
    )
    assert [record.person_name for record in records] == ["Anna"]
    assert records[0].bbox == (1, 2, 3, 4)


def test_the_same_person_twice_in_one_frame_collapses_to_the_closer_face(config: Config) -> None:
    """A detector that fires twice on one head must not put the same name on a scene twice."""
    nearly = (ANNA * 0.9 + TOM * 0.1).astype(np.float32)
    nearly /= np.linalg.norm(nearly)
    ml = ml_returning(config, face_payload((nearly, 0.9), (ANNA, 0.9)))

    records = faces_in_frame(
        ml, jpeg((64, 64)), index_of(("p1", "Anna", ANNA)), min_score=0.7, max_distance=0.5
    )

    assert len(records) == 1
    assert records[0].distance == pytest.approx(0.0, abs=1e-5)


def test_unnamed_faces_are_kept_after_the_named_ones(config: Config) -> None:
    ml = ml_returning(config, face_payload((STRANGER, 0.9), (ANNA, 0.9)))
    records = faces_in_frame(
        ml, jpeg((64, 64)), index_of(("p1", "Anna", ANNA)), min_score=0.7, max_distance=0.1
    )
    assert [record.person_name for record in records] == ["Anna", None]


# ---- thumbnails ----------------------------------------------------------


def test_a_tight_thumbnail_is_padded_and_enlarged() -> None:
    padded = Image.open(io.BytesIO(pad_thumbnail(jpeg((60, 60)))))
    assert min(padded.size) > MIN_THUMBNAIL_SIDE


def test_a_thumbnail_that_is_not_an_image_passes_straight_through() -> None:
    assert pad_thumbnail(b"this is not a jpeg") == b"this is not a jpeg"


def test_a_thumbnail_with_no_face_has_no_reference_embedding(config: Config) -> None:
    ml = ml_returning(config, {"facial-recognition": []})
    assert person_reference_embedding(ml, jpeg((256, 256))) is None


def test_the_highest_scoring_face_becomes_the_reference(config: Config) -> None:
    ml = ml_returning(config, face_payload((TOM, 0.4), (ANNA, 0.9)))
    embedding = person_reference_embedding(ml, jpeg((256, 256)))
    assert embedding is not None
    assert float(embedding @ ANNA) == pytest.approx(1.0, abs=1e-5)


# ---- the refresh pass ----------------------------------------------------


class FakePeople:
    """The two Immich endpoints the refresh needs, with per-person behaviour."""

    def __init__(self, people: list[dict], thumbnails: dict[str, bytes | Exception | None]) -> None:
        self._people = people
        self._thumbnails = thumbnails

    def people(self):
        return iter(self._people)

    def person_thumbnail(self, person_id: str):
        value = self._thumbnails.get(person_id)
        if isinstance(value, Exception):
            raise value
        return value


def test_the_refresh_stores_one_reference_per_named_person(store: Store, config: Config) -> None:
    immich = FakePeople(
        [
            {"id": "p1", "name": "Anna"},
            {"id": "p2", "name": "  "},  # unnamed in Immich, so not a reference
            {"id": "p3", "name": "Tom"},
            {"id": "p4", "name": "Gone"},
            {"id": "p5", "name": "Faceless"},
        ],
        {
            "p1": jpeg((256, 256)),
            "p3": jpeg((256, 256)),
            "p4": httpx.ConnectError("boom"),  # thumbnail endpoint is down
            "p5": None,  # Immich has no thumbnail for them yet
        },
    )

    ml = ml_returning(config, face_payload((ANNA, 0.9)))
    matched, skipped = refresh_people_refs(store, immich, ml, now=NOW)

    assert (matched, skipped) == (2, 3)
    people = load_people_index(store)
    assert sorted(name for _, name in people.identities) == ["Anna", "Tom"]
    assert people.matrix.shape == (2, DIM)


def test_a_person_whose_thumbnail_has_no_face_is_skipped(store: Store, config: Config) -> None:
    immich = FakePeople([{"id": "p1", "name": "Anna"}], {"p1": jpeg((256, 256))})
    ml = ml_returning(config, {"facial-recognition": []})

    matched, skipped = refresh_people_refs(store, immich, ml, now=NOW)

    assert (matched, skipped) == (0, 1)
    assert len(load_people_index(store)) == 0


def test_a_face_record_survives_a_round_trip_through_the_store(store: Store, config: Config) -> None:
    ml = ml_returning(config, face_payload((ANNA, 0.88)))
    records = faces_in_frame(
        ml, jpeg((64, 64)), index_of(("p1", "Anna", ANNA)), min_score=0.7, max_distance=0.5
    )
    assert records[0].score == pytest.approx(0.88)
    assert records[0].person_id == "p1"


def test_a_dead_ml_container_is_reported_not_swallowed(config: Config) -> None:
    from immich_moments.errors import MLError

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    config.max_retries = 1
    ml = MLClient(config, "ViT-B-32__openai", "buffalo_l", transport=httpx.MockTransport(refuse))
    with pytest.raises(MLError, match="cannot reach"):
        faces_in_frame(ml, jpeg((64, 64)), index_of(("p1", "Anna", ANNA)), min_score=0.7, max_distance=0.5)


def test_one_persons_broken_thumbnail_does_not_end_the_refresh(store: Store, config: Config) -> None:
    """The warning path matters: a 500 on person 1 of 300 must not cost the other 299."""
    immich = FakePeople(
        [{"id": "p1", "name": "Anna"}, {"id": "p2", "name": "Tom"}],
        {"p1": RuntimeError("thumbnail 500"), "p2": jpeg((256, 256))},
    )
    ml = ml_returning(config, face_payload((TOM, 0.9)))

    matched, skipped = refresh_people_refs(store, immich, ml, now=NOW)

    assert (matched, skipped) == (1, 1)
    assert [name for _, name in load_people_index(store).identities] == ["Tom"]


def test_the_client_is_the_real_one_when_the_server_answers(store: Store, config: Config) -> None:
    """Ties the fake above back to the shipped ImmichClient so the shape cannot drift."""
    thumbnail = jpeg((256, 256))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/thumbnail"):
            return httpx.Response(200, content=thumbnail, headers={"content-type": "image/jpeg"})
        return httpx.Response(200, json={"people": [{"id": "p1", "name": "Anna"}], "hasNextPage": False})

    ml = ml_returning(config, face_payload((ANNA, 0.9)))
    with ImmichClient(config, transport=httpx.MockTransport(handler)) as immich:
        matched, skipped = refresh_people_refs(store, immich, ml, now=NOW)

    assert (matched, skipped) == (1, 0)
