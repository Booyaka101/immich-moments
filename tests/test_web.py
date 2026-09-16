"""The web UI over a seeded index, driven through FastAPI's TestClient."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from immich_moments.config import Config
from immich_moments.errors import DimensionMismatch
from immich_moments.store import FaceRecord, SceneRecord, Store, TranscriptRecord, VectorFile
from immich_moments.web.app import create_app

from conftest import unit

NOW = "2026-09-16T10:00:00Z"
DIM = 8
CANDLES = unit(11, DIM)
CAKE = unit(12, DIM)

SYSTEM_CONFIG = {
    "machineLearning": {
        "clip": {"modelName": "ViT-B-32__openai"},
        "facialRecognition": {"modelName": "buffalo_l", "minScore": 0.7, "maxDistance": 0.5},
    }
}


def immich_transport() -> httpx.MockTransport:
    return httpx.MockTransport(lambda _request: httpx.Response(200, json=SYSTEM_CONFIG))


def ml_transport(vector: np.ndarray) -> httpx.MockTransport:
    payload = {"clip": json.dumps([float(x) for x in vector])}
    return httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))


@pytest.fixture
def seeded(config: Config) -> Config:
    with Store(config) as store:
        store.check_model("ViT-B-32__openai", DIM, reindex=False)
        vectors = VectorFile(config.vectors_path, DIM)
        store.upsert_asset(
            "birthday",
            original_file_name="birthday.mp4",
            file_created_at="2026-06-01T00:00:00Z",
            updated_at="2026-06-01T00:00:00Z",
            duration_seconds=930.0,
        )
        store.replace_scenes(
            "birthday",
            [
                SceneRecord(0, 0.0, 462.0, vector=CAKE, label="a living room", thumb_path="birthday-0.jpg"),
                SceneRecord(
                    1,
                    462.0,
                    480.0,
                    vector=CANDLES,
                    label="blowing out candles",
                    thumb_path="birthday-1.jpg",
                    faces=[
                        FaceRecord("p1", "Anna", 0.18, 0.99, (1, 2, 3, 4)),
                        FaceRecord("p2", "Tom", 0.24, 0.97, (5, 6, 7, 8)),
                    ],
                ),
            ],
            vectors,
            indexed_at=NOW,
        )
        store.replace_transcript(
            "birthday",
            [
                TranscriptRecord(10.0, 12.0, "is the camera on"),
                TranscriptRecord(30.0, 32.0, "put it on the table"),
                TranscriptRecord(463.0, 466.0, "happy birthday to you"),
            ],
            indexed_at=NOW,
            has_audio=True,
        )
    (config.thumbs_dir / "birthday-1.jpg").write_bytes(b"\xff\xd8\xff\xdbnot really a jpeg\xff\xd9")
    return config


@pytest.fixture
def client(seeded: Config) -> Iterator[TestClient]:
    app = create_app(seeded, immich_transport=immich_transport(), ml_transport=ml_transport(CANDLES))
    with TestClient(app) as opened:
        yield opened


def test_the_page_renders_with_the_index_stats(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    body = response.text
    assert "immich-moments" in body
    assert "2 scenes" in body or ">2<" in body
    assert "/static/app.js" in body


def test_stats_reports_what_is_indexed(client: TestClient) -> None:
    stats = client.get("/api/stats").json()
    assert stats["assets"] == 1
    assert stats["scenes"] == 2
    assert stats["segments"] == 3
    assert stats["visual_done"] == 1
    assert stats["clip_model"] == "ViT-B-32__openai"
    assert stats["visual_weight"] == 0.65


def test_search_returns_a_serialised_hit(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "blowing out candles"}).json()

    assert body["query"] == "blowing out candles"
    assert body["weight"] == 0.65
    assert body["count"] == 2
    top = body["hits"][0]
    assert top["asset_id"] == "birthday"
    assert top["scene_index"] == 1
    assert top["timestamp"] == "07:42"
    assert top["duration"] == "00:18"
    assert top["label"] == "blowing out candles"
    assert top["people"] == ["Anna", "Tom"]
    assert top["transcript"] == "happy birthday to you"
    assert top["thumb"] == "/thumbs/birthday-1.jpg"
    assert top["immich_url"] == "http://immich.test/photos/birthday"


def test_the_weight_slider_reaches_the_ranking(client: TestClient) -> None:
    text_only = client.get("/api/search", params={"q": "happy birthday", "weight": 0.0}).json()
    assert text_only["weight"] == 0.0
    assert text_only["hits"][0]["visual_score"] == 0.0
    assert text_only["hits"][0]["text_score"] != 0.0


def test_search_can_be_pinned_to_one_asset(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "candles", "asset": "nothing-here"}).json()
    assert body["count"] == 0


def test_a_bad_query_is_a_422_not_a_traceback(client: TestClient) -> None:
    assert client.get("/api/search", params={"q": "x", "limit": 0}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "limit": 1000}).status_code == 422
    assert client.get("/api/search", params={"q": "x", "weight": 2}).status_code == 422


def test_an_empty_search_says_what_is_missing(client: TestClient) -> None:
    response = client.get("/api/search", params={"q": ""})
    assert response.status_code == 400
    assert "person" in response.json()["detail"]


def test_a_person_filter_narrows_the_hits(client: TestClient) -> None:
    body = client.get("/api/search", params={"q": "candles", "person": "Anna"}).json()
    assert body["people"] == ["Anna"]
    assert [hit["scene_index"] for hit in body["hits"]] == [1]


def test_a_person_the_index_does_not_know_is_a_400_not_an_empty_page(client: TestClient) -> None:
    """A hand-edited URL should say the name is unknown rather than look like no matches."""
    response = client.get("/api/search", params={"q": "candles", "person": "Nobody"})

    assert response.status_code == 400
    assert "Nobody" in response.json()["detail"]
    assert "Anna" in response.json()["detail"]


def test_a_person_with_no_query_browses_that_person(client: TestClient) -> None:
    body = client.get("/api/search", params={"person": "Tom"}).json()
    assert body["count"] == 1
    assert body["hits"][0]["scene_index"] == 1
    assert body["hits"][0]["file_created_at"] == "2026-06-01T00:00:00Z"


def test_more_like_this_ranks_against_one_scene(client: TestClient) -> None:
    candles = client.get("/api/search", params={"q": "candles"}).json()["hits"][0]

    body = client.get("/api/search", params={"like": candles["scene_id"], "limit": 5}).json()

    assert body["like"]["scene_id"] == candles["scene_id"]
    assert body["like"]["label"] == "blowing out candles"
    assert [hit["scene_index"] for hit in body["hits"]] == [0]
    assert body["hits"][0]["score"] == body["hits"][0]["visual_score"]


def test_a_query_and_a_scene_to_rank_against_is_a_400(client: TestClient) -> None:
    response = client.get("/api/search", params={"q": "candles", "like": 1})
    assert response.status_code == 400
    assert "no query" in response.json()["detail"]


def test_a_scene_id_that_is_not_indexed_is_a_400(client: TestClient) -> None:
    response = client.get("/api/search", params={"like": 9999})
    assert response.status_code == 400
    assert "9999" in response.json()["detail"]


def test_the_people_endpoint_lists_who_the_index_knows(client: TestClient) -> None:
    body = client.get("/api/people").json()
    assert body["people"] == [
        {"name": "Anna", "scenes": 1},
        {"name": "Tom", "scenes": 1},
    ]


def test_thumbnails_are_served_from_the_data_directory(client: TestClient) -> None:
    response = client.get("/thumbs/birthday-1.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content.startswith(b"\xff\xd8")


def test_a_missing_thumbnail_is_a_404(client: TestClient) -> None:
    assert client.get("/thumbs/nope.jpg").status_code == 404


@pytest.mark.parametrize("name", ["..%2f..%2fmoments.sqlite3", "..%5c..%5cmoments.sqlite3"])
def test_a_traversing_thumbnail_name_cannot_escape(client: TestClient, name: str) -> None:
    assert client.get(f"/thumbs/{name}").status_code == 404


def test_a_dead_ml_container_answers_503_with_a_message(seeded: Config) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    seeded.max_retries = 1
    app = create_app(seeded, immich_transport=immich_transport(), ml_transport=httpx.MockTransport(refuse))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/search", params={"q": "candles"})

    assert response.status_code == 503
    assert "cannot reach" in response.json()["error"]


def test_serving_without_credentials_fails_before_the_port_is_opened(tmp_path: Path) -> None:
    from immich_moments.errors import ConfigError

    with pytest.raises(ConfigError, match="IMMICH_URL"):
        create_app(Config(data_dir=tmp_path))


def test_a_search_does_not_run_on_the_event_loop(client: TestClient, monkeypatch) -> None:
    """It is blocking HTTP then SQLite: on the loop, one query freezes the page and the thumbnails."""
    import immich_moments.web.app as web

    on_loop: list[bool] = []
    real = web.run_search

    def record(*args, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            on_loop.append(False)
        else:
            on_loop.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(web, "run_search", record)
    assert client.get("/api/search", params={"q": "candles"}).status_code == 200

    assert on_loop == [False]


def test_an_index_built_with_another_model_refuses_to_serve(seeded: Config) -> None:
    with Store(seeded) as store:
        store.set_state("clip_model", "ViT-L-14__openai")

    app = create_app(seeded, immich_transport=immich_transport(), ml_transport=ml_transport(CANDLES))
    with pytest.raises(DimensionMismatch, match="--reindex"), TestClient(app):
        pass  # pragma: no cover - startup raises
