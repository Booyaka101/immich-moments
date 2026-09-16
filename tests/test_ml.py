"""The ML client against responses recorded from a real immich-machine-learning container."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import numpy as np
import pytest

from immich_moments.config import Config
from immich_moments.errors import MLError
from immich_moments.ml import MLClient

RECORDED = json.loads((Path(__file__).parent / "fixtures" / "recorded.json").read_text("utf-8"))
FRAME = (Path(__file__).parent / "fixtures" / "frame.jpg").read_bytes()


def entries_of(request: httpx.Request) -> str:
    """The `entries` form field, whether httpx sent multipart (with an image) or urlencoded."""
    body = request.content.decode("utf-8", "replace")
    if request.headers.get("content-type", "").startswith("multipart/"):
        _, _, rest = body.partition('name="entries"')
        return rest.split("--", 1)[0].strip()
    return parse_qs(body)["entries"][0]


def replay(config: Config, seen: list[httpx.Request] | None = None) -> MLClient:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        if request.url.path == "/ping":
            return httpx.Response(200, text="pong")
        entries = entries_of(request)
        if '"visual"' in entries:
            return httpx.Response(200, json=RECORDED["clip_visual"])
        if '"textual"' in entries:
            return httpx.Response(200, json=RECORDED["clip_textual"])
        if "facial-recognition" in entries:
            return httpx.Response(200, json=RECORDED["faces"])
        return httpx.Response(400, text="unexpected entries")

    return MLClient(config, "ViT-B-32__openai", "buffalo_l", transport=httpx.MockTransport(handler))


def test_ping(config: Config) -> None:
    with replay(config) as ml:
        assert ml.ping() is True


def test_image_embedding_is_decoded_twice_and_normalised(config: Config) -> None:
    """The container serialises the vector with orjson before FastAPI serialises the response."""
    assert isinstance(RECORDED["clip_visual"]["clip"], str), "fixture must keep the double encoding"
    with replay(config) as ml:
        vector = ml.embed_image(FRAME)
    assert vector.shape == (512,)
    assert vector.dtype == np.float32
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(vector))


def test_text_and_image_land_in_the_same_space(config: Config) -> None:
    with replay(config) as ml:
        image = ml.embed_image(FRAME)
        text = ml.embed_text("two people at a table")
    assert image.shape == text.shape
    assert -1.0 <= float(image @ text) <= 1.0


def test_the_request_carries_entries_as_a_json_string(config: Config) -> None:
    seen: list[httpx.Request] = []
    with replay(config, seen) as ml:
        ml.embed_text("a photo")
    assert json.loads(entries_of(seen[-1])) == {"clip": {"textual": {"modelName": "ViT-B-32__openai"}}}
    assert parse_qs(seen[-1].content.decode())["text"] == ["a photo"]


def test_face_detection_sends_min_score(config: Config) -> None:
    """Without options.minScore the container raises a TypeError and answers HTTP 500."""
    seen: list[httpx.Request] = []
    config.face_min_score = 0.42
    with replay(config, seen) as ml:
        faces = ml.detect_faces(FRAME)
    detection = json.loads(entries_of(seen[-1]))["facial-recognition"]["detection"]
    assert detection["options"] == {"minScore": 0.42}
    assert len(faces) == 1
    face = faces[0]
    assert face.score > 0.7
    assert face.bbox == (233, 22, 407, 292)
    assert face.embedding.shape == (512,)
    assert pytest.approx(1.0, abs=1e-5) == float(np.linalg.norm(face.embedding))


def test_a_model_the_container_cannot_load_is_reported_clearly(config: Config) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"message": "Unknown model name ViT-L-14"})

    ml = MLClient(config, "ViT-L-14", "buffalo_l", transport=httpx.MockTransport(handler))
    with pytest.raises(MLError) as caught:
        ml.embed_text("anything")
    assert "ViT-L-14" in str(caught.value)


def test_a_dead_container_gives_up_after_the_configured_retries(config: Config) -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, text="no capacity")

    config.max_retries = 2
    ml = MLClient(config, "ViT-B-32__openai", "buffalo_l", transport=httpx.MockTransport(handler))
    with pytest.raises(MLError) as caught:
        ml.embed_text("anything")
    assert attempts == 2
    assert "2 attempts" in str(caught.value)
