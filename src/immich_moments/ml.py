"""Client for Immich's machine-learning container (`POST /predict`).

The contract, read off `machine-learning/immich_ml/main.py` on the Immich main branch: a
multipart body whose `entries` part is a JSON *string* describing the pipeline, plus either
an `image` file part or a `text` form field. Vectors come back double-encoded -- the model
code serialises them with `orjson.dumps(...).decode()` before FastAPI serialises the response
-- so `{"clip": "[0.1, 0.2]"}` is a string inside JSON, not an array.

The facial-recognition detection entry must carry `options.minScore`; without it the container
raises `FaceDetector._predict() missing 1 required positional argument` and answers HTTP 500.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np

from .config import Config
from .errors import MLError

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


@dataclass(slots=True)
class DetectedFace:
    bbox: tuple[int, int, int, int]
    score: float
    embedding: np.ndarray


class MLClient:
    """`transport` is for replaying recorded responses in tests; production passes None."""

    def __init__(
        self,
        config: Config,
        clip_model: str,
        face_model: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.config = config
        self.clip_model = clip_model
        self.face_model = face_model
        self.client = httpx.Client(
            base_url=config.ml_url.rstrip("/"),
            timeout=httpx.Timeout(config.request_timeout),
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> MLClient:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---- transport -------------------------------------------------------

    def _predict(
        self, entries: dict[str, Any], *, image: bytes | None = None, text: str | None = None
    ) -> dict[str, Any]:
        data = {"entries": json.dumps(entries)}
        files = None
        if image is not None:
            files = {"image": ("frame.jpg", image, "image/jpeg")}
        elif text is not None:
            data["text"] = text
        else:  # pragma: no cover - guarded by the callers
            raise MLError("predict needs either an image or text")

        last_error = ""
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post("/predict", data=data, files=files)
            except httpx.TimeoutException:
                last_error = f"timed out after {self.config.request_timeout:g}s"
            except httpx.TransportError as exc:
                last_error = f"cannot reach {self.config.ml_url}: {exc}"
            else:
                if response.status_code in RETRY_STATUSES:
                    last_error = _failure(response, entries)
                elif response.is_success:
                    return _body(response)
                else:
                    raise MLError(_failure(response, entries))
            time.sleep(min(2.0**attempt, 20.0) + random.random() * 0.5)  # noqa: S311
        raise MLError(
            f"POST {self.config.ml_url}/predict failed after {self.config.max_retries} attempts: {last_error}"
        )

    # ---- pipelines -------------------------------------------------------

    def embed_image(self, jpeg: bytes) -> np.ndarray:
        body = self._predict({"clip": {"visual": {"modelName": self.clip_model}}}, image=jpeg)
        return _unit(_vector(body, "clip"))

    def embed_text(self, text: str) -> np.ndarray:
        body = self._predict({"clip": {"textual": {"modelName": self.clip_model}}}, text=text)
        return _unit(_vector(body, "clip"))

    def detect_faces(self, jpeg: bytes, *, min_score: float | None = None) -> list[DetectedFace]:
        """`minScore` is not optional: the container's FaceDetector takes it as a plain argument."""
        threshold = self.config.face_min_score if min_score is None else min_score
        body = self._predict(
            {
                "facial-recognition": {
                    "detection": {"modelName": self.face_model, "options": {"minScore": threshold}},
                    "recognition": {"modelName": self.face_model},
                }
            },
            image=jpeg,
        )
        raw = body.get("facial-recognition")
        if isinstance(raw, str):
            raw = json.loads(raw)
        if raw is None:
            raise MLError("the ML container answered without a 'facial-recognition' key")
        faces: list[DetectedFace] = []
        for item in raw:
            box = item.get("boundingBox") or {}
            embedding = item.get("embedding")
            if isinstance(embedding, str):
                embedding = json.loads(embedding)
            faces.append(
                DetectedFace(
                    bbox=(
                        int(box.get("x1", 0)),
                        int(box.get("y1", 0)),
                        int(box.get("x2", 0)),
                        int(box.get("y2", 0)),
                    ),
                    score=float(item.get("score", 0.0)),
                    embedding=_unit(np.asarray(embedding, dtype=np.float32)),
                )
            )
        return faces

    def ping(self) -> bool:
        try:
            response = self.client.get("/ping")
        except httpx.TransportError as exc:
            raise MLError(
                f"cannot reach the ML container at {self.config.ml_url}: {exc}. "
                "Is immich-machine-learning running and its port published?"
            ) from exc
        return response.is_success and response.text.strip() == "pong"


def _body(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError as exc:
        raise MLError("the ML container returned a non-JSON body") from exc
    if not isinstance(body, dict):
        raise MLError(f"the ML container returned {type(body).__name__}, expected an object")
    return body


def _vector(body: dict[str, Any], key: str) -> np.ndarray:
    raw = body.get(key)
    if raw is None:
        raise MLError(f"the ML container answered without a {key!r} key: {sorted(body)}")
    if isinstance(raw, str):
        raw = json.loads(raw)
    vector = np.asarray(raw, dtype=np.float32).ravel()
    if vector.size == 0:
        raise MLError(f"the ML container returned an empty {key} vector")
    return vector


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else (vector / norm).astype(np.float32)


def _failure(response: httpx.Response, entries: dict[str, Any]) -> str:
    detail = (response.text or "").strip()[:300]
    models = ", ".join(entry["modelName"] for task in entries.values() for entry in task.values())
    if response.status_code == 500 and "Failed to load model" in detail:
        return (
            f"the ML container could not load {models}. It downloads weights on first use, so "
            f"check its logs and that it has network access. Server said: {detail}"
        )
    return f"POST /predict returned {response.status_code} for model(s) {models}: {detail}"
