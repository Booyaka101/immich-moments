"""Byte ranges, safe upstream failures, and connection cleanup for the inline player."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi.testclient import TestClient
from test_web import CANDLES, immich_transport, ml_transport

from immich_moments.config import Config
from immich_moments.store import Store
from immich_moments.web.app import create_app


@pytest.fixture
def seeded(config: Config) -> Config:
    with Store(config) as store:
        store.upsert_asset(
            "birthday", original_file_name="birthday.mp4", file_created_at=None, updated_at=None
        )
    return config


class VideoStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes = b"video bytes") -> None:
        self.content = content
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield self.content

    async def aclose(self) -> None:
        self.closed = True


def app_for(config: Config, handler):
    return create_app(
        config,
        immich_transport=immich_transport(),
        ml_transport=ml_transport(CANDLES),
        playback_transport=httpx.MockTransport(handler),
    )


@pytest.mark.parametrize("partial", [False, True])
def test_video_is_streamed_with_ranges_and_only_safe_headers(seeded: Config, partial: bool) -> None:
    stream = VideoStream()
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        headers = {
            "content-type": "video/mp4",
            "content-length": "11",
            "accept-ranges": "bytes",
            "set-cookie": "secret=upstream",
            "x-api-key": "must-not-leak",
        }
        if partial:
            headers["content-range"] = "bytes 0-10/100"
        return httpx.Response(206 if partial else 200, headers=headers, stream=stream)

    with TestClient(app_for(seeded, respond)) as client:
        headers = {"Range": "bytes=0-10", "If-Range": '"version"'} if partial else {}
        response = client.get("/api/videos/birthday", headers=headers)

    assert response.status_code == (206 if partial else 200)
    assert response.content == b"video bytes"
    assert response.headers["content-type"] == "video/mp4"
    assert response.headers["accept-ranges"] == "bytes"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "set-cookie" not in response.headers
    assert "x-api-key" not in response.headers
    assert stream.closed
    assert len(requests) == 1
    assert requests[0].url.path == "/api/assets/birthday/video/playback"
    assert requests[0].headers["x-api-key"] == seeded.immich_api_key
    assert requests[0].headers["accept-encoding"] == "identity"
    if partial:
        assert requests[0].headers["range"] == "bytes=0-10"
        assert requests[0].headers["if-range"] == '"version"'
        assert response.headers["content-range"] == "bytes 0-10/100"
    else:
        assert "range" not in requests[0].headers


def test_an_unknown_asset_never_reaches_immich(seeded: Config) -> None:
    def unexpected(_request):
        pytest.fail("unindexed assets must not be proxied")

    with TestClient(app_for(seeded, unexpected)) as client:
        assert client.get("/api/videos/unknown").status_code == 404
        assert client.get("/api/videos/..%5csecret").status_code == 404


@pytest.mark.parametrize(
    "upstream_status,expected", [(401, 502), (403, 502), (404, 404), (410, 404), (500, 502), (302, 502)]
)
def test_upstream_errors_are_closed_and_do_not_leak_details(
    seeded: Config, upstream_status: int, expected: int
) -> None:
    stream = VideoStream(b"private upstream error with credentials")
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            upstream_status, headers={"location": "https://other.test/video"}, stream=stream
        )

    with TestClient(app_for(seeded, respond)) as client:
        response = client.get("/api/videos/birthday")
    assert response.status_code == expected
    assert "private" not in response.text
    assert "location" not in response.headers
    assert stream.closed
    assert len(requests) == 1


def test_an_unsatisfiable_range_preserves_the_size(seeded: Config) -> None:
    stream = VideoStream()
    with TestClient(
        app_for(seeded, lambda _: httpx.Response(416, headers={"content-range": "bytes */11"}, stream=stream))
    ) as client:
        response = client.get("/api/videos/birthday", headers={"range": "bytes=99-"})
    assert response.status_code == 416
    assert response.headers["content-range"] == "bytes */11"
    assert response.content == b""
    assert stream.closed


def test_html_is_not_served_as_video(seeded: Config) -> None:
    stream = VideoStream(b"<script>bad()</script>")
    with TestClient(
        app_for(seeded, lambda _: httpx.Response(200, headers={"content-type": "text/html"}, stream=stream))
    ) as client:
        response = client.get("/api/videos/birthday")
    assert response.status_code == 502
    assert stream.closed


def test_a_connection_failure_is_a_safe_gateway_error(seeded: Config) -> None:
    def refuse(request):
        raise httpx.ConnectError("private transport detail", request=request)

    with TestClient(app_for(seeded, refuse)) as client:
        response = client.get("/api/videos/birthday")
    assert response.status_code == 502
    assert "private" not in response.text


def test_a_stream_failure_still_returns_the_upstream_connection(seeded: Config) -> None:
    class BrokenStream(VideoStream):
        async def __aiter__(self):
            yield b"first chunk"
            raise httpx.ReadError("stream interrupted")

    stream = BrokenStream()
    with TestClient(
        app_for(seeded, lambda _: httpx.Response(200, headers={"content-type": "video/mp4"}, stream=stream)),
        raise_server_exceptions=False,
    ) as client:
        client.get("/api/videos/birthday")
    assert stream.closed


def test_an_unavailable_indexed_asset_is_not_proxied(seeded: Config) -> None:
    with Store(seeded) as store:
        store.set_asset_status("birthday", "unavailable")

    def unexpected(_request):
        pytest.fail("unavailable assets must not be proxied")

    with TestClient(app_for(seeded, unexpected)) as client:
        assert client.get("/api/videos/birthday").status_code == 404
