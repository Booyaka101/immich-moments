"""Immich client against recorded responses, with no network anywhere."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from immich_moments.config import Config
from immich_moments.errors import AssetUnavailable, ImmichError
from immich_moments.immich import ImmichClient

RECORDED = json.loads((Path(__file__).parent / "fixtures" / "recorded.json").read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def no_backoff_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Backoff is exercised for its arithmetic, not for its wall-clock cost."""
    slept: list[float] = []
    monkeypatch.setattr("immich_moments.immich.time.sleep", slept.append)
    return slept


def client(config: Config, handler) -> ImmichClient:
    return ImmichClient(config, transport=httpx.MockTransport(handler))


def json_response(payload, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload)


def test_server_version_and_models(config: Config) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/server/version"):
            return json_response(RECORDED["server_version"])
        return json_response(RECORDED["system_config"])

    with client(config, handler) as immich:
        assert immich.server_version() == "v3.2.2"
        assert immich.model_names() == ("ViT-B-32__openai", "buffalo_l")
        assert immich.face_thresholds() == (0.7, 0.5)


def test_model_names_without_admin_rights(config: Config) -> None:
    with client(config, lambda _r: json_response({"machineLearning": {}})) as immich:
        with pytest.raises(ImmichError, match="admin key"):
            immich.model_names()


def test_iter_videos_pages_the_flat_shape(config: Config) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if body["page"] == 1:
            page = dict(RECORDED["search_page"])
            page["assets"] = dict(page["assets"], nextPage=2)
            return json_response(page)
        return json_response({"assets": {"items": [], "nextPage": None}})

    with client(config, handler) as immich:
        videos = list(immich.iter_videos())

    assert [v["originalFileName"] for v in videos] == ["umbra-short.mp4", "rotated_phone_clip.mp4"]
    assert [b["page"] for b in seen] == [1, 2]
    assert all(b["type"] == "VIDEO" for b in seen)


def test_iter_videos_falls_back_to_the_structured_filter(config: Config) -> None:
    """A v3.2+ server that has dropped the flat fields answers 400; we switch and use the cursor."""
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append(body)
        if "filter" not in body:
            return json_response({"message": ["property type should not exist"]}, status=400)
        if "cursor" not in body:
            page = dict(RECORDED["search_page"])
            page["assets"] = dict(page["assets"], nextCursor="cur-2")
            return json_response(page)
        return json_response({"assets": {"items": [], "nextCursor": None}})

    with client(config, handler) as immich:
        videos = list(immich.iter_videos(updated_after="2026-01-01T00:00:00Z"))

    assert len(videos) == 2
    assert seen[0]["type"] == "VIDEO"
    assert seen[1]["filter"] == {"type": {"eq": "VIDEO"}, "updatedAt": {"gt": "2026-01-01T00:00:00Z"}}
    assert seen[2]["cursor"] == "cur-2"


def test_rate_limit_is_retried_and_honours_retry_after(config: Config, no_backoff_sleep: list[float]) -> None:
    attempts = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, json={"message": "slow down"})
        if attempts == 2:
            return json_response({"message": "boom"}, status=503)
        return json_response(RECORDED["server_version"])

    config.max_retries = 5
    with client(config, handler) as immich:
        assert immich.server_version() == "v3.2.2"

    assert attempts == 3
    assert no_backoff_sleep[0] >= 7.0


def test_giving_up_names_the_last_failure(config: Config) -> None:
    with client(config, lambda _r: json_response({"message": "nope"}, status=503)) as immich:
        with pytest.raises(ImmichError, match=r"after 2 attempts: HTTP 503"):
            immich.server_version()


def test_a_rejected_key_says_so_in_english(config: Config) -> None:
    with client(config, lambda _r: json_response({"message": "Invalid API key"}, status=401)) as immich:
        with pytest.raises(ImmichError, match="IMMICH_API_KEY") as caught:
            immich.server_version()
    assert caught.value.status == 401


def test_other_errors_carry_the_server_detail(config: Config) -> None:
    with client(config, lambda _r: json_response({"message": ["bad", "worse"]}, status=422)) as immich:
        with pytest.raises(ImmichError, match="returned 422: bad; worse"):
            immich.server_version()


def test_download_original_streams_to_disk(config: Config, tmp_path: Path) -> None:
    payload = b"\x00\x01binary video bytes" * 100

    with client(config, lambda _r: httpx.Response(200, content=payload)) as immich:
        destination = immich.download_original("asset-1", tmp_path / "out" / "video.mp4")

    assert destination.read_bytes() == payload
    assert not list(destination.parent.glob("*.part"))


def test_a_missing_original_is_not_a_failed_run(config: Config, tmp_path: Path) -> None:
    with client(config, lambda _r: httpx.Response(404, json={"message": "Not found"})) as immich:
        with pytest.raises(AssetUnavailable, match="asset-gone"):
            immich.download_original("asset-gone", tmp_path / "video.mp4")
    assert not list(tmp_path.glob("video.mp4*"))


def test_people_pages_until_the_server_says_stop(config: Config) -> None:
    pages = [
        dict(RECORDED["people_page"], hasNextPage=True),
        {"people": [{"id": "c3", "name": "Walter"}], "hasNextPage": False},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return json_response(pages[int(request.url.params["page"]) - 1])

    with client(config, handler) as immich:
        assert [p["name"] for p in immich.people()] == ["Martin", "Elena", "Walter"]


def test_thumbnails_return_none_when_empty(config: Config) -> None:
    with client(config, lambda _r: httpx.Response(200, content=b"")) as immich:
        assert immich.person_thumbnail("p1") is None
        assert immich.asset_thumbnail("a1") is None


def test_tag_upsert_and_bulk_assign(config: Config) -> None:
    sent: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        sent.append((request.url.path, body))
        if request.url.path.endswith("/tags/assets"):
            return json_response({"count": len(body["assetIds"])})
        return json_response([{"id": f"tag-{i}", "value": value} for i, value in enumerate(body["tags"])])

    with client(config, handler) as immich:
        ids = immich.upsert_tags(["moments/people/Anna", "moments/scene/birthday cake"])
        assert ids == {"moments/people/Anna": "tag-0", "moments/scene/birthday cake": "tag-1"}
        assert immich.bulk_tag_assets(["tag-0"], ["a1", "a2"]) == 2
        assert immich.upsert_tags([]) == {}

    assert sent[-1][1] == {"tagIds": ["tag-0"], "assetIds": ["a1", "a2"]}


def test_no_request_is_made_for_an_empty_tag_assignment(config: Config) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:  # pragma: no cover - must not run
        raise AssertionError("an empty assignment must not hit the server")

    with client(config, handler) as immich:
        assert immich.bulk_tag_assets([], ["a1"]) == 0
        assert immich.bulk_tag_assets(["t1"], []) == 0


def test_a_non_json_body_is_reported_as_such(config: Config) -> None:
    response = httpx.Response(200, text="<html>proxy error</html>", headers={"content-type": "text/html"})
    with client(config, lambda _r: response) as immich:
        with pytest.raises(ImmichError, match="non-JSON"):
            immich.server_version()


def test_the_description_is_read_from_where_immich_actually_puts_it(config: Config) -> None:
    """PUT takes `description` at the top level; GET only ever returns it under `exifInfo`."""
    body = {"id": "a1", "exifInfo": {"description": "Grandma's camcorder tape"}}
    with client(config, lambda _r: httpx.Response(200, json=body)) as immich:
        assert immich.asset_annotations("a1").description == "Grandma's camcorder tape"


def test_an_asset_with_no_description_reads_as_empty(config: Config) -> None:
    body = {"id": "a1", "exifInfo": {"description": None, "make": "Canon"}}
    with client(config, lambda _r: httpx.Response(200, json=body)) as immich:
        assert immich.asset_annotations("a1").description == ""


def test_a_top_level_description_is_still_honoured(config: Config) -> None:
    """Cheap insurance in case a release starts returning it where the PUT expects it."""
    body = {"id": "a1", "description": "from the top", "exifInfo": {}}
    with client(config, lambda _r: httpx.Response(200, json=body)) as immich:
        assert immich.asset_annotations("a1").description == "from the top"


def test_the_tags_an_asset_already_carries_are_read_back(config: Config) -> None:
    """Write-back plans against these, so a second run has nothing left to add."""
    body = {
        "id": "a1",
        "exifInfo": {},
        "tags": [{"id": "t1", "value": "moments/people/Anna"}, {"id": "t2", "name": "orphan"}],
    }
    with client(config, lambda _r: httpx.Response(200, json=body)) as immich:
        assert immich.asset_annotations("a1").tags == {"moments/people/Anna"}
