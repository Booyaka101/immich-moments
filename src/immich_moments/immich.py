"""Immich REST client. Everything here goes through the public API; nothing touches Postgres."""

from __future__ import annotations

import contextlib
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .config import Config
from .errors import AssetUnavailable, ImmichError

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
PAGE_SIZE = 250


@dataclass(frozen=True, slots=True)
class Annotations:
    """The description and tag paths Immich already holds for one asset."""

    description: str
    tags: frozenset[str] | set[str]


class ImmichClient:
    """Thin, retrying wrapper over the Immich API.

    `transport` exists so tests can replay recorded responses; in the shipped product it is
    always None and httpx talks to the real server.
    """

    def __init__(self, config: Config, transport: httpx.BaseTransport | None = None) -> None:
        config.require_credentials()
        self.config = config
        self._last_request = 0.0
        self.client = httpx.Client(
            base_url=config.api_base,
            headers={
                "x-api-key": config.immich_api_key,
                "Accept": "application/json",
                "User-Agent": "immich-moments",
            },
            timeout=httpx.Timeout(config.request_timeout, read=config.request_timeout),
            transport=transport,
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> ImmichClient:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    # ---- transport -------------------------------------------------------

    def _throttle(self) -> None:
        gap = self.config.min_request_interval
        if gap <= 0:
            return
        wait = gap - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Send one request, retrying 429/5xx and connection errors with backoff."""
        last_error: str = ""
        for attempt in range(self.config.max_retries):
            self._throttle()
            try:
                response = self.client.request(method, url, **kwargs)
                self._last_request = time.monotonic()
            except httpx.TimeoutException as exc:
                last_error = f"timed out after {self.config.request_timeout:g}s ({exc.__class__.__name__})"
            except httpx.TransportError as exc:
                last_error = f"cannot reach {self.config.immich_url}: {exc}"
            else:
                if response.status_code not in RETRY_STATUSES:
                    return self._checked(response, method, url)
                last_error = f"HTTP {response.status_code}"
                if attempt < self.config.max_retries - 1:
                    self._sleep_for(response, attempt)
                continue
            if attempt < self.config.max_retries - 1:
                self._sleep_for(None, attempt)
        raise ImmichError(
            f"{method} {url} failed after {self.config.max_retries} attempts: {last_error}", url=url
        )

    def _sleep_for(self, response: httpx.Response | None, attempt: int) -> None:
        delay = min(2.0**attempt, 30.0) + random.random() * 0.5  # noqa: S311
        if response is not None:
            header = response.headers.get("retry-after")
            if header:
                with contextlib.suppress(ValueError):
                    delay = max(delay, float(header))
        time.sleep(delay)

    def _checked(self, response: httpx.Response, method: str, url: str) -> httpx.Response:
        if response.is_success:
            return response
        detail = _detail(response)
        if response.status_code in (401, 403):
            raise ImmichError(
                f"Immich rejected the API key ({response.status_code}). Check IMMICH_API_KEY and that "
                "it has not been revoked.",
                status=response.status_code,
                url=url,
            )
        raise ImmichError(
            f"{method} {url} returned {response.status_code}: {detail}",
            status=response.status_code,
            url=url,
        )

    def get_json(self, url: str, **kwargs: Any) -> Any:
        return _json(self.request("GET", url, **kwargs))

    def post_json(self, url: str, payload: dict[str, Any]) -> Any:
        return _json(self.request("POST", url, json=payload))

    def put_json(self, url: str, payload: dict[str, Any]) -> Any:
        return _json(self.request("PUT", url, json=payload))

    # ---- server ----------------------------------------------------------

    def server_version(self) -> str:
        data = self.get_json("/server/version")
        if isinstance(data, dict) and "major" in data:
            return f"v{data['major']}.{data.get('minor', 0)}.{data.get('patch', 0)}"
        return "unknown"

    def system_config(self) -> dict[str, Any]:
        data = self.get_json("/system-config")
        if not isinstance(data, dict):
            raise ImmichError("GET /api/system-config did not return an object")
        return data

    def model_names(self) -> tuple[str, str]:
        """(clip model, facial recognition model) as the server has them configured."""
        machine_learning = self.system_config().get("machineLearning") or {}
        clip = (machine_learning.get("clip") or {}).get("modelName")
        face = (machine_learning.get("facialRecognition") or {}).get("modelName")
        if not clip or not face:
            raise ImmichError(
                "system-config has no machineLearning.clip.modelName / "
                "machineLearning.facialRecognition.modelName. Is the API key an admin key?"
            )
        return clip, face

    def face_thresholds(self) -> tuple[float, float]:
        facial = (self.system_config().get("machineLearning") or {}).get("facialRecognition") or {}
        return float(facial.get("minScore", 0.7)), float(facial.get("maxDistance", 0.5))

    # ---- assets ----------------------------------------------------------

    def iter_videos(self, *, updated_after: str | None = None) -> Iterator[dict[str, Any]]:
        """Page through every video asset, newest first.

        Immich deprecated the flat search fields in v3.2.0 in favour of a structured `filter`
        tree, but kept them working. The flat shape is sent first because it is the only one
        older servers understand; a 400 means the server has dropped it, so we switch to the
        cursor-paged structured shape and carry on.
        """
        structured = False
        page, cursor = 1, None
        while True:
            try:
                body = self.post_json(
                    "/search/metadata", _search_body(structured, page, cursor, updated_after)
                )
            except ImmichError as exc:
                if structured or exc.status != 400:
                    raise
                structured = True
                continue
            if not isinstance(body, dict):
                raise ImmichError("POST /api/search/metadata did not return an object")
            bucket = body.get("assets") or {}
            items = bucket.get("items") or []
            yield from items
            if not items:
                return
            if structured:
                cursor = bucket.get("nextCursor")
                if not cursor:
                    return
            else:
                if not bucket.get("nextPage"):
                    return
                page += 1

    def get_asset(self, asset_id: str) -> dict[str, Any]:
        """The asset record. Gone from Immich is `AssetUnavailable`, the same as for its original."""
        try:
            data = self.get_json(f"/assets/{asset_id}")
        except ImmichError as exc:
            if exc.status in (404, 410):
                raise AssetUnavailable(
                    f"asset {asset_id} is no longer in Immich (HTTP {exc.status})"
                ) from exc
            raise
        if not isinstance(data, dict):
            raise ImmichError(f"GET /api/assets/{asset_id} did not return an object")
        return data

    def asset_annotations(self, asset_id: str) -> Annotations:
        """What is already written on an asset, so write-back only plans what is missing.

        `PUT /api/assets/:id` takes `description` at the top level, but the GET only returns it
        under `exifInfo`, so reading the field you write gives you nothing. Trusting that empty
        string would overwrite whatever the owner had typed there.
        """
        asset = self.get_asset(asset_id)
        exif = asset.get("exifInfo")
        if isinstance(exif, dict) and exif.get("description"):
            description = str(exif["description"])
        else:
            description = str(asset.get("description") or "")
        tags = asset.get("tags")
        values = (
            {str(tag["value"]) for tag in tags if isinstance(tag, dict) and tag.get("value")}
            if isinstance(tags, list)
            else set()
        )
        return Annotations(description=description, tags=values)

    def download_original(self, asset_id: str, destination: Path) -> Path:
        """Stream the original file to disk. A 404 means the asset is gone, not that the run failed."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_suffix(destination.suffix + ".part")
        url = f"/assets/{asset_id}/original"
        self._throttle()
        try:
            with self.client.stream(
                "GET", url, timeout=httpx.Timeout(self.config.download_timeout)
            ) as response:
                self._last_request = time.monotonic()
                if response.status_code in (404, 410):
                    raise AssetUnavailable(
                        f"original for {asset_id} is not available (HTTP {response.status_code})"
                    )
                if not response.is_success:
                    response.read()
                    self._checked(response, "GET", url)
                with partial.open("wb") as handle:
                    for chunk in response.iter_bytes(1 << 20):
                        handle.write(chunk)
        except httpx.TimeoutException as exc:
            partial.unlink(missing_ok=True)
            raise ImmichError(f"download of {asset_id} timed out: {exc}") from exc
        except httpx.TransportError as exc:
            partial.unlink(missing_ok=True)
            raise ImmichError(f"download of {asset_id} failed: {exc}") from exc
        except OSError as exc:
            partial.unlink(missing_ok=True)
            raise ImmichError(f"cannot write {partial}: {exc}") from exc
        partial.replace(destination)
        return destination

    def update_asset(self, asset_id: str, **fields: Any) -> dict[str, Any]:
        return self.put_json(f"/assets/{asset_id}", fields)

    # ---- people ----------------------------------------------------------

    def people(self, *, with_hidden: bool = False) -> list[dict[str, Any]]:
        collected: list[dict[str, Any]] = []
        page = 1
        while True:
            body = self.get_json(
                "/people",
                params={"page": page, "size": PAGE_SIZE, "withHidden": str(with_hidden).lower()},
            )
            if not isinstance(body, dict):
                raise ImmichError("GET /api/people did not return an object")
            collected.extend(body.get("people") or [])
            if not body.get("hasNextPage"):
                return collected
            page += 1

    def asset_thumbnail(self, asset_id: str, size: str = "preview") -> bytes | None:
        """A real still from the library, used by `doctor` to round-trip the ML container."""
        return self._thumbnail(f"/assets/{asset_id}/thumbnail", params={"size": size})

    def person_thumbnail(self, person_id: str) -> bytes | None:
        """None when the person has no usable thumbnail, which is not an error."""
        return self._thumbnail(f"/people/{person_id}/thumbnail")

    def _thumbnail(self, url: str, **kwargs: Any) -> bytes | None:
        """Immich answers 404 until its own thumbnail job has run, which is a wait, not a failure."""
        try:
            response = self.request("GET", url, headers={"Accept": "image/*"}, **kwargs)
        except ImmichError as exc:
            if exc.status in (404, 410):
                return None
            raise
        return response.content or None

    # ---- tags ------------------------------------------------------------

    def upsert_tags(self, names: list[str]) -> dict[str, str]:
        """Create tags by full slash path if needed; returns {value: id} for every name asked for."""
        if not names:
            return {}
        created = self.put_json("/tags", {"tags": names})
        if not isinstance(created, list):
            raise ImmichError("PUT /api/tags did not return a list")
        return {tag["value"]: tag["id"] for tag in created if "value" in tag and "id" in tag}

    def bulk_tag_assets(self, tag_ids: list[str], asset_ids: list[str]) -> int:
        if not tag_ids or not asset_ids:
            return 0
        body = self.put_json("/tags/assets", {"tagIds": tag_ids, "assetIds": asset_ids})
        return int(body.get("count", 0)) if isinstance(body, dict) else 0


def _search_body(
    structured: bool, page: int, cursor: str | None, updated_after: str | None
) -> dict[str, Any]:
    body: dict[str, Any] = {"withExif": False, "size": PAGE_SIZE}
    if structured:
        filters: dict[str, Any] = {"type": {"eq": "VIDEO"}}
        if updated_after:
            filters["updatedAt"] = {"gt": updated_after}
        body["filter"] = filters
        if cursor:
            body["cursor"] = cursor
        return body
    body["type"] = "VIDEO"
    body["page"] = page
    if updated_after:
        body["updatedAfter"] = updated_after
    return body


def _json(response: httpx.Response) -> Any:
    if not response.content:
        return None
    try:
        return response.json()
    except ValueError as exc:
        raise ImmichError(
            f"{response.request.method} {response.request.url} returned non-JSON "
            f"({response.headers.get('content-type', 'unknown type')})"
        ) from exc


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return (response.text or "").strip()[:200] or "<empty body>"
    if isinstance(body, dict):
        message = body.get("message") or body.get("error")
        if isinstance(message, list):
            return "; ".join(str(part) for part in message)
        if message:
            return str(message)
    return str(body)[:200]
