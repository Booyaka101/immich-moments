"""Stream indexed videos without giving the browser an Immich credential."""

from __future__ import annotations

import re
from urllib.parse import quote

import anyio
import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask

FORWARD_HEADERS = ("content-type", "content-length", "content-range", "accept-ranges")


async def play_video(request: Request, asset_id: str) -> Response:
    state = request.app.state
    if not re.fullmatch(r"[A-Za-z0-9-]+", asset_id):
        raise HTTPException(status_code=404, detail="no such indexed video")
    async with state.searching:
        asset = state.store.asset(asset_id)
        if asset is None or asset["status"] == "unavailable":
            raise HTTPException(status_code=404, detail="no such indexed video")

    # Only this fixed endpoint is reachable. Redirects must not forward the API key elsewhere.
    headers = {"accept": "video/*", "accept-encoding": "identity"}
    for name in ("range", "if-range"):
        if name in request.headers:
            headers[name] = request.headers[name]
    upstream_request = state.playback.build_request(
        "GET", f"/assets/{quote(asset_id, safe='')}/video/playback", headers=headers
    )
    try:
        upstream = await state.playback.send(upstream_request, stream=True)
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="Immich video playback is unavailable") from exc

    downstream_headers = {
        "cache-control": "no-store",
        "x-content-type-options": "nosniff",
        "cross-origin-resource-policy": "same-origin",
    }
    if upstream.status_code == 416:
        if "content-range" in upstream.headers:
            downstream_headers["content-range"] = upstream.headers["content-range"]
        await upstream.aclose()
        return Response(status_code=416, headers=downstream_headers)
    if upstream.status_code not in (200, 206):
        status = 404 if upstream.status_code in (404, 410) else 502
        await upstream.aclose()
        raise HTTPException(status_code=status, detail="Immich could not play this video")

    content_type = upstream.headers.get("content-type", "application/octet-stream").split(";")[0]
    encoding = upstream.headers.get("content-encoding", "identity")
    if encoding != "identity" or not (
        content_type.startswith("video/") or content_type == "application/octet-stream"
    ):
        await upstream.aclose()
        raise HTTPException(status_code=502, detail="Immich did not return a video")
    downstream_headers.update(
        {name: upstream.headers[name] for name in FORWARD_HEADERS if name in upstream.headers}
    )

    async def chunks():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            # A seek cancels the previous request. Return its connection even on cancellation.
            with anyio.CancelScope(shield=True):
                await upstream.aclose()

    return StreamingResponse(
        chunks(),
        status_code=upstream.status_code,
        headers=downstream_headers,
        background=BackgroundTask(upstream.aclose),
    )
