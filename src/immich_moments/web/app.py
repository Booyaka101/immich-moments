"""FastAPI app behind `immich-moments serve`: one page, one search box, scene thumbnails."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool

from ..config import Config
from ..errors import ConfigError, MomentsError
from ..immich import ImmichClient
from ..ml import MLClient
from ..search import Filters, date_range, resolve_albums, resolve_people
from ..search import search as run_search
from ..search import similar as run_similar
from ..store import Store

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
MAX_LIMIT = 100


def create_app(
    config: Config,
    *,
    immich_transport: httpx.BaseTransport | None = None,
    ml_transport: httpx.BaseTransport | None = None,
) -> FastAPI:
    """The transports are the same test seam the clients have; `serve` passes neither."""
    config.require_credentials()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        immich = ImmichClient(config, transport=immich_transport)
        clip_model, face_model = immich.model_names()
        app.state.config = config
        app.state.store = Store(config)
        app.state.immich = immich
        app.state.ml = MLClient(config, clip_model, face_model, transport=ml_transport)
        app.state.clip_model = clip_model
        app.state.store.assert_model(clip_model)
        # One search at a time: it is the only thing that touches the store off the event loop.
        app.state.searching = asyncio.Lock()
        try:
            yield
        finally:
            app.state.ml.close()
            immich.close()
            app.state.store.close()

    app = FastAPI(title="immich-moments", version=_version(), lifespan=lifespan)
    templates = Jinja2Templates(directory=str(HERE / "templates"))
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.exception_handler(MomentsError)
    async def _moments_error(_request: Request, exc: MomentsError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=503)

    @app.get("/")
    async def home(request: Request):
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"stats": await _stats(request.app)},
        )

    @app.get("/api/stats")
    async def stats(request: Request):
        return await _stats(request.app)

    @app.get("/api/people")
    async def people(request: Request):
        async with request.app.state.searching:
            rows = await run_in_threadpool(request.app.state.store.people_in_index)
        return {"people": [{"name": row["name"], "scenes": row["scenes"]} for row in rows]}

    @app.get("/api/albums")
    async def albums(request: Request):
        async with request.app.state.searching:
            rows = await run_in_threadpool(request.app.state.store.albums_in_index)
        return {"albums": [{"name": row["name"], "videos": row["videos"]} for row in rows]}

    @app.get("/api/search")
    async def search(
        request: Request,
        q: str = Query("", description="What you are looking for."),
        limit: int = Query(24, ge=1, le=MAX_LIMIT),
        weight: float | None = Query(None, ge=0.0, le=1.0),
        asset: str | None = Query(None),
        person: list[str] = Query([], description="Only scenes this person appears in."),
        album: list[str] = Query([], description="Only videos in this Immich album."),
        like: int | None = Query(None, description="Scene id to find more of, instead of a query."),
        since: str | None = Query(None, description="Only videos taken on or after this day."),
        until: str | None = Query(None, description="Only videos taken on or before this day."),
    ):
        state = request.app.state
        try:
            first, last = date_range(since, until)
        except ConfigError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if like is not None and q.strip():
            raise HTTPException(status_code=400, detail="like ranks against one scene, so it takes no query")
        if like is None and not any((q.strip(), person, album, asset, first, last)):
            raise HTTPException(status_code=400, detail="give me a query, or a person to browse")
        # A search is an HTTP call to the ML container and then SQLite, both blocking. On the event
        # loop it would freeze the page and every thumbnail behind one query.
        reference = None
        async with state.searching:
            try:
                people = await run_in_threadpool(resolve_people, state.store, person)
                in_albums = await run_in_threadpool(resolve_albums, state.store, album)
                filters = Filters(
                    asset_id=asset,
                    people=tuple(people),
                    albums=tuple(in_albums),
                    since=first,
                    until=last,
                )
                if like is not None:
                    reference, hits = await run_in_threadpool(
                        run_similar, state.store, like, limit=limit, filters=filters
                    )
                else:
                    hits = await run_in_threadpool(
                        run_search,
                        state.store,
                        state.ml,
                        q,
                        limit=limit,
                        visual_weight=config.visual_weight if weight is None else weight,
                        filters=filters,
                    )
            except ConfigError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "query": q,
            "people": people,
            "albums": in_albums,
            "since": first,
            "until": last,
            "like": reference.as_dict(config.browser_url) if reference else None,
            "weight": config.visual_weight if weight is None else weight,
            "count": len(hits),
            "hits": [hit.as_dict(config.browser_url) for hit in hits],
        }

    @app.get("/thumbs/{name}")
    async def thumb(request: Request, name: str):
        path = (request.app.state.config.thumbs_dir / name).resolve()
        if path.parent != config.thumbs_dir.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="no such thumbnail")
        return FileResponse(path, media_type="image/jpeg")

    return app


async def _stats(app: FastAPI) -> dict:
    async with app.state.searching:
        counts = await run_in_threadpool(app.state.store.counts)
    return {
        "assets": counts["assets"],
        "scenes": counts["scenes"],
        "segments": counts["segments"],
        "people": counts["people"],
        "visual_done": counts["visual_done"],
        "audio_done": counts["audio_done"],
        "clip_model": app.state.clip_model,
        "visual_weight": app.state.config.visual_weight,
    }


def _version() -> str:
    from .. import __version__

    return __version__
