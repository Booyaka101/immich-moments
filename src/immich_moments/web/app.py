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
from ..errors import MomentsError
from ..immich import ImmichClient
from ..ml import MLClient
from ..search import Filters
from ..search import search as run_search
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

    @app.get("/api/search")
    async def search(
        request: Request,
        q: str = Query("", description="What you are looking for."),
        limit: int = Query(24, ge=1, le=MAX_LIMIT),
        weight: float | None = Query(None, ge=0.0, le=1.0),
        asset: str | None = Query(None),
        person: list[str] = Query([], description="Only scenes this person appears in."),
    ):
        state = request.app.state
        filters = Filters(asset_id=asset, people=tuple(name for name in person if name.strip()))
        if not q.strip() and not filters:
            raise HTTPException(status_code=400, detail="give me a query, or a person to browse")
        # A search is an HTTP call to the ML container and then SQLite, both blocking. On the event
        # loop it would freeze the page and every thumbnail behind one query.
        async with state.searching:
            hits = await run_in_threadpool(
                run_search,
                state.store,
                state.ml,
                q,
                limit=limit,
                visual_weight=config.visual_weight if weight is None else weight,
                filters=filters,
            )
        return {
            "query": q,
            "people": list(filters.people),
            "weight": config.visual_weight if weight is None else weight,
            "count": len(hits),
            "hits": [hit.as_dict(config.immich_url) for hit in hits],
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
        counts = app.state.store.counts()
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
