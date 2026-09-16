"""FastAPI app behind `immich-moments serve`: one page, one search box, scene thumbnails."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..config import Config
from ..errors import MomentsError
from ..immich import ImmichClient
from ..ml import MLClient
from ..search import Hit, format_timestamp
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
            context={"stats": _stats(request.app)},
        )

    @app.get("/api/stats")
    async def stats(request: Request):
        return _stats(request.app)

    @app.get("/api/search")
    async def search(
        request: Request,
        q: str = Query(..., min_length=1, description="What you are looking for."),
        limit: int = Query(24, ge=1, le=MAX_LIMIT),
        weight: float | None = Query(None, ge=0.0, le=1.0),
        asset: str | None = Query(None),
    ):
        state = request.app.state
        hits = run_search(
            state.store,
            state.ml,
            q,
            limit=limit,
            visual_weight=config.visual_weight if weight is None else weight,
            asset_id=asset,
        )
        return {
            "query": q,
            "weight": config.visual_weight if weight is None else weight,
            "count": len(hits),
            "hits": [_serialise(hit, config) for hit in hits],
        }

    @app.get("/thumbs/{name}")
    async def thumb(request: Request, name: str):
        path = (request.app.state.config.thumbs_dir / name).resolve()
        if path.parent != config.thumbs_dir.resolve() or not path.is_file():
            raise HTTPException(status_code=404, detail="no such thumbnail")
        return FileResponse(path, media_type="image/jpeg")

    return app


def _serialise(hit: Hit, config: Config) -> dict:
    return {
        "scene_id": hit.scene_id,
        "asset_id": hit.asset_id,
        "file_name": hit.original_file_name,
        "scene_index": hit.scene_index,
        "start_seconds": round(hit.start_seconds, 2),
        "end_seconds": round(hit.end_seconds, 2),
        "timestamp": hit.timestamp,
        "duration": format_timestamp(hit.end_seconds - hit.start_seconds),
        "label": hit.label,
        "people": hit.people,
        "transcript": hit.transcript,
        "score": round(hit.score, 4),
        "visual_score": round(hit.visual_score, 4),
        "text_score": round(hit.text_score, 4),
        "thumb": f"/thumbs/{hit.thumb_path}" if hit.thumb_path else None,
        "immich_url": hit.immich_url(config.immich_url),
    }


def _stats(app: FastAPI) -> dict:
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
