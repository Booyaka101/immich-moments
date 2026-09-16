"""Blended search: cosine similarity over scene vectors plus BM25 over the transcript."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .ml import MLClient
from .store import Store

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
CANDIDATES = 400
# Cosine margin over the library average that counts as a certain visual hit. CLIP is
# trained with a logit scale of 100, so 0.10 of cosine is ten logits, and that holds across
# the OpenCLIP variants Immich ships rather than being fitted to one of them.
VISUAL_DECISIVE_MARGIN = 0.10


@dataclass(slots=True)
class TextMatch:
    """The best-scoring transcript segment in one scene, and where it was said."""

    score: float
    start_seconds: float
    text: str


@dataclass(slots=True)
class Hit:
    scene_id: int
    asset_id: str
    original_file_name: str
    scene_index: int
    start_seconds: float
    end_seconds: float
    label: str | None
    label_score: float | None
    thumb_path: str | None
    visual_score: float
    text_score: float
    score: float
    people: list[str] = field(default_factory=list)
    transcript: str = ""
    spoken_at_seconds: float | None = None

    @property
    def moment_seconds(self) -> float:
        """Where to send someone: the words if the words matched, otherwise the cut."""
        return self.start_seconds if self.spoken_at_seconds is None else self.spoken_at_seconds

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.moment_seconds)

    def immich_url(self, base: str) -> str:
        return f"{base.rstrip('/')}/photos/{self.asset_id}"


def format_timestamp(seconds: float) -> str:
    total = int(max(seconds, 0.0))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:02d}:{secs:02d}"


def fts_query(text: str) -> str:
    """FTS5 has its own query syntax; quote every token so user punctuation cannot break it."""
    tokens = TOKEN.findall(text.lower())
    return " OR ".join(f'"{token}"' for token in tokens)


def search(
    store: Store,
    ml: MLClient,
    query: str,
    *,
    limit: int = 20,
    visual_weight: float = 0.65,
    asset_id: str | None = None,
) -> list[Hit]:
    query = query.strip()
    if not query:
        return []

    visual = _visual_scores(store, ml, query, asset_id) if visual_weight > 0 else {}
    textual = _text_scores(store, query, asset_id) if visual_weight < 1 else {}
    if not visual and not textual:
        return []

    visual_norm = _normalise_visual(visual)
    text_norm = _normalise_text({key: match.score for key, match in textual.items()})
    blended = {
        scene_id: visual_weight * visual_norm.get(scene_id, 0.0)
        + (1 - visual_weight) * text_norm.get(scene_id, 0.0)
        for scene_id in set(visual_norm) | set(text_norm)
    }
    # Both channels saturate, so ties are normal; the stronger raw cosine breaks them rather
    # than whichever scene happens to have been indexed first.
    ranked = sorted(blended.items(), key=lambda item: (-item[1], -visual.get(item[0], 0.0), item[0]))[:limit]
    return _hydrate(store, ranked, visual, textual)


def _visual_scores(store: Store, ml: MLClient, query: str, asset_id: str | None) -> dict[int, float]:
    dim_state = store.get_state("vector_dim")
    if dim_state is None:
        return {}
    vectors = store.vectors().read_all()
    if vectors.size == 0:
        return {}

    sql = "SELECT id, vector_row FROM scenes WHERE vector_row IS NOT NULL"
    params: tuple = ()
    if asset_id:
        sql += " AND asset_id = ?"
        params = (asset_id,)
    rows = store.db.execute(sql, params).fetchall()
    usable = [(r["id"], r["vector_row"]) for r in rows if r["vector_row"] < vectors.shape[0]]
    if not usable:
        return {}

    embedding = ml.embed_text(query)
    scene_ids = np.array([scene_id for scene_id, _ in usable])
    scores = vectors[np.array([row for _, row in usable])] @ embedding
    top = np.argsort(-scores)[:CANDIDATES]
    return {int(scene_ids[i]): float(scores[i]) for i in top}


def _text_scores(store: Store, query: str, asset_id: str | None) -> dict[int, TextMatch]:
    match = fts_query(query)
    if not match:
        return {}
    sql = """
        SELECT s.scene_id AS scene_id, s.start_seconds AS start_seconds, s.text AS text,
               -bm25(transcript_fts) AS score
        FROM transcript_fts
        JOIN transcript_segments s ON s.id = transcript_fts.rowid
        WHERE transcript_fts MATCH ? AND s.scene_id IS NOT NULL
    """
    params: list = [match]
    if asset_id:
        sql += " AND s.asset_id = ?"
        params.append(asset_id)
    sql += " ORDER BY score DESC LIMIT ?"
    params.append(CANDIDATES)
    try:
        rows = store.db.execute(sql, params).fetchall()
    except sqlite3.OperationalError:
        return {}

    best: dict[int, TextMatch] = {}
    for row in rows:
        scene_id = int(row["scene_id"])
        current = best.get(scene_id)
        if current is None or row["score"] > current.score:
            best[scene_id] = TextMatch(float(row["score"]), float(row["start_seconds"]), row["text"])
    return best


def _normalise_visual(scores: dict[int, float]) -> dict[int, float]:
    """Cosines onto [0, 1] by their margin over the library average for this query.

    The margin has to stay in cosine units. Dividing by the spread instead, which is what
    standard scores do, divides out the signal: a query with a real match in the library has a
    wide spread and one with none has a narrow one, so both ended up with a top scene near 1.0
    and an exact transcript match lost to a scene CLIP had no opinion about. Subtracting the
    mean is still worth doing, because it cancels the per-query offset that makes raw CLIP
    cosines incomparable between queries.
    """
    if not scores:
        return {}
    values = np.fromiter(scores.values(), dtype=np.float64, count=len(scores))
    mean = float(values.mean())
    return {
        key: float(np.clip((value - mean) / VISUAL_DECISIVE_MARGIN, 0.0, 1.0))
        for key, value in scores.items()
    }


def _normalise_text(scores: dict[int, float]) -> dict[int, float]:
    """BM25 onto [0, 1] against the best match, which is the only comparable thing about it.

    Only scenes whose transcript matched are in here at all, so the weakest of them still
    matched and must not normalise to zero the way min-max would leave it.
    """
    if not scores:
        return {}
    best = max(scores.values())
    if best <= 0:
        return dict.fromkeys(scores, 1.0)
    return {key: max(value, 0.0) / best for key, value in scores.items()}


def _hydrate(
    store: Store,
    ranked: list[tuple[int, float]],
    visual: dict[int, float],
    textual: dict[int, TextMatch],
) -> list[Hit]:
    if not ranked:
        return []
    scene_ids = [scene_id for scene_id, _ in ranked]
    placeholders = ",".join("?" * len(scene_ids))
    rows = {
        row["id"]: row
        for row in store.db.execute(
            f"""
            SELECT s.*, a.original_file_name
            FROM scenes s JOIN assets a ON a.id = s.asset_id
            WHERE s.id IN ({placeholders})
            """,  # noqa: S608
            tuple(scene_ids),
        ).fetchall()
    }
    faces = store.faces_for_scenes(scene_ids)
    snippets = _snippets(store, scene_ids, textual)

    hits: list[Hit] = []
    for scene_id, score in ranked:
        row = rows.get(scene_id)
        if row is None:
            continue
        hits.append(
            Hit(
                scene_id=scene_id,
                asset_id=row["asset_id"],
                original_file_name=row["original_file_name"],
                scene_index=row["idx"],
                start_seconds=row["start_seconds"],
                end_seconds=row["end_seconds"],
                label=row["label"],
                label_score=row["label_score"],
                thumb_path=row["thumb_path"],
                visual_score=visual.get(scene_id, 0.0),
                text_score=textual[scene_id].score if scene_id in textual else 0.0,
                score=score,
                people=unique_names([face["person_name"] for face in faces.get(scene_id, [])]),
                transcript=snippets.get(scene_id, ""),
                spoken_at_seconds=textual[scene_id].start_seconds if scene_id in textual else None,
            )
        )
    return hits


def _snippets(store: Store, scene_ids: list[int], textual: dict[int, TextMatch]) -> dict[int, str]:
    """The scene's speech, starting at the segment that matched so the quote leads with it."""
    placeholders = ",".join("?" * len(scene_ids))
    rows = store.db.execute(
        f"SELECT scene_id, start_seconds, text FROM transcript_segments "  # noqa: S608
        f"WHERE scene_id IN ({placeholders}) ORDER BY start_seconds",
        tuple(scene_ids),
    ).fetchall()
    joined: dict[int, list[str]] = {}
    for row in rows:
        match = textual.get(row["scene_id"])
        if match is not None and row["start_seconds"] < match.start_seconds:
            continue
        joined.setdefault(row["scene_id"], []).append(row["text"])
    return {scene_id: " ".join(parts).strip() for scene_id, parts in joined.items()}


def unique_names(names: Iterable[str | None]) -> list[str]:
    """Names in the order they were detected, once each, with the unnamed faces dropped."""
    seen: list[str] = []
    for name in names:
        if name and name not in seen:
            seen.append(name)
    return seen
