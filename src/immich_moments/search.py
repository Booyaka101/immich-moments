"""Blended search: cosine similarity over scene vectors plus BM25 over the transcript."""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .errors import ConfigError
from .ml import MLClient
from .store import Store

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
CANDIDATES = 400
# Cosine margin over the library average that counts as a certain visual hit. CLIP is
# trained with a logit scale of 100, so 0.10 of cosine is ten logits, and that holds across
# the OpenCLIP variants Immich ships rather than being fitted to one of them.
VISUAL_DECISIVE_MARGIN = 0.10


@dataclass(frozen=True, slots=True)
class Filters:
    """Narrows the candidate scenes before either channel scores them."""

    asset_id: str | None = None
    people: tuple[str, ...] = ()
    since: str | None = None
    until: str | None = None

    def __bool__(self) -> bool:
        return bool(self.asset_id or self.people or self.since or self.until)

    def sql(self, asset_column: str, scene_column: str) -> tuple[str, list]:
        """Extra AND clauses for a query that can reach both columns, and their parameters."""
        clauses: list[str] = []
        params: list = []
        if self.asset_id:
            clauses.append(f"{asset_column} = ?")
            params.append(self.asset_id)
        # One clause each, so two names mean both people in the same scene rather than either.
        for name in self.people:
            clauses.append(
                f"{scene_column} IN (SELECT scene_id FROM scene_faces WHERE lower(person_name) = ?)"  # noqa: S608
            )
            params.append(name.strip().lower())
        # Immich stores the capture time, so compare the date part and leave the clock out of it.
        for bound, comparison in ((self.since, ">="), (self.until, "<=")):
            if bound:
                clauses.append(
                    f"{asset_column} IN (SELECT id FROM assets "  # noqa: S608
                    f"WHERE substr(file_created_at, 1, 10) {comparison} ?)"
                )
                params.append(bound)
        return "".join(f" AND {clause}" for clause in clauses), params


def resolve_people(store: Store, wanted: Iterable[str]) -> list[str]:
    """Names as the index spells them, so a typo is an error instead of an empty result.

    SQLite's lower() is ASCII only, which is the other reason not to send a typed name straight
    into the filter.
    """
    known = {row["name"].lower(): row["name"] for row in store.people_in_index()}
    cleaned = [name.strip() for name in wanted if name.strip()]
    missing = [name for name in cleaned if name.lower() not in known]
    if missing:
        have = ", ".join(sorted(known.values())) or "nobody yet"
        raise ConfigError(f"no indexed scenes name {', '.join(missing)}. Indexed people: {have}")
    return [known[name.lower()] for name in cleaned]


def date_range(since: str | None, until: str | None) -> tuple[str | None, str | None]:
    """The two calendar-day bounds, checked here so a typo is an error instead of no matches."""
    bounds = []
    for value, label in ((since, "since"), (until, "until")):
        if value is None or not value.strip():
            bounds.append(None)
            continue
        try:
            bounds.append(date.fromisoformat(value.strip()).isoformat())
        except ValueError as exc:
            raise ConfigError(f"{label} wants a date like 2019-07-04, not {value!r}.") from exc
    first, last = bounds
    if first and last and first > last:
        raise ConfigError(f"the range is backwards: since {first} is after until {last}.")
    return first, last


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
    file_created_at: str | None
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

    def as_dict(self, immich_url: str) -> dict:
        """One JSON shape for the web API and `search --json`, so they cannot drift apart."""
        return {
            "scene_id": self.scene_id,
            "asset_id": self.asset_id,
            "file_name": self.original_file_name,
            "file_created_at": self.file_created_at,
            "scene_index": self.scene_index,
            "start_seconds": round(self.start_seconds, 2),
            "end_seconds": round(self.end_seconds, 2),
            "timestamp": self.timestamp,
            "duration": format_timestamp(self.end_seconds - self.start_seconds),
            "label": self.label,
            "people": self.people,
            "transcript": self.transcript,
            "score": round(self.score, 4),
            "visual_score": round(self.visual_score, 4),
            "text_score": round(self.text_score, 4),
            "thumb": f"/thumbs/{self.thumb_path}" if self.thumb_path else None,
            "immich_url": self.immich_url(immich_url),
        }


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
    filters: Filters = Filters(),
) -> list[Hit]:
    query = query.strip()
    if not query:
        return browse(store, filters, limit=limit) if filters else []

    visual, library_mean = _visual_scores(store, ml, query, filters) if visual_weight > 0 else ({}, 0.0)
    textual = _text_scores(store, query, filters) if visual_weight < 1 else {}
    if not visual and not textual:
        return []

    visual_norm = _normalise_visual(visual, library_mean)
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


def browse(store: Store, filters: Filters, *, limit: int = 20) -> list[Hit]:
    """Every scene the filters allow, newest video first. No query, so nothing is scored."""
    where, params = filters.sql("s.asset_id", "s.id")
    rows = store.db.execute(
        f"""
        SELECT s.id FROM scenes s JOIN assets a ON a.id = s.asset_id
        WHERE 1=1{where}
        ORDER BY a.file_created_at DESC, s.asset_id, s.idx
        LIMIT ?
        """,  # noqa: S608
        (*params, limit),
    ).fetchall()
    return _hydrate(store, [(int(row["id"]), 0.0) for row in rows], {}, {})


def _candidates(store: Store, filters: Filters) -> tuple[np.ndarray, np.ndarray]:
    """Scene ids the filters allow, and the vectors that go with them, in id order."""
    if store.get_state("vector_dim") is None:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    vectors = store.vectors().read_all()
    if vectors.size == 0:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)

    where, params = filters.sql("asset_id", "id")
    rows = store.db.execute(
        f"SELECT id, vector_row FROM scenes WHERE vector_row IS NOT NULL{where} ORDER BY id",  # noqa: S608
        params,
    ).fetchall()
    usable = [(r["id"], r["vector_row"]) for r in rows if r["vector_row"] < vectors.shape[0]]
    if not usable:
        return np.empty(0, dtype=np.int64), np.empty((0, 0), dtype=np.float32)
    scene_ids = np.array([scene_id for scene_id, _ in usable], dtype=np.int64)
    return scene_ids, vectors[np.array([row for _, row in usable])]


def similar(
    store: Store, scene_id: int, *, limit: int = 20, filters: Filters = Filters()
) -> tuple[Hit, list[Hit]]:
    """The scene itself and the scenes whose CLIP vector is nearest to it.

    The score here is a plain cosine between two scene vectors, not the blended score a query
    produces: there is no query to calibrate it against.
    """
    row = store.db.execute("SELECT vector_row FROM scenes WHERE id = ?", (scene_id,)).fetchone()
    if row is None or row["vector_row"] is None:
        raise ConfigError(f"scene {scene_id} is not in the index, or was indexed without a vector.")
    vectors = store.vectors().read_all()
    if row["vector_row"] >= vectors.shape[0]:
        raise ConfigError(f"scene {scene_id} points past the end of the vector file.")
    reference = _hydrate(store, [(scene_id, 1.0)], {scene_id: 1.0}, {})[0]

    scene_ids, matrix = _candidates(store, filters)
    if scene_ids.size == 0:
        return reference, []
    scores = matrix @ vectors[row["vector_row"]]
    ranked = [
        (int(scene_ids[i]), float(scores[i]))
        for i in np.argsort(-scores, kind="stable")[: limit + 1]
        if int(scene_ids[i]) != scene_id
    ][:limit]
    return reference, _hydrate(store, ranked, dict(ranked), {})


def _visual_scores(
    store: Store, ml: MLClient, query: str, filters: Filters
) -> tuple[dict[int, float], float]:
    """The best `CANDIDATES` scenes for this query, and the mean cosine over all of them.

    The mean has to come from every scene the filters allow. Taking it from the survivors of
    the cut instead makes it climb with the library size, which quietly drains the visual
    channel: on a library of 10,000 scenes the top 400 average far above the whole, so a scene
    CLIP is certain about normalises as if it were ordinary.
    """
    scene_ids, matrix = _candidates(store, filters)
    if scene_ids.size == 0:
        return {}, 0.0
    scores = matrix @ ml.embed_text(query)
    top = np.argsort(-scores)[:CANDIDATES]
    return {int(scene_ids[i]): float(scores[i]) for i in top}, float(scores.mean())


def _text_scores(store: Store, query: str, filters: Filters) -> dict[int, TextMatch]:
    match = fts_query(query)
    if not match:
        return {}
    where, filter_params = filters.sql("s.asset_id", "s.scene_id")
    sql = f"""
        SELECT s.scene_id AS scene_id, s.start_seconds AS start_seconds, s.text AS text,
               -bm25(transcript_fts) AS score
        FROM transcript_fts
        JOIN transcript_segments s ON s.id = transcript_fts.rowid
        WHERE transcript_fts MATCH ? AND s.scene_id IS NOT NULL{where}
        ORDER BY score DESC LIMIT ?
    """  # noqa: S608
    try:
        rows = store.db.execute(sql, (match, *filter_params, CANDIDATES)).fetchall()
    except sqlite3.OperationalError:
        return {}

    best: dict[int, TextMatch] = {}
    for row in rows:
        scene_id = int(row["scene_id"])
        current = best.get(scene_id)
        if current is None or row["score"] > current.score:
            best[scene_id] = TextMatch(float(row["score"]), float(row["start_seconds"]), row["text"])
    return best


def _normalise_visual(scores: dict[int, float], library_mean: float) -> dict[int, float]:
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
    return {
        key: float(np.clip((value - library_mean) / VISUAL_DECISIVE_MARGIN, 0.0, 1.0))
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
            SELECT s.*, a.original_file_name, a.file_created_at
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
                file_created_at=row["file_created_at"],
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
