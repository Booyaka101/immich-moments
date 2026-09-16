"""SQLite index plus a flat float32 vector file living beside it."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import Config
from .errors import DimensionMismatch, MomentsError

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
    id                 TEXT PRIMARY KEY,
    original_file_name TEXT NOT NULL DEFAULT '',
    file_created_at    TEXT,
    updated_at         TEXT,
    duration_seconds   REAL,
    has_audio          INTEGER,
    visual_indexed_at  TEXT,
    audio_indexed_at   TEXT,
    status             TEXT NOT NULL DEFAULT 'pending',
    error              TEXT
);

CREATE TABLE IF NOT EXISTS scenes (
    id             INTEGER PRIMARY KEY,
    asset_id       TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    idx            INTEGER NOT NULL,
    start_seconds  REAL NOT NULL,
    end_seconds    REAL NOT NULL,
    vector_row     INTEGER,
    label          TEXT,
    label_score    REAL,
    thumb_path     TEXT,
    UNIQUE (asset_id, idx)
);
CREATE INDEX IF NOT EXISTS scenes_asset ON scenes(asset_id);

CREATE TABLE IF NOT EXISTS scene_faces (
    id          INTEGER PRIMARY KEY,
    scene_id    INTEGER NOT NULL REFERENCES scenes(id) ON DELETE CASCADE,
    person_id   TEXT,
    person_name TEXT,
    distance    REAL,
    score       REAL,
    bbox        TEXT
);
CREATE INDEX IF NOT EXISTS scene_faces_scene ON scene_faces(scene_id);

CREATE TABLE IF NOT EXISTS people_refs (
    person_id  TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    vector     BLOB NOT NULL,
    dim        INTEGER NOT NULL,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS transcript_segments (
    id            INTEGER PRIMARY KEY,
    asset_id      TEXT NOT NULL REFERENCES assets(id) ON DELETE CASCADE,
    scene_id      INTEGER REFERENCES scenes(id) ON DELETE SET NULL,
    start_seconds REAL NOT NULL,
    end_seconds   REAL NOT NULL,
    text          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS transcript_asset ON transcript_segments(asset_id);
CREATE INDEX IF NOT EXISTS transcript_scene ON transcript_segments(scene_id);

CREATE VIRTUAL TABLE IF NOT EXISTS transcript_fts USING fts5(
    text,
    content='transcript_segments',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS transcript_ai AFTER INSERT ON transcript_segments BEGIN
    INSERT INTO transcript_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS transcript_ad AFTER DELETE ON transcript_segments BEGIN
    INSERT INTO transcript_fts(transcript_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS transcript_au AFTER UPDATE ON transcript_segments BEGIN
    INSERT INTO transcript_fts(transcript_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO transcript_fts(rowid, text) VALUES (new.id, new.text);
END;

CREATE TABLE IF NOT EXISTS run_state (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


@dataclass(slots=True)
class FaceRecord:
    person_id: str | None
    person_name: str | None
    distance: float | None
    score: float
    bbox: tuple[int, int, int, int]


@dataclass(slots=True)
class SceneRecord:
    idx: int
    start_seconds: float
    end_seconds: float
    vector: np.ndarray | None = None
    label: str | None = None
    label_score: float | None = None
    thumb_path: str | None = None
    faces: list[FaceRecord] | None = None


@dataclass(slots=True)
class TranscriptRecord:
    start_seconds: float
    end_seconds: float
    text: str


class VectorFile:
    """Append-only float32 matrix. Row indices are handed out by `append`."""

    def __init__(self, path: Path, dim: int) -> None:
        self.path = path
        self.dim = dim

    @property
    def rows(self) -> int:
        if not self.path.exists():
            return 0
        size = self.path.stat().st_size
        stride = self.dim * 4
        if size % stride:
            raise MomentsError(
                f"{self.path} is {size} bytes, not a whole number of {self.dim}-dim vectors. "
                "Run `immich-moments index --reindex` to rebuild it."
            )
        return size // stride

    def append(self, vector: np.ndarray) -> int:
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        if vec.shape != (self.dim,):
            raise DimensionMismatch(f"expected a {self.dim}-dim vector, got shape {vec.shape}")
        row = self.rows
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("ab") as handle:
            handle.write(vec.tobytes())
        return row

    def truncate(self, rows: int) -> None:
        """Drop everything past `rows`, including a half-written vector at the tail."""
        if not self.path.exists():
            return
        wanted = rows * self.dim * 4
        if self.path.stat().st_size > wanted:
            with self.path.open("r+b") as handle:
                handle.truncate(wanted)

    def read_all(self) -> np.ndarray:
        if not self.path.exists():
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.fromfile(self.path, dtype=np.float32).reshape(-1, self.dim)


class Store:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.path = config.db_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The web app runs searches in a worker thread, so the connection outlives its
        # creating thread. sqlite3.threadsafety is 3 on every build we support, and the app
        # holds a lock so only one thread is ever inside it.
        self.db = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.set_state("schema_version", str(SCHEMA_VERSION))
        self._reclaim_orphan_vectors()

    def _reclaim_orphan_vectors(self) -> None:
        """Vectors are appended before the transaction that points at them, so a run killed in
        between leaves rows nothing references. The file is append-only, so the tail is reusable."""
        stored_dim = self.get_state("vector_dim")
        if stored_dim is None:
            return
        top = self.db.execute("SELECT MAX(vector_row) AS top FROM scenes").fetchone()["top"]
        rows = 0 if top is None else int(top) + 1
        VectorFile(self.config.vectors_path, int(stored_dim)).truncate(rows)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        self.db.execute("BEGIN")
        try:
            yield self.db
        except BaseException:
            self.db.execute("ROLLBACK")
            raise
        else:
            self.db.execute("COMMIT")

    # ---- run state -------------------------------------------------------

    def get_state(self, key: str, default: str | None = None) -> str | None:
        row = self.db.execute("SELECT value FROM run_state WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_state(self, key: str, value: str) -> None:
        self.db.execute(
            "INSERT INTO run_state(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def vectors(self, dim: int | None = None) -> VectorFile:
        stored = self.get_state("vector_dim")
        if dim is None:
            if stored is None:
                raise MomentsError("nothing has been indexed yet; run `immich-moments index` first")
            dim = int(stored)
        return VectorFile(self.config.vectors_path, dim)

    def check_model(self, clip_model: str, dim: int, *, reindex: bool) -> None:
        """Refuse to mix vector spaces: a model or dimension change forces a rebuild."""
        stored_model = self.get_state("clip_model")
        stored_dim = self.get_state("vector_dim")
        changed = (stored_model is not None and stored_model != clip_model) or (
            stored_dim is not None and int(stored_dim) != dim
        )
        if changed and not reindex:
            raise DimensionMismatch(
                f"the index was built with CLIP model {stored_model!r} ({stored_dim}-dim) but the "
                f"server now reports {clip_model!r} ({dim}-dim). Mixing the two would make every "
                "score meaningless. Re-run with --reindex to rebuild from scratch."
            )
        if reindex:
            self.reset_index()
        self.set_state("clip_model", clip_model)
        self.set_state("vector_dim", str(dim))

    def assert_model(self, clip_model: str) -> None:
        """Searching is the other half of check_model: the query and the index share a space."""
        stored = self.get_state("clip_model")
        if stored is not None and stored != clip_model:
            raise DimensionMismatch(
                f"the index was built with CLIP model {stored!r} but the server now reports "
                f"{clip_model!r}. Every score would be meaningless. Run "
                "`immich-moments index --reindex` to rebuild."
            )

    def reset_index(self) -> None:
        with self.transaction() as db:
            db.execute("DELETE FROM transcript_segments")
            db.execute("DELETE FROM scenes")
            db.execute("DELETE FROM people_refs")
            db.execute(
                "UPDATE assets SET visual_indexed_at = NULL, audio_indexed_at = NULL, "
                "status = 'pending', error = NULL"
            )
        self.db.execute("INSERT INTO transcript_fts(transcript_fts) VALUES ('rebuild')")
        self.config.vectors_path.unlink(missing_ok=True)

    # ---- assets ----------------------------------------------------------

    def upsert_asset(
        self,
        asset_id: str,
        *,
        original_file_name: str,
        file_created_at: str | None,
        updated_at: str | None,
        duration_seconds: float | None = None,
    ) -> None:
        self.db.execute(
            """
            INSERT INTO assets (id, original_file_name, file_created_at, updated_at, duration_seconds)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                original_file_name = excluded.original_file_name,
                file_created_at    = excluded.file_created_at,
                updated_at         = excluded.updated_at,
                duration_seconds   = COALESCE(excluded.duration_seconds, assets.duration_seconds)
            """,
            (asset_id, original_file_name, file_created_at, updated_at, duration_seconds),
        )

    def set_media_info(self, asset_id: str, duration_seconds: float, has_audio: bool) -> None:
        """What ffprobe says about the file, which beats what the API said about the asset."""
        self.db.execute(
            "UPDATE assets SET duration_seconds = ?, has_audio = ? WHERE id = ?",
            (duration_seconds, int(has_audio), asset_id),
        )

    def set_asset_status(self, asset_id: str, status: str, error: str | None = None) -> None:
        self.db.execute("UPDATE assets SET status = ?, error = ? WHERE id = ?", (status, error, asset_id))

    def asset(self, asset_id: str) -> sqlite3.Row | None:
        return self.db.execute("SELECT * FROM assets WHERE id = ?", (asset_id,)).fetchone()

    def assets_needing(self, phase: str) -> list[sqlite3.Row]:
        column = {"visual": "visual_indexed_at", "audio": "audio_indexed_at"}[phase]
        return self.db.execute(
            f"SELECT * FROM assets WHERE {column} IS NULL AND status != 'unavailable' "  # noqa: S608
            "ORDER BY file_created_at DESC, id"
        ).fetchall()

    def indexed_asset_ids(self) -> list[str]:
        """Assets with a visual index, oldest first, which is what write-back walks."""
        rows = self.db.execute(
            "SELECT id FROM assets WHERE visual_indexed_at IS NOT NULL ORDER BY file_created_at"
        ).fetchall()
        return [row["id"] for row in rows]

    def thumbnail_names(self) -> set[str]:
        """Thumbnail filenames still referenced by a scene, so the rest can be deleted."""
        rows = self.db.execute("SELECT thumb_path FROM scenes WHERE thumb_path IS NOT NULL").fetchall()
        return {row["thumb_path"] for row in rows}

    def counts(self) -> dict[str, int]:
        one = self.db.execute(
            """
            SELECT (SELECT COUNT(*) FROM assets)                                     AS assets,
                   (SELECT COUNT(*) FROM scenes)                                     AS scenes,
                   (SELECT COUNT(*) FROM transcript_segments)                        AS segments,
                   (SELECT COUNT(*) FROM people_refs)                                AS people,
                   (SELECT COUNT(*) FROM assets WHERE visual_indexed_at IS NOT NULL) AS visual_done,
                   (SELECT COUNT(*) FROM assets WHERE audio_indexed_at IS NOT NULL)  AS audio_done,
                   (SELECT COUNT(*) FROM assets WHERE status = 'unavailable')        AS unavailable
            """
        ).fetchone()
        return dict(one)

    # ---- scenes ----------------------------------------------------------

    def replace_scenes(
        self, asset_id: str, scenes: Sequence[SceneRecord], vectors: VectorFile, *, indexed_at: str
    ) -> list[int]:
        """Swap in a fresh set of scenes for one asset and mark its visual phase done."""
        rows = [vectors.append(s.vector) if s.vector is not None else None for s in scenes]
        scene_ids: list[int] = []
        with self.transaction() as db:
            db.execute("DELETE FROM scenes WHERE asset_id = ?", (asset_id,))
            for scene, row in zip(scenes, rows, strict=True):
                cursor = db.execute(
                    "INSERT INTO scenes (asset_id, idx, start_seconds, end_seconds, vector_row, "
                    "label, label_score, thumb_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        asset_id,
                        scene.idx,
                        scene.start_seconds,
                        scene.end_seconds,
                        row,
                        scene.label,
                        scene.label_score,
                        scene.thumb_path,
                    ),
                )
                scene_id = int(cursor.lastrowid)
                scene_ids.append(scene_id)
                for face in scene.faces or []:
                    db.execute(
                        "INSERT INTO scene_faces (scene_id, person_id, person_name, distance, score, bbox) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            scene_id,
                            face.person_id,
                            face.person_name,
                            face.distance,
                            face.score,
                            json.dumps(list(face.bbox)),
                        ),
                    )
            db.execute(
                "UPDATE assets SET visual_indexed_at = ?, status = 'indexed', error = NULL WHERE id = ?",
                (indexed_at, asset_id),
            )
        return scene_ids

    def scenes_for(self, asset_id: str) -> list[sqlite3.Row]:
        return self.db.execute("SELECT * FROM scenes WHERE asset_id = ? ORDER BY idx", (asset_id,)).fetchall()

    def faces_for_scenes(self, scene_ids: Sequence[int]) -> dict[int, list[sqlite3.Row]]:
        if not scene_ids:
            return {}
        placeholders = ",".join("?" * len(scene_ids))
        rows = self.db.execute(
            f"SELECT * FROM scene_faces WHERE scene_id IN ({placeholders}) "  # noqa: S608
            "AND person_id IS NOT NULL ORDER BY distance",
            tuple(scene_ids),
        ).fetchall()
        grouped: dict[int, list[sqlite3.Row]] = {}
        for row in rows:
            grouped.setdefault(row["scene_id"], []).append(row)
        return grouped

    # ---- transcripts -----------------------------------------------------

    def replace_transcript(
        self, asset_id: str, segments: Sequence[TranscriptRecord], *, indexed_at: str, has_audio: bool
    ) -> None:
        scenes = self.scenes_for(asset_id)
        with self.transaction() as db:
            db.execute("DELETE FROM transcript_segments WHERE asset_id = ?", (asset_id,))
            for segment in segments:
                db.execute(
                    "INSERT INTO transcript_segments (asset_id, scene_id, start_seconds, end_seconds, text) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        asset_id,
                        _scene_at(scenes, segment.start_seconds),
                        segment.start_seconds,
                        segment.end_seconds,
                        segment.text,
                    ),
                )
            db.execute(
                "UPDATE assets SET audio_indexed_at = ?, has_audio = ? WHERE id = ?",
                (indexed_at, int(has_audio), asset_id),
            )

    def transcript_for(self, asset_id: str) -> list[sqlite3.Row]:
        return self.db.execute(
            "SELECT * FROM transcript_segments WHERE asset_id = ? ORDER BY start_seconds", (asset_id,)
        ).fetchall()

    # ---- people ----------------------------------------------------------

    def put_person_ref(self, person_id: str, name: str, vector: np.ndarray, updated_at: str) -> None:
        vec = np.ascontiguousarray(vector, dtype=np.float32)
        self.db.execute(
            "INSERT INTO people_refs (person_id, name, vector, dim, updated_at) VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(person_id) DO UPDATE SET name = excluded.name, vector = excluded.vector, "
            "dim = excluded.dim, updated_at = excluded.updated_at",
            (person_id, name, vec.tobytes(), vec.shape[0], updated_at),
        )

    def people_refs(self) -> tuple[list[tuple[str, str]], np.ndarray]:
        rows = self.db.execute(
            "SELECT person_id, name, vector, dim FROM people_refs ORDER BY name"
        ).fetchall()
        if not rows:
            return [], np.zeros((0, 0), dtype=np.float32)
        dim = rows[0]["dim"]
        keep = [r for r in rows if r["dim"] == dim]
        matrix = np.stack([np.frombuffer(r["vector"], dtype=np.float32) for r in keep])
        return [(r["person_id"], r["name"]) for r in keep], matrix


def _scene_at(scenes: Sequence[sqlite3.Row], seconds: float) -> int | None:
    for scene in scenes:
        if scene["start_seconds"] <= seconds < scene["end_seconds"]:
            return scene["id"]
    return scenes[-1]["id"] if scenes else None
