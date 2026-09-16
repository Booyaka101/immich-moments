"""Face detection on scene frames, matched against Immich's own named people.

Immich does not expose face embeddings over REST (`PersonResponseDto` carries a thumbnail
path and nothing else), so the reference vector for each person is re-derived by pushing that
person's thumbnail back through the same `buffalo_l` pipeline the server uses. The vectors
therefore live in the same space as the ones taken from video frames, which is all the
matching needs.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np

from .errors import MLError
from .ml import DetectedFace, MLClient
from .store import FaceRecord, Store

log = logging.getLogger(__name__)

# SCRFD wants context around a face; person thumbnails are cropped tight, so pad before detecting.
THUMBNAIL_PAD = 0.45
MIN_THUMBNAIL_SIDE = 256


@dataclass(slots=True)
class PeopleIndex:
    identities: list[tuple[str, str]]
    matrix: np.ndarray

    def __len__(self) -> int:
        return len(self.identities)

    def match(self, embedding: np.ndarray, max_distance: float) -> tuple[str, str, float] | None:
        """Nearest named person by cosine distance, or None if nobody is close enough."""
        return self.match_many(embedding.reshape(1, -1), max_distance)[0]

    def match_many(self, embeddings: np.ndarray, max_distance: float) -> list[tuple[str, str, float] | None]:
        """One match per row, in one matmul, which is what re-matching a whole index needs."""
        if not self.identities or embeddings.shape[1] != self.matrix.shape[1]:
            return [None] * len(embeddings)
        distances = 1.0 - embeddings @ self.matrix.T
        nearest = np.argmin(distances, axis=1)
        hits: list[tuple[str, str, float] | None] = []
        for row, best in enumerate(nearest):
            distance = float(distances[row, best])
            hits.append((*self.identities[int(best)], distance) if distance <= max_distance else None)
        return hits


def load_people_index(store: Store) -> PeopleIndex:
    identities, matrix = store.people_refs()
    return PeopleIndex(identities, matrix)


def refresh_people_refs(store: Store, immich, ml: MLClient, *, now: str) -> tuple[int, int]:
    """Build a reference embedding per named Immich person. Returns (matched, skipped).

    Someone Immich no longer lists by name (deleted, merged, hidden or un-named) loses their
    reference here, so the faces that were theirs get re-matched on the next pass. A person
    whose thumbnail merely failed to fetch keeps the reference they already had.
    """
    matched = skipped = 0
    named: list[str] = []
    for person in immich.people():
        name = (person.get("name") or "").strip()
        person_id = person.get("id")
        if not name or not person_id:
            skipped += 1
            continue
        named.append(person_id)
        try:
            thumbnail = immich.person_thumbnail(person_id)
        except Exception as exc:  # one bad thumbnail must not end the run
            log.warning("person %s: thumbnail unavailable (%s)", name, exc)
            skipped += 1
            continue
        if not thumbnail:
            skipped += 1
            continue
        embedding = person_reference_embedding(ml, thumbnail)
        if embedding is None:
            log.warning("person %s: no face found in their Immich thumbnail, skipping", name)
            skipped += 1
            continue
        store.put_person_ref(person_id, name, embedding, now)
        matched += 1
    dropped = store.delete_person_refs_except(named)
    if dropped:
        log.info("%d people are no longer named in Immich; their reference is gone", dropped)
    return matched, skipped


def rematch_faces(store: Store, people: PeopleIndex, *, max_distance: float) -> int:
    """Put every stored face against the current references again. Returns how many changed.

    Naming, renaming or merging someone in Immich changes what an already-indexed face should
    be called. The embedding is on disk, so no video has to be downloaded again for that.
    """
    changed = store.rename_people()
    rows = store.faces_with_embeddings()
    decoded = [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]
    # Faces embedded by a different face model than the references cannot be compared with
    # them; they keep whatever name they have until a reindex.
    if len(people):
        keep = [i for i, vector in enumerate(decoded) if vector.shape[0] == people.matrix.shape[1]]
        rows, decoded = [rows[i] for i in keep], [decoded[i] for i in keep]
    if not rows:
        return changed
    hits = people.match_many(np.stack(decoded), max_distance)

    # A merge in Immich can leave one scene with two faces that now both match the same
    # person. One frame cannot show someone twice, so only the closest keeps the name.
    closest: dict[tuple[int, str], tuple[float, int]] = {}
    for row, hit in zip(rows, hits, strict=True):
        if hit is None:
            continue
        key = (row["scene_id"], hit[0])
        if key not in closest or hit[2] < closest[key][0]:
            closest[key] = (hit[2], row["id"])

    updates: list[tuple[str | None, str | None, float | None, int]] = []
    for row, hit in zip(rows, hits, strict=True):
        if hit is not None and closest[(row["scene_id"], hit[0])][1] != row["id"]:
            hit = None
        person_id, name, distance = hit if hit is not None else (None, None, None)
        if (person_id, name) != (row["person_id"], row["person_name"]):
            updates.append((person_id, name, distance, row["id"]))
    store.set_face_matches(updates)
    return changed + len(updates)


def person_reference_embedding(ml: MLClient, thumbnail: bytes) -> np.ndarray | None:
    try:
        faces = ml.detect_faces(pad_thumbnail(thumbnail))
    except MLError as exc:
        log.warning("face detection on a person thumbnail failed: %s", exc)
        return None
    if not faces:
        return None
    return max(faces, key=lambda face: face.score).embedding


def pad_thumbnail(thumbnail: bytes) -> bytes:
    """Re-encode the thumbnail with a border so the detector has context to work with."""
    from PIL import Image, ImageOps, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(thumbnail)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        return thumbnail
    if max(image.size) < MIN_THUMBNAIL_SIDE:
        scale = MIN_THUMBNAIL_SIDE / max(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    border = round(max(image.size) * THUMBNAIL_PAD)
    padded = ImageOps.expand(image, border=border, fill=(0, 0, 0))
    buffer = io.BytesIO()
    padded.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def faces_in_frame(
    ml: MLClient, jpeg: bytes, people: PeopleIndex, *, min_score: float, max_distance: float
) -> list[FaceRecord]:
    """Detected faces for one frame, each tagged with a person when one is close enough."""
    detected: list[DetectedFace] = ml.detect_faces(jpeg)
    records: list[FaceRecord] = []
    for face in detected:
        if face.score < min_score:
            continue
        hit = people.match(face.embedding, max_distance)
        records.append(
            FaceRecord(
                person_id=hit[0] if hit else None,
                person_name=hit[1] if hit else None,
                distance=hit[2] if hit else None,
                score=face.score,
                bbox=face.bbox,
                embedding=face.embedding,
            )
        )
    return _best_per_person(records)


def _best_per_person(records: list[FaceRecord]) -> list[FaceRecord]:
    """One frame cannot show the same person twice; keep the closest match per identity."""
    best: dict[str, FaceRecord] = {}
    unnamed: list[FaceRecord] = []
    for record in records:
        if record.person_id is None:
            unnamed.append(record)
            continue
        current = best.get(record.person_id)
        if current is None or (record.distance or 1.0) < (current.distance or 1.0):
            best[record.person_id] = record
    return sorted(best.values(), key=lambda r: r.distance or 1.0) + unnamed
