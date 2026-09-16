"""The index run: enumerate, download, segment, embed, detect faces, transcribe, checkpoint.

Visual work and speech work are two separate passes over the library. The visual pass writes
a small 16 kHz FLAC next to the index while it already has the video open, so the speech pass
never has to download an original a second time.
"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .config import Config
from .errors import AssetUnavailable, ConfigError, MediaError, MomentsError
from .faces import PeopleIndex, faces_in_frame, load_people_index, refresh_people_refs, rematch_faces
from .immich import ImmichClient
from .labels import LabelIndex, build_label_index
from .media import extract_audio, extract_frame, probe
from .ml import MLClient
from .scenes import detect_scenes
from .store import SceneRecord, Store

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


@dataclass
class PhaseTiming:
    name: str
    seconds: float = 0.0
    assets: int = 0


@dataclass
class IndexReport:
    discovered: int = 0
    visual_indexed: int = 0
    audio_indexed: int = 0
    scenes: int = 0
    segments: int = 0
    unavailable: list[str] = field(default_factory=list)
    failed: list[tuple[str, str]] = field(default_factory=list)
    people_refs: int = 0
    faces_rematched: int = 0
    pruned: list[str] = field(default_factory=list)
    timings: list[PhaseTiming] = field(default_factory=list)
    no_audio_track: int = 0
    no_speech: int = 0

    def timing(self, name: str) -> PhaseTiming:
        for entry in self.timings:
            if entry.name == name:
                return entry
        entry = PhaseTiming(name)
        self.timings.append(entry)
        return entry


class Indexer:
    def __init__(
        self,
        config: Config,
        store: Store,
        immich: ImmichClient,
        ml: MLClient,
        *,
        progress: Progress | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self.immich = immich
        self.ml = ml
        self.say: Progress = progress or (lambda _message: None)

    # ---- phases ----------------------------------------------------------

    def discover(
        self, since: str | None, limit: int | None = None, *, prune: bool = False
    ) -> tuple[int, list[str]]:
        """Record every video asset Immich knows about. Cheap, and safe to repeat.

        With `prune`, anything the index holds that Immich did not list is dropped, which only
        means something when the whole library was walked. Returns (found, file names dropped).
        """
        if prune and (since or limit is not None):
            raise ConfigError("--prune has to walk the whole library, so it cannot take --since or --limit.")
        seen: list[str] = []
        truncated = False
        for asset in self.immich.iter_videos(updated_after=since):
            asset_id = asset.get("id")
            if not asset_id:
                continue
            seen.append(asset_id)
            self.store.upsert_asset(
                asset_id,
                original_file_name=asset.get("originalFileName") or asset_id,
                file_created_at=asset.get("fileCreatedAt"),
                updated_at=asset.get("updatedAt"),
                duration_seconds=_duration(asset.get("duration")),
            )
            if limit is not None and len(seen) >= limit:
                truncated = True
                break
        # Checkpointing a truncated enumeration would hide every video --limit never reached from
        # the next `--since auto` run, permanently.
        if not truncated:
            self.store.set_state("last_discovery", _now())
        if not prune:
            return len(seen), []
        gone = self.store.prune_assets(seen)
        for row in gone:
            (self.config.audio_dir / f"{row['id']}.flac").unlink(missing_ok=True)
        if gone:
            _prune_thumbnails(self.config, self.store)
        return len(seen), [row["original_file_name"] for row in gone]

    def refresh_people(self) -> tuple[int, int]:
        """(people with a reference, faces whose name changed as a result)."""
        matched, skipped = refresh_people_refs(self.store, self.immich, self.ml, now=_now())
        if skipped:
            log.info("%d Immich people skipped (unnamed, hidden or no detectable thumbnail)", skipped)
        rematched = rematch_faces(
            self.store, load_people_index(self.store), max_distance=self.config.face_max_distance
        )
        return matched, rematched

    def visual_pass(self, report: IndexReport, labels: LabelIndex, limit: int | None = None) -> None:
        people = load_people_index(self.store)
        pending = self.store.assets_needing("visual")[: limit or None]
        timing = report.timing("visual")
        for position, asset in enumerate(pending, start=1):
            self.say(f"visual {position}/{len(pending)}  {asset['original_file_name']}")
            started = time.monotonic()
            try:
                scenes = self._index_one_visual(asset["id"], labels, people)
            except AssetUnavailable as exc:
                log.warning("%s", exc)
                self.store.set_asset_status(asset["id"], "unavailable", str(exc))
                report.unavailable.append(asset["original_file_name"])
                continue
            except (MediaError, MomentsError) as exc:
                log.warning("%s: %s", asset["original_file_name"], exc)
                self.store.set_asset_status(asset["id"], "failed", str(exc))
                report.failed.append((asset["original_file_name"], str(exc)))
                continue
            timing.seconds += time.monotonic() - started
            timing.assets += 1
            report.visual_indexed += 1
            report.scenes += scenes

    def audio_pass(self, report: IndexReport, limit: int | None = None) -> None:
        from .transcribe import Transcriber

        pending = [
            asset for asset in self.store.assets_needing("audio") if asset["visual_indexed_at"] is not None
        ][: limit or None]
        if not pending:
            return
        transcriber = Transcriber(self.config)
        timing = report.timing("speech")
        try:
            for position, asset in enumerate(pending, start=1):
                self.say(f"speech {position}/{len(pending)}  {asset['original_file_name']}")
                started = time.monotonic()
                try:
                    count = self._index_one_audio(asset["id"], transcriber)
                except AssetUnavailable as exc:
                    log.warning("%s", exc)
                    self.store.set_asset_status(asset["id"], "unavailable", str(exc))
                    report.unavailable.append(asset["original_file_name"])
                    continue
                except (MediaError, MomentsError) as exc:
                    log.warning("%s: %s", asset["original_file_name"], exc)
                    self.store.set_asset_status(asset["id"], "failed", str(exc))
                    report.failed.append((asset["original_file_name"], str(exc)))
                    continue
                timing.seconds += time.monotonic() - started
                timing.assets += 1
                report.audio_indexed += 1
                if count is None:
                    report.no_audio_track += 1
                elif count == 0:
                    report.no_speech += 1
                else:
                    report.segments += count
        finally:
            transcriber.unload()

    # ---- per asset -------------------------------------------------------

    def _index_one_visual(self, asset_id: str, labels: LabelIndex, people: PeopleIndex) -> int:
        with tempfile.TemporaryDirectory(prefix="immich-moments-") as workdir:
            original = self.immich.download_original(asset_id, Path(workdir) / f"{asset_id}.bin")
            info = probe(original)
            self.store.set_media_info(asset_id, info.duration_seconds, info.has_audio)
            cut_list = detect_scenes(
                original,
                info,
                threshold=self.config.scene_threshold,
                min_seconds=self.config.min_scene_seconds,
                max_seconds=self.config.max_scene_seconds,
            )
            records: list[SceneRecord] = []
            for scene in cut_list:
                jpeg = extract_frame(original, scene.mid_seconds, info)
                vector = self.ml.embed_image(jpeg)
                label = labels.best(vector)
                thumb = self.config.thumbs_dir / f"{asset_id}-{scene.idx:04d}.jpg"
                thumb.write_bytes(jpeg)
                records.append(
                    SceneRecord(
                        idx=scene.idx,
                        start_seconds=scene.start_seconds,
                        end_seconds=scene.end_seconds,
                        vector=vector,
                        label=label[0] if label else None,
                        label_score=label[1] if label else None,
                        thumb_path=thumb.name,
                        # Detected even when nobody is named yet: the embedding is kept, so a
                        # name given in Immich later reaches this scene without a reindex.
                        faces=faces_in_frame(
                            self.ml,
                            jpeg,
                            people,
                            min_score=self.config.face_min_score,
                            max_distance=self.config.face_max_distance,
                        ),
                    )
                )

            if info.has_audio:
                extract_audio(original, self.config.audio_dir / f"{asset_id}.flac")

        vectors = self.store.vectors()
        self.store.replace_scenes(asset_id, records, vectors, indexed_at=_now())
        return len(records)

    def _index_one_audio(self, asset_id: str, transcriber) -> int | None:
        """Segment count, or None when the video carries no audio track to transcribe."""
        audio = self.config.audio_dir / f"{asset_id}.flac"
        if not audio.exists() and not self._reextract_audio(asset_id, audio):
            self.store.replace_transcript(asset_id, [], indexed_at=_now(), has_audio=False)
            return None
        segments = transcriber.transcribe(audio)
        self.store.replace_transcript(asset_id, segments, indexed_at=_now(), has_audio=True)
        audio.unlink(missing_ok=True)
        return len(segments)

    def _reextract_audio(self, asset_id: str, audio: Path) -> bool:
        """Rebuild a missing sidecar from the original. True once one is on disk.

        The visual pass writes the sidecar and the speech pass deletes it, so a run that was
        interrupted between the two, or a speech pass on a box that never did the visual one,
        finds nothing. Believing the absence would silently file a talkative video as silent.
        """
        if not self.store.asset(asset_id)["has_audio"]:
            return False
        log.info("%s: audio sidecar is missing, taking it from the original again", asset_id)
        with tempfile.TemporaryDirectory(prefix="immich-moments-") as workdir:
            original = self.immich.download_original(asset_id, Path(workdir) / f"{asset_id}.bin")
            if not probe(original).has_audio:
                return False
            extract_audio(original, audio)
        return True


def run_index(
    config: Config,
    store: Store,
    immich: ImmichClient,
    ml: MLClient,
    *,
    since: str | None,
    limit: int | None,
    phases: tuple[str, ...],
    labels_path: Path | None,
    reindex: bool,
    prune: bool = False,
    progress: Progress | None = None,
) -> IndexReport:
    indexer = Indexer(config, store, immich, ml, progress=progress)
    report = IndexReport()

    started = time.monotonic()
    report.discovered, report.pruned = indexer.discover(since, limit, prune=prune)
    report.timing("discover").seconds = time.monotonic() - started
    report.timing("discover").assets = report.discovered

    if "visual" in phases:
        started = time.monotonic()
        report.people_refs, report.faces_rematched = indexer.refresh_people()
        report.timing("people").seconds = time.monotonic() - started
        report.timing("people").assets = report.people_refs

        labels = build_label_index(
            ml,
            config.data_dir,
            labels_path=labels_path,
            min_similarity=config.label_min_similarity,
        )
        indexer.visual_pass(report, labels, limit)

    if "audio" in phases:
        indexer.audio_pass(report, limit)

    if reindex:
        _prune_thumbnails(config, store)
    return report


def _prune_thumbnails(config: Config, store: Store) -> None:
    keep = store.thumbnail_names()
    for path in config.thumbs_dir.glob("*.jpg"):
        if path.name not in keep:
            path.unlink(missing_ok=True)


def _duration(raw: object) -> float | None:
    """Immich 3.2 reports duration as integer milliseconds; older releases sent `HH:MM:SS.ssssss`."""
    if raw is None or raw == "":
        return None
    if isinstance(raw, int | float):
        return float(raw) / 1000.0 or None
    try:
        numbers = [float(part) for part in str(raw).split(":")]
    except ValueError:
        return None
    seconds = 0.0
    for number in numbers:
        seconds = seconds * 60 + number
    return seconds or None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")
