"""Push what was indexed back into Immich as tags and a fenced description block.

Everything here is non-destructive. Tags are added, never removed, and the description is
only ever edited between the two markers, so whatever the user wrote around them survives.
Running it twice produces byte-identical results.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field

from .errors import AssetUnavailable
from .immich import Annotations, ImmichClient
from .search import format_timestamp, unique_names
from .store import Store

log = logging.getLogger(__name__)

BEGIN = "<!-- immich-moments:begin -->"
END = "<!-- immich-moments:end -->"
BLOCK = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.DOTALL)

PEOPLE_TAG = "moments/people/{}"
SCENE_TAG = "moments/scene/{}"
MAX_LINES = 12
SNIPPET_CHARS = 120


@dataclass(slots=True)
class AssetPlan:
    asset_id: str
    original_file_name: str
    tags: list[str] = field(default_factory=list)
    description: str = ""
    description_changed: bool = False
    current_description: str = ""

    @property
    def empty(self) -> bool:
        return not self.tags and not self.description_changed


@dataclass(slots=True)
class WriteResult:
    planned: list[AssetPlan]
    considered: int = 0
    skipped: list[str] = field(default_factory=list)
    tags_created: int = 0
    assets_tagged: int = 0
    descriptions_written: int = 0
    dry_run: bool = True


def plan_asset(store: Store, asset_id: str, *, current: Annotations) -> AssetPlan:
    asset = store.asset(asset_id)
    if asset is None:
        raise KeyError(asset_id)
    scenes = store.scenes_for(asset_id)
    faces = store.faces_for_scenes([scene["id"] for scene in scenes])
    transcript = _transcript_by_scene(store, asset_id)

    people: list[str] = []
    labels: list[str] = []
    lines: list[str] = []
    for scene in scenes:
        names = unique_names(face["person_name"] for face in faces.get(scene["id"], []))
        for name in names:
            if name not in people:
                people.append(name)
        if scene["label"] and scene["label"] not in labels:
            labels.append(scene["label"])
        line = _line(scene, names, transcript.get(scene["id"], ""))
        if line:
            lines.append(line)

    block = _block(_collapse(lines)[:MAX_LINES])
    description = _merge(current.description, block)
    wanted = [PEOPLE_TAG.format(name) for name in people] + [SCENE_TAG.format(label) for label in labels]
    return AssetPlan(
        asset_id=asset_id,
        original_file_name=asset["original_file_name"],
        tags=[tag for tag in wanted if tag not in current.tags],
        description=description,
        description_changed=description != current.description,
        current_description=current.description,
    )


def _line(scene: sqlite3.Row, names: list[str], snippet: str) -> str:
    """One description line per scene that has something worth saying about it."""
    parts = [format_timestamp(scene["start_seconds"])]
    if scene["label"]:
        parts.append(scene["label"])
    if names:
        parts.append(f"({', '.join(names)})")
    line = " ".join(parts)
    if snippet:
        line += f' | "{_clip(snippet)}"'
    return line if (scene["label"] or names or snippet) else ""


def _clip(text: str) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= SNIPPET_CHARS:
        return collapsed
    return collapsed[: SNIPPET_CHARS - 1].rstrip() + "…"


def _collapse(lines: list[str]) -> list[str]:
    """Drop a line that says the same thing as the one before it, timestamp aside.

    A steady thirteen-minute take of the same room segments into a dozen scenes that all
    label identically. Repeating that line a dozen times tells a reader nothing and dilutes
    the description Immich's own search reads.
    """
    kept: list[str] = []
    previous = None
    for line in lines:
        _, _, rest = line.partition(" ")
        if rest and rest == previous:
            continue
        previous = rest
        kept.append(line)
    return kept


def _block(lines: list[str]) -> str:
    return "" if not lines else "\n".join([BEGIN, *lines, END])


def _merge(current: str, block: str) -> str:
    """Replace the fenced block in `current`, or append it, leaving user text untouched."""
    existing = BLOCK.search(current or "")
    if existing:
        if not block:
            return re.sub(r"\n*" + BLOCK.pattern + r"\n*", "\n", current, flags=re.DOTALL).strip()
        return (current[: existing.start()] + block + current[existing.end() :]).strip()
    if not block:
        return current
    user_text = (current or "").strip()
    return f"{user_text}\n\n{block}" if user_text else block


def _transcript_by_scene(store: Store, asset_id: str) -> dict[int, str]:
    grouped: dict[int, list[str]] = {}
    for row in store.transcript_for(asset_id):
        if row["scene_id"] is not None:
            grouped.setdefault(row["scene_id"], []).append(row["text"])
    return {scene_id: " ".join(parts) for scene_id, parts in grouped.items()}


def write_back(store: Store, immich: ImmichClient, asset_ids: list[str], *, dry_run: bool) -> WriteResult:
    """Plan every mutation first, then apply it. With dry_run, no write request is sent."""
    plans: list[AssetPlan] = []
    skipped: list[str] = []
    for asset_id in asset_ids:
        try:
            current = immich.asset_annotations(asset_id)
        except AssetUnavailable as exc:
            log.warning("%s", exc)
            skipped.append(store.asset(asset_id)["original_file_name"])
            continue
        plan = plan_asset(store, asset_id, current=current)
        if not plan.empty:
            plans.append(plan)

    result = WriteResult(planned=plans, considered=len(asset_ids), skipped=skipped, dry_run=dry_run)
    if dry_run or not plans:
        return result

    tag_names = sorted({tag for plan in plans for tag in plan.tags})
    tag_ids = immich.upsert_tags(tag_names) if tag_names else {}
    result.tags_created = len(tag_ids)

    by_tag: dict[str, list[str]] = {}
    for plan in plans:
        for tag in plan.tags:
            by_tag.setdefault(tag, []).append(plan.asset_id)
    for tag, assets in by_tag.items():
        tag_id = tag_ids.get(tag)
        if tag_id is None:
            log.warning("Immich did not return an id for tag %s, skipping it", tag)
            continue
        result.assets_tagged += immich.bulk_tag_assets([tag_id], assets)

    for plan in plans:
        if plan.description_changed:
            immich.update_asset(plan.asset_id, description=plan.description)
            result.descriptions_written += 1
    return result
