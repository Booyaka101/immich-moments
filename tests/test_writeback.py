"""Write-back: idempotent, non-destructive, and silent under --dry-run."""

from __future__ import annotations

import json
import re

import httpx
import pytest

from immich_moments.config import Config
from immich_moments.immich import Annotations, ImmichClient
from immich_moments.store import FaceRecord, SceneRecord, Store, TranscriptRecord, VectorFile
from immich_moments.writeback import BEGIN, END, plan_asset, write_back

from conftest import unit

NOW = "2026-09-16T10:00:00Z"
DIM = 8
ASSET = "birthday"


class FakeImmich:
    """Just enough of the server to answer a write-back, recording every request it is sent."""

    def __init__(self, description: str = "") -> None:
        self.descriptions = {ASSET: description}
        self.tags: dict[str, str] = {}
        self.assigned: dict[str, list[str]] = {}
        self.requests: list[tuple[str, str]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))
        if request.method == "GET" and path.startswith("/api/assets/"):
            asset_id = path.rsplit("/", 1)[-1]
            # Immich returns the description under exifInfo and never at the top level.
            carried = [
                {"id": tag_id, "value": value}
                for value, tag_id in self.tags.items()
                if asset_id in self.assigned.get(tag_id, [])
            ]
            return httpx.Response(
                200,
                json={
                    "id": asset_id,
                    "exifInfo": {"description": self.descriptions[asset_id]},
                    "tags": carried,
                },
            )
        body = json.loads(request.content) if request.content else {}
        if request.method == "PUT" and path == "/api/tags":
            for name in body["tags"]:
                self.tags.setdefault(name, f"tag-{len(self.tags)}")
            return httpx.Response(200, json=[{"id": self.tags[n], "value": n} for n in body["tags"]])
        if request.method == "PUT" and path == "/api/tags/assets":
            for tag_id in body["tagIds"]:
                self.assigned.setdefault(tag_id, []).extend(body["assetIds"])
            return httpx.Response(200, json={"count": len(body["assetIds"])})
        if request.method == "PUT" and path.startswith("/api/assets/"):
            self.descriptions[path.rsplit("/", 1)[-1]] = body["description"]
            return httpx.Response(200, json={"id": ASSET})
        raise AssertionError(f"unexpected {request.method} {path}")

    @property
    def writes(self) -> list[tuple[str, str]]:
        return [call for call in self.requests if call[0] != "GET"]

    def client(self, config: Config) -> ImmichClient:
        return ImmichClient(config, transport=httpx.MockTransport(self.handler))


@pytest.fixture
def seeded(store: Store, config: Config) -> Store:
    store.check_model("ViT-B-32__openai", DIM, reindex=False)
    vectors = VectorFile(config.vectors_path, DIM)
    store.upsert_asset(
        ASSET,
        original_file_name="birthday.mp4",
        file_created_at="2026-06-01T00:00:00Z",
        updated_at="2026-06-01T00:00:00Z",
        duration_seconds=930.0,
    )
    store.replace_scenes(
        ASSET,
        [
            SceneRecord(0, 0.0, 462.0, vector=unit(1, DIM), label="a living room"),
            SceneRecord(
                1,
                462.0,
                480.0,
                vector=unit(2, DIM),
                label="birthday cake",
                faces=[
                    FaceRecord("p1", "Anna", 0.18, 0.99, (1, 2, 3, 4)),
                    FaceRecord("p2", "Tom", 0.24, 0.97, (5, 6, 7, 8)),
                ],
            ),
        ],
        vectors,
        indexed_at=NOW,
    )
    store.replace_transcript(
        ASSET,
        [TranscriptRecord(463.0, 466.0, "happy birthday to you")],
        indexed_at=NOW,
        has_audio=True,
    )
    return store


def block_of(text: str) -> str:
    match = re.search(re.escape(BEGIN) + r".*?" + re.escape(END), text, re.DOTALL)
    assert match, f"no fenced block in {text!r}"
    return match.group(0)


def test_the_planned_block_reads_like_the_worked_example(seeded: Store) -> None:
    plan = plan_asset(seeded, ASSET, current=Annotations("", set()))
    assert plan.description == (
        f'{BEGIN}\n00:00 a living room\n07:42 birthday cake (Anna, Tom) | "happy birthday to you"\n{END}'
    )
    assert plan.tags == [
        "moments/people/Anna",
        "moments/people/Tom",
        "moments/scene/a living room",
        "moments/scene/birthday cake",
    ]


def test_dry_run_sends_no_write_request(seeded: Store, config: Config) -> None:
    server = FakeImmich()
    with server.client(config) as immich:
        result = write_back(seeded, immich, [ASSET], dry_run=True)

    assert result.dry_run is True
    assert len(result.planned) == 1
    assert server.writes == []
    assert server.descriptions[ASSET] == ""
    assert result.descriptions_written == 0
    assert result.assets_tagged == 0


def test_a_real_write_back_tags_and_describes(seeded: Store, config: Config) -> None:
    server = FakeImmich()
    with server.client(config) as immich:
        result = write_back(seeded, immich, [ASSET], dry_run=False)

    assert result.descriptions_written == 1
    assert result.tags_created == 4
    assert result.assets_tagged == 4
    assert sorted(server.tags) == [
        "moments/people/Anna",
        "moments/people/Tom",
        "moments/scene/a living room",
        "moments/scene/birthday cake",
    ]
    assert "07:42 birthday cake (Anna, Tom)" in server.descriptions[ASSET]


def test_two_consecutive_runs_are_byte_identical(seeded: Store, config: Config) -> None:
    server = FakeImmich()
    with server.client(config) as immich:
        write_back(seeded, immich, [ASSET], dry_run=False)
        first = server.descriptions[ASSET]
        second_result = write_back(seeded, immich, [ASSET], dry_run=False)
        second = server.descriptions[ASSET]

    assert first == second
    assert second.count(BEGIN) == 1
    assert second_result.descriptions_written == 0  # nothing changed, so nothing was written


def test_existing_user_text_survives_and_is_not_duplicated(seeded: Store, config: Config) -> None:
    server = FakeImmich("Grandma's camcorder tape, digitised 2019.")
    with server.client(config) as immich:
        write_back(seeded, immich, [ASSET], dry_run=False)
        after_first = server.descriptions[ASSET]
        write_back(seeded, immich, [ASSET], dry_run=False)

    assert server.descriptions[ASSET] == after_first
    assert server.descriptions[ASSET].startswith("Grandma's camcorder tape, digitised 2019.")
    assert server.descriptions[ASSET].count(BEGIN) == 1


def test_user_text_after_the_block_is_kept_in_place(seeded: Store, config: Config) -> None:
    server = FakeImmich(f"before\n\n{BEGIN}\nstale line\n{END}\n\nafter")
    with server.client(config) as immich:
        write_back(seeded, immich, [ASSET], dry_run=False)

    written = server.descriptions[ASSET]
    assert written.startswith("before")
    assert written.endswith("after")
    assert "stale line" not in written
    assert "07:42 birthday cake (Anna, Tom)" in block_of(written)


def test_an_asset_with_nothing_to_say_is_left_alone(store: Store, config: Config) -> None:
    store.check_model("ViT-B-32__openai", DIM, reindex=False)
    store.upsert_asset(
        ASSET,
        original_file_name="blank.mp4",
        file_created_at=None,
        updated_at=None,
        duration_seconds=5.0,
    )
    store.replace_scenes(
        ASSET,
        [SceneRecord(0, 0.0, 5.0, vector=unit(1, DIM))],
        VectorFile(config.vectors_path, DIM),
        indexed_at=NOW,
    )

    server = FakeImmich("a note the user wrote")
    with server.client(config) as immich:
        result = write_back(store, immich, [ASSET], dry_run=False)

    assert result.planned == []
    assert server.writes == []
    assert server.descriptions[ASSET] == "a note the user wrote"


def test_a_block_is_removed_when_there_is_nothing_left_to_say(store: Store, config: Config) -> None:
    store.check_model("ViT-B-32__openai", DIM, reindex=False)
    store.upsert_asset(
        ASSET, original_file_name="blank.mp4", file_created_at=None, updated_at=None, duration_seconds=5.0
    )
    store.replace_scenes(
        ASSET,
        [SceneRecord(0, 0.0, 5.0, vector=unit(1, DIM))],
        VectorFile(config.vectors_path, DIM),
        indexed_at=NOW,
    )
    plan = plan_asset(store, ASSET, current=Annotations(f"keep me\n\n{BEGIN}\nold\n{END}", set()))
    assert plan.description == "keep me"
    assert plan.description_changed is True


def test_a_long_transcript_line_is_clipped(seeded: Store) -> None:
    seeded.replace_transcript(
        ASSET,
        [TranscriptRecord(463.0, 470.0, "happy birthday " * 40)],
        indexed_at=NOW,
        has_audio=True,
    )
    plan = plan_asset(seeded, ASSET, current=Annotations("", set()))
    line = next(line for line in plan.description.splitlines() if line.startswith("07:42"))
    assert line.endswith('…"')
    assert len(line) < 200


def test_planning_an_unknown_asset_is_a_key_error(store: Store) -> None:
    with pytest.raises(KeyError):
        plan_asset(store, "nope", current=Annotations("", set()))


def test_a_long_unbroken_take_does_not_repeat_itself(store: Store, config: Config) -> None:
    """Twelve scenes of the same room are one line, not twelve identical ones."""
    store.check_model("ViT-B-32__openai", DIM, reindex=False)
    store.upsert_asset(
        ASSET,
        original_file_name="piano.mp4",
        file_created_at=None,
        updated_at=None,
        duration_seconds=240.0,
    )
    store.replace_scenes(
        ASSET,
        [
            SceneRecord(
                idx, idx * 20.0, idx * 20.0 + 20.0, vector=unit(idx, DIM), label="someone playing a piano"
            )
            for idx in range(12)
        ],
        VectorFile(config.vectors_path, DIM),
        indexed_at=NOW,
    )

    plan = plan_asset(store, ASSET, current=Annotations("", set()))

    assert plan.description == f"{BEGIN}\n00:00 someone playing a piano\n{END}"
    assert plan.tags == ["moments/scene/someone playing a piano"]


def test_the_same_label_returning_later_is_said_again(store: Store, config: Config) -> None:
    """Only consecutive repeats collapse; coming back to the kitchen is worth a second line."""
    store.check_model("ViT-B-32__openai", DIM, reindex=False)
    store.upsert_asset(
        ASSET, original_file_name="day.mp4", file_created_at=None, updated_at=None, duration_seconds=90.0
    )
    store.replace_scenes(
        ASSET,
        [
            SceneRecord(0, 0.0, 30.0, vector=unit(1, DIM), label="a kitchen"),
            SceneRecord(1, 30.0, 60.0, vector=unit(2, DIM), label="a garden"),
            SceneRecord(2, 60.0, 90.0, vector=unit(3, DIM), label="a kitchen"),
        ],
        VectorFile(config.vectors_path, DIM),
        indexed_at=NOW,
    )

    plan = plan_asset(store, ASSET, current=Annotations("", set()))

    assert plan.description == (f"{BEGIN}\n00:00 a kitchen\n00:30 a garden\n01:00 a kitchen\n{END}")


def test_a_repeated_label_still_earns_a_line_when_someone_speaks(seeded: Store, config: Config) -> None:
    """The collapse compares the whole line, so speech and faces keep a scene in."""
    seeded.replace_scenes(
        ASSET,
        [
            SceneRecord(0, 0.0, 30.0, vector=unit(1, DIM), label="a living room"),
            SceneRecord(1, 30.0, 60.0, vector=unit(2, DIM), label="a living room"),
        ],
        VectorFile(config.vectors_path, DIM),
        indexed_at=NOW,
    )
    seeded.replace_transcript(
        ASSET, [TranscriptRecord(31.0, 33.0, "make a wish")], indexed_at=NOW, has_audio=True
    )

    plan = plan_asset(seeded, ASSET, current=Annotations("", set()))

    assert plan.description == (f'{BEGIN}\n00:00 a living room\n00:30 a living room | "make a wish"\n{END}')


def test_a_second_run_has_no_tags_left_to_add(seeded: Store, config: Config) -> None:
    """The asset comes back carrying its tags, so the plan is empty rather than merely harmless."""
    server = FakeImmich()
    with server.client(config) as immich:
        first = write_back(seeded, immich, [ASSET], dry_run=False)
        second = write_back(seeded, immich, [ASSET], dry_run=False)

    assert first.planned[0].tags
    assert second.planned == []
    assert second.assets_tagged == 0
    assert second.descriptions_written == 0
