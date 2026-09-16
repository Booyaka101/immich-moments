"""The command line itself: version, help, and what a user sees when something is wrong.

Every failure here must reach the terminal as a sentence, not a traceback, so these drive
`main()` rather than the Typer app where the error handling lives.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from typer.testing import CliRunner

from immich_moments import __version__
from immich_moments.cli import app, main
from immich_moments.config import Config
from immich_moments.errors import ConfigError, ImmichError, StorageError
from immich_moments.labels import LabelIndex
from immich_moments.search import Hit
from immich_moments.store import FaceRecord, SceneRecord, Store

from conftest import unit

runner = CliRunner()

CREDENTIALS = ("IMMICH_URL", "IMMICH_API_KEY", "IMMICH_ML_URL", "DATA_DIR", "IMMICH_MOMENTS_CONFIG")


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A developer's own server must not decide whether these tests pass."""
    for name in CREDENTIALS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)


def run(*args: str, monkeypatch: pytest.MonkeyPatch) -> tuple[int, str]:
    """Drive the installed entry point the way the shell does."""
    monkeypatch.setattr(sys, "argv", ["immich-moments", *args])
    with pytest.raises(SystemExit) as exit_info:
        main()
    return exit_info.value.code, ""


def test_version_is_the_packaged_one() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == __version__


def test_bare_invocation_shows_the_commands() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code == 2  # click's code for "pick a command"
    for command in ("doctor", "index", "search", "serve"):
        assert command in result.stdout


def test_missing_credentials_name_the_variable(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    code, _ = run("doctor", monkeypatch=monkeypatch)
    assert code == ConfigError.exit_code
    captured = capsys.readouterr()
    assert "IMMICH_URL is not set" in captured.err
    assert "Traceback" not in captured.err


def test_a_bad_phase_is_a_sentence_not_a_stack_trace(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("IMMICH_URL", "http://localhost:2283")
    monkeypatch.setenv("IMMICH_API_KEY", "key")

    code, _ = run("index", "--phase", "speech", monkeypatch=monkeypatch)

    assert code == 1
    captured = capsys.readouterr()
    assert "--phase must be all, visual or audio" in captured.err
    assert "Traceback" not in captured.err


def test_a_server_that_is_not_there_is_reported_as_such(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """Port 1 is reserved and nothing listens on it, so this is a real connection failure."""
    monkeypatch.setenv("IMMICH_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("IMMICH_API_KEY", "key")
    monkeypatch.setenv("IMMICH_MOMENTS_MAX_RETRIES", "1")

    code, _ = run("doctor", monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == ImmichError.exit_code
    assert "cannot reach" in captured.err
    assert "Traceback" not in captured.err


def test_an_unknown_option_in_the_config_file_names_the_file(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path
) -> None:
    config = tmp_path / "immich-moments.toml"
    config.write_text("[immich_moments]\nvisual_wieght = 0.5\n", encoding="utf-8")

    code, _ = run("search", "anything", "-c", str(config), monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == ConfigError.exit_code
    assert "visual_wieght" in captured.err
    assert "Traceback" not in captured.err


def test_a_full_disk_is_the_environment_talking_not_a_bug(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """ENOSPC used to print "unexpected OSError" and ask the user to report it."""
    monkeypatch.setenv("IMMICH_URL", "http://localhost:2283")
    monkeypatch.setenv("IMMICH_API_KEY", "key")

    def full(*_args, **_kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr("immich_moments.cli._clients", full)

    code, _ = run("index", monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == StorageError.exit_code
    assert "No space left on device" in captured.err
    assert "report it" not in captured.err
    assert "Traceback" not in captured.err


class StubClient:
    """Stands in for the Immich and ML clients, neither of which a filter test needs."""

    def __enter__(self) -> StubClient:
        return self

    def __exit__(self, *_exc) -> None:
        return None


def stub_clients(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMMICH_URL", "http://immich.test")
    monkeypatch.setenv("IMMICH_API_KEY", "key")
    monkeypatch.setattr(
        "immich_moments.cli._clients",
        lambda _config: (StubClient(), StubClient(), "ViT-B-32__openai", "buffalo_l"),
    )


def test_searching_for_nothing_at_all_says_what_to_type(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    stub_clients(monkeypatch)

    code, _ = run("search", monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == ConfigError.exit_code
    assert "--person" in captured.err


def test_a_person_nobody_is_indexed_under_is_not_silently_empty(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """A typo would otherwise be indistinguishable from a person who is in no video."""
    stub_clients(monkeypatch)

    code, _ = run("search", "--person", "Ana", monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == ConfigError.exit_code
    assert "Ana" in captured.err
    assert "none yet" in captured.err


def capture_filters(monkeypatch: pytest.MonkeyPatch) -> dict:
    """The kwargs `search` hands the ranker, so a filter test needs no ranking at all."""
    seen: dict = {}

    def capture(_store, _ml, _query, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("immich_moments.cli.run_search", capture)
    return seen


def test_a_person_is_filtered_by_the_spelling_the_index_uses(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path
) -> None:
    """SQLite's lower() is ASCII only, so the typed name is resolved before it reaches SQL."""
    stub_clients(monkeypatch)
    seed_index(tmp_path / "data", ["a garden"], person="Zoë")
    seen = {}

    def capture(_store, _ml, _query, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("immich_moments.cli.run_search", capture)

    code, _ = run("search", "--person", "ZOË", monkeypatch=monkeypatch)

    assert code == 1  # no hits, because the stub returns none
    assert seen["filters"].people == ("Zoë",)


def test_a_date_range_reaches_the_filters(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path) -> None:
    stub_clients(monkeypatch)
    seed_index(tmp_path / "data", ["a garden"])
    seen = capture_filters(monkeypatch)

    code, _ = run("search", "--since", "2019-07-01", "--until", "2019-07-31", monkeypatch=monkeypatch)

    assert code == 1  # no hits, because the stub returns none
    assert (seen["filters"].since, seen["filters"].until) == ("2019-07-01", "2019-07-31")


def test_an_empty_result_repeats_what_was_asked_for(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path
) -> None:
    """A filter that leaves nothing looks like a broken index unless the message says otherwise."""
    stub_clients(monkeypatch)
    seed_index(tmp_path / "data", ["a garden"], person="Zoë")
    capture_filters(monkeypatch)

    code, _ = run("search", "--person", "Zoë", "--since", "2019-07-01", monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == 1
    assert "with Zoë since 2019-07-01" in captured.out


def test_a_day_that_is_not_a_date_is_a_sentence_not_a_stack_trace(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    stub_clients(monkeypatch)

    code, _ = run("search", "candles", "--since", "last summer", monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == ConfigError.exit_code
    assert "2019-07-04" in captured.err
    assert "Traceback" not in captured.err


def test_a_query_and_a_scene_to_rank_against_do_not_mix(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    stub_clients(monkeypatch)

    code, _ = run("search", "candles", "--like", "3", monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == ConfigError.exit_code
    assert "takes no query" in captured.err


def test_json_output_is_machine_readable(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    stub_clients(monkeypatch)
    hit = Hit(
        scene_id=1,
        asset_id="a1",
        original_file_name="birthday.mp4",
        scene_index=2,
        file_created_at="2026-06-01T00:00:00Z",
        start_seconds=10.0,
        end_seconds=20.0,
        label="blowing out candles",
        label_score=0.3,
        thumb_path="a1-2.jpg",
        visual_score=0.4,
        text_score=0.0,
        score=0.9,
        people=["Anna"],
    )
    monkeypatch.setattr("immich_moments.cli.run_search", lambda *_a, **_k: [hit])

    code, _ = run("search", "candles", "--json", monkeypatch=monkeypatch)

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload == [hit.as_dict("http://immich.test")]
    assert payload[0]["timestamp"] == "00:10"
    assert payload[0]["immich_url"] == "http://immich.test/photos/a1"


def seed_index(data_dir, labels, person: str | None = None, album: str | None = None) -> None:
    """A two-scene index on disk, so `relabel` has real vectors to read back."""
    config = Config(
        immich_url="http://immich.test",
        immich_api_key="key",
        ml_url="http://ml.test",
        data_dir=data_dir,
    )
    config.ensure_dirs()
    with Store(config) as store:
        store.check_model("ViT-B-32__openai", 8, reindex=False)
        store.upsert_asset(
            "a1",
            original_file_name="a1.mp4",
            file_created_at="2026-06-01T00:00:00Z",
            updated_at="2026-06-01T00:00:00Z",
            duration_seconds=30.0,
        )
        store.replace_scenes(
            "a1",
            [
                SceneRecord(
                    index,
                    0.0,
                    5.0,
                    vector=unit(index),
                    label=label,
                    label_score=0.3,
                    faces=[FaceRecord("p1", person, 0.2, 0.9, (1, 2, 3, 4))] if person else None,
                )
                for index, label in enumerate(labels)
            ],
            store.vectors(8),
            indexed_at="2026-06-01T00:00:00Z",
        )
        if album:
            store.replace_albums([("a1", "al1", album)])


def stub_labels(monkeypatch: pytest.MonkeyPatch, *, wins: str) -> None:
    """Every scene lands on one label, whatever its vector, so the diff is the thing under test."""
    monkeypatch.setattr(
        "immich_moments.cli.build_label_index",
        lambda *_a, **_k: LabelIndex([wins], np.ones((1, 8), np.float32), min_similarity=-1.0),
    )


def test_relabelling_an_empty_index_says_to_index_first(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    stub_clients(monkeypatch)

    code, _ = run("relabel", monkeypatch=monkeypatch)

    assert code == 1
    assert "index` first" in capsys.readouterr().out


def test_a_dry_run_relabel_writes_nothing(monkeypatch: pytest.MonkeyPatch, capsys, tmp_path) -> None:
    stub_clients(monkeypatch)
    seed_index(tmp_path / "data", ["a garden", "a garden"])
    stub_labels(monkeypatch, wins="a birthday cake")

    code, _ = run("relabel", "--dry-run", monkeypatch=monkeypatch)

    out = capsys.readouterr().out
    assert code == 0
    assert "2 change(s)" in out
    assert "nothing written" in out
    assert labels_on_disk(tmp_path / "data") == ["a garden", "a garden"]


def test_relabelling_rewrites_the_labels_that_moved(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path
) -> None:
    stub_clients(monkeypatch)
    seed_index(tmp_path / "data", ["a garden", "a birthday cake"])
    stub_labels(monkeypatch, wins="a birthday cake")

    code, _ = run("relabel", monkeypatch=monkeypatch)

    out = capsys.readouterr().out
    assert code == 0
    assert "1 change(s): 0 newly labelled, 0 cleared, 1 moved to another label" in out
    assert labels_on_disk(tmp_path / "data") == ["a birthday cake", "a birthday cake"]


def labels_on_disk(data_dir) -> list[str | None]:
    config = Config(
        immich_url="http://immich.test", immich_api_key="key", ml_url="http://ml.test", data_dir=data_dir
    )
    with Store(config) as store:
        return [row["label"] for row in store.labelled_scenes()]


@pytest.mark.parametrize("extra", [("--limit", "3"), ("--since", "2026-01-01")])
def test_prune_with_a_partial_walk_is_refused_before_anything_connects(
    extra: tuple[str, ...], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """Honouring one and ignoring the other would silently drop every video outside the window."""
    code, _ = run("index", "--prune", *extra, monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == ConfigError.exit_code
    assert "whole library" in captured.err


def test_an_album_is_filtered_by_the_spelling_the_index_uses(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path
) -> None:
    stub_clients(monkeypatch)
    seed_index(tmp_path / "data", ["a garden"], album="Föhr 2026")
    seen = capture_filters(monkeypatch)

    code, _ = run("search", "--album", "föhr 2026", monkeypatch=monkeypatch)

    assert code == 1  # no hits, because the stub returns none
    assert seen["filters"].albums == ("Föhr 2026",)
    assert "in Föhr 2026" in capsys.readouterr().out


def test_an_album_the_index_has_never_seen_is_not_silently_empty(
    monkeypatch: pytest.MonkeyPatch, capsys, tmp_path
) -> None:
    """Immich albums holding only photos never reach the index, and neither does a typo."""
    stub_clients(monkeypatch)
    seed_index(tmp_path / "data", ["a garden"], album="Föhr 2026")

    code, _ = run("search", "--album", "Holiday", monkeypatch=monkeypatch)

    captured = capsys.readouterr()
    assert code == ConfigError.exit_code
    assert "Holiday" in captured.err
    assert "Indexed albums: Föhr 2026" in captured.err
