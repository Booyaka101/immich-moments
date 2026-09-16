"""The command line itself: version, help, and what a user sees when something is wrong.

Every failure here must reach the terminal as a sentence, not a traceback, so these drive
`main()` rather than the Typer app where the error handling lives.
"""

from __future__ import annotations

import json
import sys

import pytest
from typer.testing import CliRunner

from immich_moments import __version__
from immich_moments.cli import app, main
from immich_moments.errors import ConfigError, ImmichError, StorageError
from immich_moments.search import Hit

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
    assert "nobody yet" in captured.err


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
