"""Configuration precedence: defaults, then TOML, then environment, then CLI flags."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from immich_moments.config import DEFAULT_ML_URL, Config, load_config
from immich_moments.errors import ConfigError


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Nothing in the developer's own shell may leak into these assertions."""
    for name in list(os.environ):
        if name.startswith("IMMICH") or name == "DATA_DIR":
            monkeypatch.delenv(name, raising=False)
    # The data dir is searched for immich-moments.toml, so without this the developer's own
    # config decides whether the defaults below hold.
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)


def test_defaults_without_anything_set() -> None:
    config = load_config()
    assert config.immich_url == ""
    assert config.ml_url == DEFAULT_ML_URL
    assert config.visual_weight == 0.65
    assert config.port == 8099


def test_environment_beats_the_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "immich-moments.toml"
    path.write_text(
        '[immich_moments]\nimmich_url = "http://from-file:2283"\nvisual_weight = 0.9\nport = 9000\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("IMMICH_URL", "http://from-env:2283")
    monkeypatch.setenv("IMMICH_MOMENTS_PORT", "9100")

    config = load_config(path)

    assert config.immich_url == "http://from-env:2283"
    assert config.port == 9100
    assert config.visual_weight == 0.9  # untouched by the environment, so the file still wins


def test_cli_overrides_beat_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMMICH_URL", "http://from-env:2283")
    config = load_config(overrides={"immich_url": "http://from-flag:2283", "port": None})
    assert config.immich_url == "http://from-flag:2283"
    assert config.port == 8099  # a None override means "not given", not "reset to default"


def test_a_config_file_is_found_in_the_working_directory(tmp_path: Path) -> None:
    (tmp_path / "immich-moments.toml").write_text(
        '[immich_moments]\nimmich_url = "http://cwd:2283"\n', encoding="utf-8"
    )
    assert load_config().immich_url == "http://cwd:2283"


def test_a_bare_table_without_the_section_header_also_works(tmp_path: Path) -> None:
    path = tmp_path / "bare.toml"
    path.write_text('immich_url = "http://bare:2283"\n', encoding="utf-8")
    assert load_config(path).immich_url == "http://bare:2283"


def test_an_unknown_option_is_named_not_ignored(tmp_path: Path) -> None:
    path = tmp_path / "typo.toml"
    path.write_text('[immich_moments]\nvisual_wieght = 0.5\nimmich_urll = "x"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="unknown option\\(s\\) immich_urll, visual_wieght"):
        load_config(path)


def test_a_renamed_option_says_what_it_became(tmp_path: Path) -> None:
    """Left as an unknown option it would read as a typo, and left out it would go on doing nothing."""
    path = tmp_path / "old.toml"
    path.write_text("[immich_moments]\nlabel_min_similarity = 0.22\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="label_min_similarity is now label_min_zscore"):
        load_config(path)


def test_the_renamed_environment_variable_is_refused_not_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMMICH_MOMENTS_LABEL_MIN_SIMILARITY", "0.22")
    with pytest.raises(ConfigError, match="IMMICH_MOMENTS_LABEL_MIN_ZSCORE"):
        load_config()


def test_broken_toml_names_the_file(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("[immich_moments\nimmich_url = ", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"broken.toml"):
        load_config(path)


def test_a_missing_explicit_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="config file not found"):
        load_config(tmp_path / "nope.toml")


def test_a_missing_env_pointed_config_file_is_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMMICH_MOMENTS_CONFIG", str(tmp_path / "nope.toml"))
    with pytest.raises(ConfigError, match="IMMICH_MOMENTS_CONFIG"):
        load_config()


def test_a_non_numeric_value_is_rejected_with_the_option_name(tmp_path: Path) -> None:
    path = tmp_path / "bad.toml"
    path.write_text('[immich_moments]\nport = "eight thousand"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="port: expected int"):
        load_config(path)


def test_visual_weight_must_be_a_fraction(tmp_path: Path) -> None:
    path = tmp_path / "weight.toml"
    path.write_text("[immich_moments]\nvisual_weight = 1.5\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="visual_weight must be between 0 and 1"):
        load_config(path)


def test_data_dir_expands_and_drives_every_derived_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATA_DIR", "~/moments-data")
    config = load_config()
    assert config.data_dir == Path.home() / "moments-data"
    assert config.db_path == config.data_dir / "moments.sqlite3"
    assert config.vectors_path == config.data_dir / "vectors.f32"
    assert config.thumbs_dir == config.data_dir / "thumbs"
    assert config.audio_dir == config.data_dir / "audio"


def test_an_empty_environment_variable_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMMICH_URL", "")
    assert load_config().immich_url == ""


def test_api_base_tolerates_a_trailing_slash() -> None:
    assert Config(immich_url="http://host:2283/").api_base == "http://host:2283/api"


def test_the_browser_link_falls_back_to_the_api_url() -> None:
    assert Config(immich_url="http://host:2283/").browser_url == "http://host:2283"


def test_a_public_url_overrides_the_api_url_for_links() -> None:
    """Beside Immich in compose the API is a service name and the link has to be the real one."""
    config = Config(immich_url="http://immich-server:2283", immich_public_url="http://nas:2283/")
    assert config.browser_url == "http://nas:2283"
    assert config.api_base == "http://immich-server:2283/api"


@pytest.mark.parametrize(
    ("url", "internal"),
    [
        ("http://immich-server:2283", True),
        ("http://immich_server:2283", True),
        ("http://localhost:2283", False),
        ("http://192.168.1.4:2283", False),
        ("http://photos.example.com", False),
        ("", False),
    ],
)
def test_a_link_only_the_container_network_can_follow_is_recognised(url: str, internal: bool) -> None:
    """A dead "Open in Immich" is the whole compose setup the README recommends."""
    assert Config(immich_url=url).browser_url_is_internal is internal


def test_missing_credentials_say_what_to_set() -> None:
    with pytest.raises(ConfigError, match="IMMICH_URL is not set"):
        Config().require_credentials()
    with pytest.raises(ConfigError, match="IMMICH_API_KEY is not set"):
        Config(immich_url="http://host:2283").require_credentials()


def test_ensure_dirs_reports_the_directory_it_could_not_create(tmp_path: Path) -> None:
    blocker = tmp_path / "data"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ConfigError, match="cannot create"):
        Config(data_dir=blocker).ensure_dirs()
