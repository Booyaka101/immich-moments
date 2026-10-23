"""Accuracy metrics use event timestamps, and private queries stay on loopback."""

from __future__ import annotations

import json

import httpx
import pytest

from immich_moments.benchmark import evaluate, load_cases, local_url, main, run, summarize

EVENT = {"asset_id": "birthday", "start_seconds": 10, "end_seconds": 12}
CASE = {"id": "candles", "query": "blowing out candles", "expected": [EVENT]}


def result(*hits, warning=False):
    return {
        "hits": [{"asset_id": asset, "moment_seconds": seconds} for asset, seconds in hits],
        "nothing_close": warning,
    }


def test_the_right_video_at_the_wrong_second_is_not_a_moment_match():
    row = evaluate(CASE, result(("birthday", 1), ("other", 11), ("birthday", 11)), top_k=5, tolerance=0)
    assert row["asset_rank"] == 1
    assert row["moment_rank"] == 3


def test_tolerance_is_explicit_and_the_top_k_limit_is_respected():
    response = result(("birthday", 8), ("birthday", 10))
    assert evaluate(CASE, response, top_k=1, tolerance=0)["moment_rank"] is None
    assert evaluate(CASE, response, top_k=1, tolerance=2)["moment_rank"] == 1


def test_multiple_expected_occurrences_are_accepted():
    case = {**CASE, "expected": [EVENT, {"asset_id": "holiday", "start_seconds": 90, "end_seconds": 95}]}
    assert evaluate(case, result(("holiday", 91)), top_k=5, tolerance=0)["moment_rank"] == 1


def test_absent_queries_measure_the_warning_not_the_existence_of_ranked_results():
    rows = [
        evaluate(CASE, result(("birthday", 11)), top_k=5, tolerance=0),
        evaluate({**CASE, "id": "miss"}, result(("other", 1), warning=True), top_k=5, tolerance=0),
        evaluate(
            {"id": "absent", "query": "a whale", "expected": []},
            result(("birthday", 11), warning=True),
            top_k=5,
            tolerance=0,
        ),
    ]
    assert summarize(rows) == {
        "cases": 3,
        "answerable_cases": 2,
        "absent_cases": 1,
        "asset_hit_at_k": 0.5,
        "moment_hit_at_k": 0.5,
        "moment_mrr_at_k": 0.5,
        "absent_warning_rate": 1.0,
        "answerable_warning_rate": 0.5,
    }


def test_unmeasured_categories_are_null_not_perfect_scores():
    assert summarize([])["moment_hit_at_k"] is None
    assert summarize([evaluate(CASE, result(), top_k=5, tolerance=0)])["absent_warning_rate"] is None


@pytest.mark.parametrize(
    "url",
    [
        "http://192.168.1.2:8099",
        "https://example.com",
        "http://localhost.evil.test",
        "file:///tmp/index",
        "http://key@localhost",
        "http://localhost/api",
        "http://localhost?key=secret",
        "http://localhost#fragment",
    ],
)
def test_nonlocal_or_credentialled_benchmark_urls_are_rejected(url):
    with pytest.raises(ValueError, match="loopback"):
        local_url(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8099", "http://localhost:8099", "http://[::1]:8099"])
def test_loopback_servers_are_accepted(url):
    assert local_url(url + "/") == url


@pytest.mark.parametrize(
    "change",
    [
        {"id": ""},
        {"query": ""},
        {"expected": None},
        {"expected": [{**EVENT, "start_seconds": -1}]},
        {"expected": [{**EVENT, "end_seconds": 1}]},
        {"expected": [{**EVENT, "start_seconds": float("nan")}]},
        {"expected": [{**EVENT, "start_seconds": True}]},
    ],
)
def test_bad_cases_fail_before_any_search(tmp_path, change):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({**CASE, **change}), encoding="utf-8")
    with pytest.raises(ValueError, match=":1:"):
        load_cases(path)


def test_duplicate_case_ids_are_rejected(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text((json.dumps(CASE) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_cases(path)


def test_empty_manifest_is_rejected(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="at least one"):
        load_cases(path)


def test_old_servers_fail_instead_of_guessing_the_moment():
    with pytest.raises(ValueError, match="moment_seconds"):
        evaluate(CASE, {"hits": [{"asset_id": "birthday"}], "nothing_close": False}, top_k=5, tolerance=0)


def test_the_runner_uses_only_read_endpoints_and_records_settings():
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200, json={"scenes": 2} if request.url.path == "/api/stats" else result(("birthday", 11))
        )

    with httpx.Client(base_url="http://127.0.0.1:8099", transport=httpx.MockTransport(respond)) as client:
        report = run(client, [CASE], top_k=3, tolerance=0, weight=0.4)
    assert report["summary"]["moment_hit_at_k"] == 1
    assert report["settings"] == {"top_k": 3, "tolerance_seconds": 0, "visual_weight": 0.4}
    assert [request.method for request in requests] == ["GET"] * 3
    assert requests[1].url.params["q"] == CASE["query"]
    assert requests[1].url.params["weight"] == "0.4"


def test_an_index_changing_during_the_run_is_rejected():
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        return httpx.Response(
            200, json={"scenes": calls} if request.url.path == "/api/stats" else result(("birthday", 11))
        )

    with (
        httpx.Client(base_url="http://127.0.0.1:8099", transport=httpx.MockTransport(respond)) as client,
        pytest.raises(ValueError, match="changed"),
    ):
        run(client, [CASE], top_k=3, tolerance=0, weight=0.4)


@pytest.mark.parametrize("arguments", [["--top-k", "0"], ["--tolerance", "nan"], ["--weight", "2"]])
def test_cli_rejects_invalid_settings_before_reading_cases(monkeypatch, capsys, tmp_path, arguments):
    import sys

    monkeypatch.setattr(
        sys, "argv", ["benchmark", "nonexistent.jsonl", "--output", str(tmp_path / "out.json"), *arguments]
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    assert "Benchmark failed:" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_cli_cannot_overwrite_the_labelled_cases(monkeypatch, capsys, tmp_path):
    import sys

    path = tmp_path / "cases.jsonl"
    content = json.dumps(CASE)
    path.write_text(content, encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["benchmark", str(path), "--output", str(path)])
    with pytest.raises(SystemExit):
        main()
    assert "overwrite" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == content
