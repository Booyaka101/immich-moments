"""Evaluate a local search server against private, timestamp-labelled JSONL cases.

Run with `python -m immich_moments.benchmark cases.jsonl --output results.json`.
Only search requests are sent, only to loopback; no footage or credentials are needed.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
from pathlib import Path
from urllib.parse import urlparse

import httpx


def _number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def load_cases(path: Path) -> list[dict]:
    cases = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            case = json.loads(line)
            if not isinstance(case, dict):
                raise ValueError("each case must be an object")
            if not isinstance(case.get("id"), str) or not case["id"].strip() or case["id"] in seen:
                raise ValueError("id must be a unique, non-empty string")
            if not isinstance(case.get("query"), str) or not case["query"].strip():
                raise ValueError("query must be a non-empty string")
            expected = case.get("expected")
            if not isinstance(expected, list):
                raise ValueError("expected must be a list; use [] for an absent event")
            for event in expected:
                if not isinstance(event, dict) or not isinstance(event.get("asset_id"), str):
                    raise ValueError("each expected event needs an asset_id")
                start, end = event.get("start_seconds"), event.get("end_seconds")
                if not event["asset_id"] or not _number(start) or not _number(end) or not 0 <= start <= end:
                    raise ValueError("expected events need an asset_id and 0 <= start_seconds <= end_seconds")
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
        seen.add(case["id"])
        cases.append(case)
    if not cases:
        raise ValueError("the benchmark needs at least one case")
    return cases


def local_url(value: str) -> str:
    parsed = urlparse(value)
    _ = parsed.port  # Reject malformed ports before constructing the HTTP client.
    host = parsed.hostname or ""
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if (
        not loopback
        or parsed.scheme not in ("http", "https")
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("--url must be a loopback HTTP(S) origin, such as http://127.0.0.1:8099")
    return value.rstrip("/")


def evaluate(case: dict, response: dict, *, top_k: int, tolerance: float) -> dict:
    """Require the returned moment, not merely an overlapping scene, to find the event."""
    if (
        not isinstance(response, dict)
        or not isinstance(response.get("hits"), list)
        or not isinstance(response.get("nothing_close"), bool)
    ):
        raise ValueError("search response needs hits and a boolean nothing_close")
    hits = response["hits"][:top_k]
    asset_rank = moment_rank = None
    for rank, hit in enumerate(hits, 1):
        if not isinstance(hit, dict):
            raise ValueError("each search hit must be an object")
        moment = hit.get("moment_seconds")
        if not _number(moment) or not isinstance(hit.get("asset_id"), str):
            raise ValueError("search hits need asset_id and numeric moment_seconds; update the server")
        for event in case["expected"]:
            if hit["asset_id"] != event["asset_id"]:
                continue
            if asset_rank is None:
                asset_rank = rank
            if (
                moment_rank is None
                and event["start_seconds"] - tolerance <= moment <= event["end_seconds"] + tolerance
            ):
                moment_rank = rank
    return {
        "id": case["id"],
        "query": case["query"],
        "answerable": bool(case["expected"]),
        "asset_rank": asset_rank,
        "moment_rank": moment_rank,
        "nothing_close": response["nothing_close"],
        "hits": [{"asset_id": hit["asset_id"], "moment_seconds": hit["moment_seconds"]} for hit in hits],
    }


def summarize(rows: list[dict]) -> dict:
    answerable = [row for row in rows if row["answerable"]]
    absent = [row for row in rows if not row["answerable"]]

    def mean(values: list) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "cases": len(rows),
        "answerable_cases": len(answerable),
        "absent_cases": len(absent),
        "asset_hit_at_k": mean([row["asset_rank"] is not None for row in answerable]),
        "moment_hit_at_k": mean([row["moment_rank"] is not None for row in answerable]),
        "moment_mrr_at_k": mean([1 / row["moment_rank"] if row["moment_rank"] else 0 for row in answerable]),
        "absent_warning_rate": mean([row["nothing_close"] for row in absent]),
        "answerable_warning_rate": mean([row["nothing_close"] for row in answerable]),
    }


def run(client: httpx.Client, cases: list[dict], *, top_k: int, tolerance: float, weight: float) -> dict:
    before = client.get("/api/stats")
    before.raise_for_status()
    rows = []
    for case in cases:
        response = client.get("/api/search", params={"q": case["query"], "limit": top_k, "weight": weight})
        response.raise_for_status()
        rows.append(evaluate(case, response.json(), top_k=top_k, tolerance=tolerance))
    after = client.get("/api/stats")
    after.raise_for_status()
    if before.json() != after.json():
        raise ValueError("index statistics changed during the benchmark; stop indexing and retry")
    return {
        "settings": {"top_k": top_k, "tolerance_seconds": tolerance, "visual_weight": weight},
        "index": before.json(),
        "summary": summarize(rows),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8099")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--tolerance", type=float, default=2.0)
    parser.add_argument("--weight", type=float, default=0.65)
    args = parser.parse_args()
    try:
        if not 1 <= args.top_k <= 100:
            raise ValueError("--top-k must be between 1 and 100")
        if not _number(args.tolerance) or args.tolerance < 0:
            raise ValueError("--tolerance must be finite and non-negative")
        if not _number(args.weight) or not 0 <= args.weight <= 1:
            raise ValueError("--weight must be between 0 and 1")
        if args.output.resolve() == args.cases.resolve():
            raise ValueError("--output must not overwrite the cases file")
        cases = load_cases(args.cases)
        with httpx.Client(base_url=local_url(args.url), timeout=120, trust_env=False) as client:
            report = run(client, cases, top_k=args.top_k, tolerance=args.tolerance, weight=args.weight)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (ValueError, OSError, httpx.HTTPError) as exc:
        parser.exit(1, f"Benchmark failed: {exc}\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
