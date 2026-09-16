"""Build a synthetic index of a given size and time real searches against it.

Everything goes in through the real `Store` write path, so the schema, the FTS index and the
flat vector file are what a real index would have. Only the query embedding is stubbed, because
that is Immich's work rather than ours and a network call would hide our own cost behind it.

    python tools/scale_bench.py 25000 100000

Numbers from a run of this live in PROGRESS.md.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import shutil
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from immich_moments.config import Config
from immich_moments.search import Filters, search, similar
from immich_moments.store import SceneRecord, Store, TranscriptRecord

DIM = 512
SCENES_PER_ASSET = 14
WORDS = """beach holiday cake candles birthday party garden snow christmas tree dog running
kitchen dinner guitar singing wedding speech boat harbour mountain hiking rain city
night fireworks pool swimming grandma laughing baby first steps school concert""".split()  # noqa: SIM905


class StubML:
    """A deterministic unit-norm text embedding, so scoring does real work on fake vectors."""

    def embed_text(self, text: str) -> np.ndarray:
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        vec = rng.standard_normal(DIM).astype(np.float32)
        return vec / np.linalg.norm(vec)


class _WindowsCounters(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.c_uint32),
        ("PageFaultCount", ctypes.c_uint32),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def rss_mb(*, peak: bool = False) -> float:
    """Resident set size in MB, without adding a dependency the project does not already have."""
    if sys.platform != "win32":
        wanted = "VmHWM:" if peak else "VmRSS:"
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(wanted):
                return int(line.split()[1]) / 1024  # the kernel reports kB
        raise OSError(f"{wanted} missing from /proc/self/status")

    kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    # The pseudo-handle is -1 as a pointer, so it truncates to 32 bits without this.
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    counters = _WindowsCounters()
    counters.cb = ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(
        ctypes.c_void_p(kernel32.GetCurrentProcess()), ctypes.byref(counters), counters.cb
    ):
        raise OSError("GetProcessMemoryInfo failed")
    return (counters.PeakWorkingSetSize if peak else counters.WorkingSetSize) / 1048576


def build(data_dir: Path, scenes_total: int) -> Store:
    store = Store(Config(immich_url="http://example", immich_api_key="k", data_dir=data_dir))
    store.set_state("vector_dim", str(DIM))
    vectors = store.vectors(DIM)
    rng = np.random.default_rng(7)

    started = time.perf_counter()
    for asset_index in range(scenes_total // SCENES_PER_ASSET):
        asset_id = f"asset-{asset_index:06d}"
        store.upsert_asset(
            asset_id,
            original_file_name=f"clip {asset_index:06d}.mp4",
            file_created_at=f"{2015 + asset_index % 11}-0{1 + asset_index % 9}-15T10:00:00.000Z",
            updated_at="2026-01-01T00:00:00.000Z",
            duration_seconds=float(SCENES_PER_ASSET * 8),
        )
        batch = rng.standard_normal((SCENES_PER_ASSET, DIM)).astype(np.float32)
        batch /= np.linalg.norm(batch, axis=1, keepdims=True)
        store.replace_scenes(
            asset_id,
            [
                SceneRecord(
                    idx=i,
                    start_seconds=float(i * 8),
                    end_seconds=float(i * 8 + 8),
                    vector=batch[i],
                    label=WORDS[(asset_index + i) % len(WORDS)],
                    label_score=0.3,
                    thumb_path=f"{asset_id}/{i}.jpg",
                )
                for i in range(SCENES_PER_ASSET)
            ],
            vectors,
            indexed_at="2026-01-01T00:00:00.000Z",
        )
        # Every third scene carries speech, which is roughly what the live 16-video run produced.
        store.replace_transcript(
            asset_id,
            [
                TranscriptRecord(
                    start_seconds=float(i * 8 + 1),
                    end_seconds=float(i * 8 + 7),
                    text=" ".join(WORDS[(asset_index * 3 + i * 5 + k) % len(WORDS)] for k in range(12)),
                )
                for i in range(0, SCENES_PER_ASSET, 3)
            ],
            indexed_at="2026-01-01T00:00:00.000Z",
            has_audio=True,
        )
        if asset_index and asset_index % 500 == 0:
            done = (asset_index + 1) * SCENES_PER_ASSET
            rate = done / (time.perf_counter() - started)
            print(f"  built {done}/{scenes_total}  {rate:.0f}/s", flush=True)
    return store


def report(name: str, timings: list[float]) -> None:
    ordered = sorted(timings)
    p95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    print(f"  {name:<26} median {statistics.median(ordered):7.1f} ms   p95 {p95:7.1f} ms")


def measure(store: Store, ml: StubML, queries: list[str], weight: float, filters: Filters) -> list[float]:
    timings = []
    for query in queries:
        start = time.perf_counter()
        search(store, ml, query, limit=20, visual_weight=weight, filters=filters)
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def run(data_dir: Path, size: int) -> None:
    print(f"\n=== {size:,} scenes, about {size // SCENES_PER_ASSET:,} videos ===", flush=True)
    started = time.perf_counter()
    store = build(data_dir, size)
    print(f"  build {time.perf_counter() - started:.1f}s")
    print(
        f"  on disk: sqlite {store.config.db_path.stat().st_size / 1048576:.1f} MB, "
        f"vectors {store.config.vectors_path.stat().st_size / 1048576:.1f} MB"
    )

    ml = StubML()
    queries = [" ".join(WORDS[i % len(WORDS) : i % len(WORDS) + 2]) for i in range(12)]
    gc.collect()

    cold = time.perf_counter()
    search(store, ml, "beach holiday", limit=20)
    print(f"  {'cold first query':<26} {(time.perf_counter() - cold) * 1000:7.1f} ms")

    report("blended (default 0.65)", measure(store, ml, queries, 0.65, Filters()))
    report("visual only", measure(store, ml, queries, 1.0, Filters()))
    report("text only", measure(store, ml, queries, 0.0, Filters()))
    # A date range that keeps roughly a tenth of the library. A filter matching nothing would
    # return before any vector work and measure the wrong thing entirely.
    narrowed = Filters(since="2018-01-01", until="2018-12-31")
    kept = store.db.execute(
        "SELECT COUNT(*) AS n FROM scenes WHERE asset_id IN "
        "(SELECT id FROM assets WHERE substr(file_created_at, 1, 10) BETWEEN ? AND ?)",
        ("2018-01-01", "2018-12-31"),
    ).fetchone()["n"]
    report(f"blended, {kept / size:.0%} date filter", measure(store, ml, queries, 0.65, narrowed))

    scene_id = int(store.db.execute("SELECT id FROM scenes LIMIT 1").fetchone()["id"])
    nearest = []
    for _ in range(6):
        start = time.perf_counter()
        similar(store, scene_id, limit=20)
        nearest.append((time.perf_counter() - start) * 1000)
    report("more like this", nearest)
    print(f"  RSS {rss_mb():.0f} MB now, {rss_mb(peak=True):.0f} MB peak")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sizes", nargs="*", type=int, default=[5_000, 25_000, 100_000])
    parser.add_argument("--keep", action="store_true", help="leave the synthetic stores behind")
    args = parser.parse_args()

    root = Path(__file__).parent / ".scale-stores"
    for size in args.sizes:
        data_dir = root / str(size)
        shutil.rmtree(data_dir, ignore_errors=True)
        try:
            run(data_dir, size)
        finally:
            if not args.keep:
                gc.collect()
                shutil.rmtree(data_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
