"""Command line: doctor, index, search, serve."""

from __future__ import annotations

import contextlib
import logging
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import __version__
from .config import Config, load_config
from .errors import MomentsError
from .faces import pad_thumbnail
from .immich import ImmichClient
from .indexer import run_index
from .ml import MLClient
from .search import search as run_search
from .store import Store
from .writeback import write_back as apply_write_back

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    # Without this click never runs the group callback, so `--version` alone would only ever
    # answer "Missing command".
    invoke_without_command=True,
    pretty_exceptions_enable=False,
    help="Scene, face and speech search for the videos already in your Immich library.",
)
# Transcripts and filenames carry whatever people say and type, which is more than a legacy
# Windows code page can encode; without this a stray character aborts a finished write-back.
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

console = Console()
err = Console(stderr=True)

ConfigOption = Annotated[
    Path | None, typer.Option("--config", "-c", help="Path to an immich-moments.toml file.")
]
VerboseOption = Annotated[bool, typer.Option("--verbose", "-v", help="Show per-step logging.")]


@app.callback()
def _root(
    version: Annotated[
        bool, typer.Option("--version", help="Print the version and exit.", is_eager=True)
    ] = False,
) -> None:
    if version:
        console.print(__version__)
        raise typer.Exit


def _setup(config_path: Path | None, verbose: bool, **overrides) -> Config:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    config = load_config(config_path, overrides)
    config.ensure_dirs()
    return config


def _clients(config: Config) -> tuple[ImmichClient, MLClient, str, str]:
    immich = ImmichClient(config)
    clip_model, face_model = immich.model_names()
    return immich, MLClient(config, clip_model, face_model), clip_model, face_model


@app.command()
def doctor(
    config_path: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Check the Immich API key, report the configured models, and round-trip one real frame."""
    config = _setup(config_path, verbose)
    table = Table(show_header=False, box=None, pad_edge=False)
    ok = True

    immich = ImmichClient(config)
    with immich:
        version = immich.server_version()
        table.add_row("[green]OK[/]", "Immich API", f"{config.immich_url} ({version})")

        clip_model, face_model = immich.model_names()
        table.add_row("[green]OK[/]", "CLIP model", clip_model)
        table.add_row("[green]OK[/]", "Face model", face_model)
        min_score, max_distance = immich.face_thresholds()
        table.add_row("", "Face thresholds", f"minScore {min_score:g}, maxDistance {max_distance:g}")

        videos = 0
        first_video: dict | None = None
        for asset in immich.iter_videos():
            videos += 1
            if first_video is None:
                first_video = asset
        table.add_row("[green]OK[/]", "Videos visible", str(videos))

        people = [p for p in immich.people() if (p.get("name") or "").strip()]
        table.add_row("[green]OK[/]", "Named people", str(len(people)))

        ml = MLClient(config, clip_model, face_model)
        with ml:
            ml.ping()
            table.add_row("[green]OK[/]", "ML container", f"{config.ml_url} answered /ping")

            text_vector = ml.embed_text("a photo of a birthday cake")
            table.add_row("[green]OK[/]", "CLIP text", f"{text_vector.shape[0]}-dim vector")

            probe, source, expect_face = _probe_image(immich, people, first_video)
            if probe is None:
                ok = False
                table.add_row(
                    "[red]!![/]",
                    "CLIP image",
                    "no person or video thumbnail could be fetched, so /predict was never "
                    "exercised on a real image",
                )
            else:
                image_vector = ml.embed_image(probe)
                table.add_row(
                    "[green]OK[/]",
                    "CLIP image",
                    f"round-tripped {source}, {image_vector.shape[0]}-dim, cosine "
                    f"{float(image_vector @ text_vector):+.3f} against the probe text",
                )
                faces = ml.detect_faces(probe)
                if expect_face and not faces:
                    ok = False
                table.add_row(
                    "[green]OK[/]" if faces else "[yellow]--[/]",
                    "Face pipeline",
                    f"{len(faces)} face(s) in {source}",
                )

        with Store(config) as store:
            stored_model = store.get_state("clip_model")
            counts = store.counts()
            if stored_model and stored_model != clip_model:
                ok = False
                table.add_row(
                    "[red]!![/]",
                    "Index",
                    f"built with {stored_model}, server now says {clip_model}. "
                    "Run `immich-moments index --reindex`.",
                )
            else:
                table.add_row(
                    "[green]OK[/]",
                    "Index",
                    f"{counts['assets']} assets, {counts['scenes']} scenes, "
                    f"{counts['segments']} transcript segments at {config.db_path}",
                )

    console.print(table)
    if not ok:
        raise typer.Exit(1)


@app.command()
def index(
    since: Annotated[
        str, typer.Option("--since", help="'auto' to continue from the last run, 'all', or an ISO date.")
    ] = "auto",
    limit: Annotated[int | None, typer.Option("--limit", help="Stop after this many assets.")] = None,
    phase: Annotated[str, typer.Option("--phase", help="all, visual or audio.")] = "all",
    reindex: Annotated[bool, typer.Option("--reindex", help="Throw the index away and rebuild it.")] = False,
    labels: Annotated[
        Path | None, typer.Option("--labels", help="Custom scene label vocabulary, one per line.")
    ] = None,
    write_back: Annotated[
        bool, typer.Option("--write-back", help="Write tags and descriptions back into Immich.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="With --write-back, print the mutations and send none.")
    ] = False,
    config_path: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Index every video: scenes, CLIP vectors, faces and speech, checkpointed per asset."""
    config = _setup(config_path, verbose)
    phases = _phases(phase)

    immich, ml, clip_model, face_model = _clients(config)
    with immich, ml, Store(config) as store:
        dimension = ml.embed_text("a photo").shape[0]
        store.check_model(clip_model, dimension, reindex=reindex)

        console.print(
            f"[dim]clip={clip_model} ({dimension}-dim)  faces={face_model}  data={config.data_dir}[/]"
        )
        report = run_index(
            config,
            store,
            immich,
            ml,
            since=_since(store, since),
            limit=limit,
            phases=phases,
            labels_path=labels,
            reindex=reindex,
            progress=lambda message: console.print(f"[dim]{escape(message)}[/]"),
        )
        _print_report(report)

        if write_back:
            _print_writeback(apply_write_back(store, immich, store.indexed_asset_ids(), dry_run=dry_run))


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What you are looking for.")],
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many scenes to show.")] = 10,
    weight: Annotated[
        float | None,
        typer.Option("--weight", "-w", help="Visual weight, 0 = speech only, 1 = vision only."),
    ] = None,
    asset: Annotated[str | None, typer.Option("--asset", help="Restrict to one asset id.")] = None,
    config_path: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Search the index and print the matching scenes with timestamps."""
    config = _setup(config_path, verbose)
    immich, ml, _clip, _face = _clients(config)
    with immich, ml, Store(config) as store:
        hits = run_search(
            store,
            ml,
            query,
            limit=limit,
            visual_weight=config.visual_weight if weight is None else weight,
            asset_id=asset,
        )
    if not hits:
        console.print("[yellow]No scenes matched.[/] Index some videos first, or try fewer words.")
        raise typer.Exit(1)

    table = Table(title=f"{len(hits)} scene(s) for {query!r}", header_style="bold")
    table.add_column("score", justify="right")
    table.add_column("at", justify="right")
    # The filename is how you find the video again, so it keeps its width and "said" gives way.
    table.add_column("video", min_width=18, overflow="fold")
    table.add_column("scene", min_width=12)
    table.add_column("people")
    table.add_column("said", max_width=30)
    for hit in hits:
        table.add_row(
            f"{hit.score:.3f}",
            hit.timestamp,
            escape(hit.original_file_name),
            escape(hit.label or "-"),
            escape(", ".join(hit.people) or "-"),
            escape(_shorten(hit.transcript, 48)),
        )
    console.print(table)
    console.print(f"[dim]{hits[0].immich_url(config.immich_url)}[/]")


@app.command()
def serve(
    host: Annotated[str | None, typer.Option("--host")] = None,
    port: Annotated[int | None, typer.Option("--port")] = None,
    config_path: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Start the web UI."""
    import uvicorn

    config = _setup(config_path, verbose, host=host, port=port)
    config.require_credentials()
    from .web.app import create_app

    with ImmichClient(config) as immich:
        immich.model_names()  # fail here with a readable message rather than inside uvicorn

    console.print(f"immich-moments on http://{config.host}:{config.port}")
    uvicorn.run(create_app(config), host=config.host, port=config.port, log_level="info")


# ---- helpers -------------------------------------------------------------


def _probe_image(
    immich: ImmichClient, people: list[dict], first_video: dict | None
) -> tuple[bytes | None, str, bool]:
    """A real image from the library for doctor to push through /predict.

    A named person's thumbnail proves the whole face path, so a face is expected there. Falling
    back to a video still still exercises CLIP, but it may legitimately contain no face.
    """
    for person in people:
        thumbnail = immich.person_thumbnail(person["id"])
        if thumbnail:
            return pad_thumbnail(thumbnail), f"{person.get('name')}'s face thumbnail", True
    if first_video is not None:
        thumbnail = immich.asset_thumbnail(first_video["id"])
        if thumbnail:
            return thumbnail, f"the thumbnail of {first_video.get('originalFileName')}", False
    return None, "", False


def _phases(value: str) -> tuple[str, ...]:
    choice = value.strip().lower()
    if choice == "all":
        return ("visual", "audio")
    if choice in ("visual", "audio"):
        return (choice,)
    raise MomentsError(f"--phase must be all, visual or audio, not {value!r}")


def _since(store: Store, value: str) -> str | None:
    choice = value.strip().lower()
    if choice == "all":
        return None
    if choice == "auto":
        return store.get_state("last_discovery")
    return value


def _shorten(text: str, width: int) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= width else collapsed[: width - 1] + "…"


def _print_report(report) -> None:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_row("discovered", str(report.discovered))
    table.add_row("people references", str(report.people_refs))
    table.add_row("videos indexed (visual)", str(report.visual_indexed))
    table.add_row("videos indexed (speech)", str(report.audio_indexed))
    table.add_row("scenes", str(report.scenes))
    table.add_row("transcript segments", str(report.segments))
    if report.no_audio_track:
        table.add_row("no audio track", str(report.no_audio_track))
    if report.no_speech:
        table.add_row("no speech found", str(report.no_speech))
    if report.unavailable:
        table.add_row("[yellow]unavailable[/]", ", ".join(report.unavailable))
    for name, message in report.failed:
        table.add_row("[red]failed[/]", f"{name}: {message}")
    console.print(table)

    timings = Table(title="timings", header_style="bold")
    timings.add_column("phase")
    timings.add_column("assets", justify="right")
    timings.add_column("seconds", justify="right")
    timings.add_column("per asset", justify="right")
    for entry in report.timings:
        per = f"{entry.seconds / entry.assets:.1f}s" if entry.assets else "-"
        timings.add_row(entry.name, str(entry.assets), f"{entry.seconds:.1f}", per)
    console.print(timings)


def _print_writeback(result) -> None:
    heading = "planned mutations (nothing was sent)" if result.dry_run else "write-back"
    console.print(f"\n[bold]{heading}[/]")
    if not result.planned:
        console.print(
            "[dim]nothing to change: Immich already has everything this index knows.[/]"
            if result.considered
            else "[dim]nothing to write: index some videos first.[/]"
        )
        return
    for plan in result.planned:
        console.print(f"\n[bold]{plan.original_file_name}[/] [dim]{plan.asset_id}[/]")
        for tag in plan.tags:
            console.print(f"  + tag {escape(tag)}")
        if plan.description_changed:
            console.print("  ~ description")
            for line in plan.description.splitlines():
                console.print(f"    [dim]{escape(line)}[/]")
    if result.dry_run:
        console.print(
            f"\n[dim]{len(result.planned)} asset(s) would change. Re-run without --dry-run to apply.[/]"
        )
    else:
        console.print(
            f"\n{result.tags_created} tag(s) upserted, {result.assets_tagged} asset-tag link(s), "
            f"{result.descriptions_written} description(s) written. "
            f'Search Immich for "immich-moments" to see them.'
        )


def main() -> None:
    try:
        app()
    except MomentsError as exc:
        err.print(f"[red]error:[/] {exc}")
        raise SystemExit(exc.exit_code or 1) from exc
    except KeyboardInterrupt:
        err.print("[yellow]interrupted.[/] Progress is checkpointed; re-run to continue.")
        raise SystemExit(130) from None
    except Exception as exc:
        err.print(f"[red]unexpected {exc.__class__.__name__}:[/] {exc}")
        err.print("[dim]Re-run with --verbose for the traceback, and please report it.[/]")
        if logging.getLogger().isEnabledFor(logging.INFO):
            raise
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
