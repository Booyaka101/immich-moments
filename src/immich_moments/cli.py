"""Command line: doctor, index, search, relabel, serve."""

from __future__ import annotations

import contextlib
import errno
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from . import __version__
from .config import Config, load_config
from .errors import ConfigError, MomentsError, StorageError
from .faces import pad_thumbnail
from .immich import ImmichClient
from .indexer import run_index
from .labels import build_label_index
from .ml import MLClient
from .search import Filters, Hit, date_range, resolve_albums, resolve_people
from .search import search as run_search
from .search import similar as run_similar
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
# What LabelIndex returns for one scene: the winning label and how far it stands above the
# rest of the vocabulary, or nothing.
Pick = tuple[str, float] | None


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
        if config.browser_url_is_internal:
            table.add_row(
                "[yellow]--[/]",
                "Immich links",
                f"{config.browser_url} resolves inside the container network, so "
                "'Open in Immich' will not open in a browser. Set IMMICH_PUBLIC_URL to the "
                "address you use, e.g. http://localhost:2283.",
            )

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
            problems = store.problem_assets()
            if problems:
                table.add_row(
                    "[yellow]--[/]",
                    "Not indexed",
                    "\n".join(f"{row['original_file_name']}: {row['error']}" for row in problems),
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
    prune: Annotated[
        bool,
        typer.Option("--prune", help="Walk the whole library and drop videos Immich no longer has."),
    ] = False,
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
    if prune and (limit is not None or since.strip().lower() not in ("auto", "all")):
        raise ConfigError(
            "--prune has to walk the whole library, so it cannot take --limit or a --since date."
        )

    immich, ml, clip_model, face_model = _clients(config)
    with immich, ml, Store(config) as store:
        dimension = ml.embed_text("a photo").shape[0]
        store.check_model(clip_model, dimension, reindex=reindex)

        console.print(
            f"[dim]clip={clip_model} ({dimension}-dim)  faces={face_model}  data={config.data_dir}[/]"
        )
        with _index_progress() as report_progress:
            report = run_index(
                config,
                store,
                immich,
                ml,
                since=None if prune else _since(store, since),
                limit=limit,
                phases=phases,
                labels_path=labels,
                reindex=reindex,
                prune=prune,
                progress=report_progress,
            )
        _print_report(report)

        if write_back:
            _print_writeback(apply_write_back(store, immich, store.indexed_asset_ids(), dry_run=dry_run))


@app.command()
def relabel(
    labels: Annotated[
        Path | None, typer.Option("--labels", help="Custom scene label vocabulary, one per line.")
    ] = None,
    min_zscore: Annotated[
        float | None,
        typer.Option(
            "--min-zscore",
            help="How far above the rest of the vocabulary the winning label has to sit, in "
            "standard deviations. Below it a scene gets no label.",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print what would change and write nothing.")
    ] = False,
    config_path: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Re-label indexed scenes from a vocabulary, without fetching the videos again.

    Labels are read off the scene vectors that are already in the index, so trying a different
    vocabulary costs one embedding pass over the word list, not another pass over your library.
    """
    config = _setup(config_path, verbose)
    immich, ml, clip_model, _face = _clients(config)
    with immich, ml, Store(config) as store:
        store.assert_model(clip_model)
        scenes = store.labelled_scenes()
        if not scenes:
            console.print("[yellow]Nothing to re-label.[/] Run `immich-moments index` first.")
            raise typer.Exit(1)

        index = build_label_index(
            ml,
            config.data_dir,
            labels_path=labels,
            min_zscore=config.label_min_zscore if min_zscore is None else min_zscore,
        )
        vectors = store.vectors().read_all()
        usable = [scene for scene in scenes if scene["vector_row"] < vectors.shape[0]]
        picks = index.best_many(vectors[[scene["vector_row"] for scene in usable]])

        changes = [
            (scene["id"], scene["label"], pick)
            for scene, pick in zip(usable, picks, strict=True)
            if (pick[0] if pick else None) != scene["label"]
        ]
        _print_relabel(len(usable), picks, changes)
        if dry_run:
            console.print("[dim]--dry-run: nothing written.[/]")
            return
        store.set_labels(
            [
                (scene_id, pick[0] if pick else None, pick[1] if pick else None)
                for scene_id, _old, pick in changes
            ]
        )
        if changes:
            console.print(
                "[dim]Tags in Immich are not rewritten; write-back never removes a tag it added.[/]"
            )


def _print_relabel(total: int, picks: list[Pick], changes: list[tuple[int, str | None, Pick]]) -> None:
    labelled = sum(1 for pick in picks if pick)
    gained = sum(1 for _id, old, pick in changes if old is None)
    lost = sum(1 for _id, old, pick in changes if pick is None)
    console.print(
        f"{total} scene(s) with vectors, {labelled} labelled, {total - labelled} below the threshold"
    )
    console.print(
        f"{len(changes)} change(s): {gained} newly labelled, {lost} cleared, "
        f"{len(changes) - gained - lost} moved to another label"
    )
    if not changes:
        return
    table = Table(title="first changes", header_style="bold")
    table.add_column("was")
    table.add_column("now")
    table.add_column("z", justify="right")
    for _id, old, pick in changes[:10]:
        table.add_row(escape(old or "-"), escape(pick[0] if pick else "-"), f"{pick[1]:.3f}" if pick else "-")
    console.print(table)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="What you are looking for.")] = "",
    limit: Annotated[int, typer.Option("--limit", "-n", help="How many scenes to show.")] = 10,
    weight: Annotated[
        float | None,
        typer.Option("--weight", "-w", help="Visual weight, 0 = speech only, 1 = vision only."),
    ] = None,
    asset: Annotated[str | None, typer.Option("--asset", help="Restrict to one asset id.")] = None,
    person: Annotated[
        list[str] | None,
        typer.Option("--person", "-p", help="Only scenes this person appears in. Repeatable."),
    ] = None,
    album: Annotated[
        list[str] | None,
        typer.Option("--album", "-a", help="Only videos in this Immich album. Repeatable."),
    ] = None,
    like: Annotated[
        int | None,
        typer.Option("--like", help="Scene id to find more of, instead of a query."),
    ] = None,
    since: Annotated[
        str | None,
        typer.Option("--since", help="Only videos taken on or after this day, as YYYY-MM-DD."),
    ] = None,
    until: Annotated[
        str | None,
        typer.Option("--until", help="Only videos taken on or before this day, as YYYY-MM-DD."),
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the hits as JSON.")] = False,
    config_path: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Search the index and print the matching scenes with timestamps.

    With --person, --album, --asset or a date range and no query, it lists those scenes
    newest video first. With --like it ranks by picture alone against the scene you name.
    """
    config = _setup(config_path, verbose)
    wanted = [name.strip() for name in (person or []) if name.strip()]
    in_albums = [name.strip() for name in (album or []) if name.strip()]
    first, last = date_range(since, until)
    nothing_asked = not any((query.strip(), wanted, in_albums, asset, first, last))
    if like is not None and query.strip():
        raise ConfigError("--like ranks against one scene's picture, so it takes no query.")
    if like is None and nothing_asked:
        raise ConfigError("give me something to search for, or --person NAME to browse.")

    immich, ml, clip_model, _face = _clients(config)
    reference = None
    with immich, ml, Store(config) as store:
        store.assert_model(clip_model)
        people = resolve_people(store, wanted)
        albums = resolve_albums(store, in_albums)
        filters = Filters(
            asset_id=asset,
            people=tuple(people),
            albums=tuple(albums),
            since=first,
            until=last,
        )
        if like is not None:
            reference, hits = run_similar(store, like, limit=limit, filters=filters)
        else:
            hits = run_search(
                store,
                ml,
                query,
                limit=limit,
                visual_weight=config.visual_weight if weight is None else weight,
                filters=filters,
            )
    if as_json:
        console.print_json(json.dumps([hit.as_dict(config.browser_url) for hit in hits]))
        raise typer.Exit(0 if hits else 1)
    if not hits:
        what = _describe(query, filters, reference)
        console.print(
            f"[yellow]No scenes matched{' ' + what if what else ''}.[/] "
            "Try fewer words, a wider range, or index more videos."
        )
        raise typer.Exit(1)

    table = Table(title=f"{len(hits)} scene(s) {_describe(query, filters, reference)}", header_style="bold")
    # Nothing is scored when there is no query, so the column shows the date that ordered them.
    table.add_column(_score_column(query, reference), justify="right")
    table.add_column("at", justify="right")
    # The filename is how you find the video again, so it keeps its width and "said" gives way.
    table.add_column("video", min_width=18, overflow="fold")
    table.add_column("scene", min_width=12)
    table.add_column("people")
    table.add_column("said", max_width=30)
    for hit in hits:
        table.add_row(
            (hit.file_created_at or "")[:10] if not query.strip() and not reference else f"{hit.score:.3f}",
            hit.timestamp,
            escape(hit.original_file_name),
            escape(hit.label or "-"),
            escape(", ".join(hit.people) or "-"),
            escape(_shorten(hit.transcript, 48)),
        )
    console.print(table)
    console.print(f"[dim]{hits[0].immich_url(config.browser_url)}[/]")


def _describe(query: str, filters: Filters, reference: Hit | None = None) -> str:
    """The table title: what was asked for, in the order the filters were applied."""
    parts = []
    if query.strip():
        parts.append(f"for {query.strip()!r}")
    if reference is not None:
        what = f"{reference.label!r}" if reference.label else f"scene {reference.scene_index}"
        parts.append(f"like {what} in {reference.original_file_name}")
    if filters.people:
        parts.append("with " + " and ".join(filters.people))
    if filters.albums:
        parts.append("in " + " and ".join(filters.albums))
    if filters.since:
        parts.append(f"since {filters.since}")
    if filters.until:
        parts.append(f"until {filters.until}")
    return " ".join(parts)


def _score_column(query: str, reference: Hit | None) -> str:
    """A cosine against one scene, a blended score for a query, and a date when neither ranks."""
    if reference is not None:
        return "cosine"
    return "score" if query.strip() else "date"


@app.command()
def serve(
    host: Annotated[
        str | None, typer.Option("--host", help="Address to bind. Defaults to 127.0.0.1.")
    ] = None,
    port: Annotated[int | None, typer.Option("--port", help="Port to listen on. Defaults to 8099.")] = None,
    config_path: ConfigOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Serve the search page: one box, thumbnails, filters, and a link into Immich per scene.

    It listens on localhost and has no authentication of its own, so put it behind something
    that does before you give it an address the rest of the network can reach.
    """
    import uvicorn

    config = _setup(config_path, verbose, host=host, port=port)
    config.require_credentials()
    from .web.app import create_app

    # Fail here with a readable message rather than inside uvicorn.
    with ImmichClient(config) as immich, Store(config) as store:
        clip_model, _face = immich.model_names()
        store.assert_model(clip_model)

    url = f"http://{config.host}:{config.port}"
    # A link, so terminals that support OSC 8 make it clickable and the rest print the same text.
    console.print(f"immich-moments on [link={url}]{url}[/]")
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
    if report.albums:
        table.add_row("albums", str(report.albums))
    if report.faces_rematched:
        table.add_row("faces renamed or re-matched", str(report.faces_rematched))
    table.add_row("videos indexed (visual)", str(report.visual_indexed))
    table.add_row("videos indexed (speech)", str(report.audio_indexed))
    table.add_row("scenes", str(report.scenes))
    table.add_row("transcript segments", str(report.segments))
    if report.no_audio_track:
        table.add_row("no audio track", str(report.no_audio_track))
    if report.no_speech:
        table.add_row("no speech found", str(report.no_speech))
    if report.pruned:
        table.add_row("dropped, gone from Immich", ", ".join(report.pruned))
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


@contextlib.contextmanager
def _index_progress():
    """A live bar per phase on a terminal, one plain line per video when the output is piped.

    Indexing a library is a long wait, so a terminal gets a bar with a remaining-time estimate.
    A log file or a pipe gets the plain lines, which stay readable and diff cleanly.
    """
    if not console.is_terminal:
        yield lambda phase, position, total, name: console.print(
            f"[dim]{phase} {position}/{total}  {escape(name)}[/]"
        )
        return

    bar = Progress(
        SpinnerColumn(style="cyan"),
        TextColumn("[bold]{task.description}[/]"),
        BarColumn(complete_style="cyan", finished_style="cyan"),
        MofNCompleteColumn(),
        TimeRemainingColumn(compact=True, elapsed_when_finished=True),
        TextColumn("[dim]{task.fields[name]}[/]"),
        console=console,
        transient=True,
    )
    tasks: dict[str, tuple[int, int]] = {}
    with bar:

        def report(phase: str, position: int, total: int, name: str) -> None:
            if phase not in tasks:
                # The phases run one after the other, so anything already on screen is finished.
                for done_id, done_total in tasks.values():
                    bar.update(done_id, completed=done_total)
                tasks[phase] = (bar.add_task(phase, total=total, name=""), total)
            task_id, _total = tasks[phase]
            # The callback fires before the work, so the bar shows what is finished behind it.
            bar.update(task_id, completed=position - 1, total=total, name=escape(_shorten(name, 30)))

        yield report
        for task_id, total in tasks.values():
            bar.update(task_id, completed=total)


def _print_writeback(result) -> None:
    heading = "planned mutations (nothing was sent)" if result.dry_run else "write-back"
    console.print(f"\n[bold]{heading}[/]")
    if result.skipped:
        console.print(f"[yellow]skipped, gone from Immich: {escape(', '.join(result.skipped))}[/]")
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


def _reader_hung_up(exc: OSError) -> bool:
    """Whether the thing reading stdout is gone, as `| head` leaves it.

    Windows raises EINVAL for a write to a closed pipe rather than EPIPE, and EINVAL on its own
    could be anything, so the pipe is probed before a real error gets swallowed as one.
    """
    if isinstance(exc, BrokenPipeError):
        return True
    if exc.errno != errno.EINVAL:
        return False
    try:
        sys.stdout.flush()
    except OSError:
        return True
    return False


def main() -> None:
    try:
        app()
    except MomentsError as exc:
        err.print(f"[red]error:[/] {exc}")
        raise SystemExit(exc.exit_code or 1) from exc
    except KeyboardInterrupt:
        err.print("[yellow]interrupted.[/] Progress is checkpointed; re-run to continue.")
        raise SystemExit(130) from None
    except OSError as exc:
        if _reader_hung_up(exc):
            # Python flushes stdout again on the way out, and that write cannot be caught here.
            with contextlib.suppress(OSError):
                os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
            raise SystemExit(0) from None
        # A full disk or a read-only data dir is the environment talking, not a bug to report.
        err.print(f"[red]error:[/] {exc}")
        err.print("[dim]Progress is checkpointed; fix the disk and re-run to continue.[/]")
        raise SystemExit(StorageError.exit_code) from exc
    except Exception as exc:
        err.print(f"[red]unexpected {exc.__class__.__name__}:[/] {exc}")
        err.print("[dim]Re-run with --verbose for the traceback, and please report it.[/]")
        if logging.getLogger().isEnabledFor(logging.INFO):
            raise
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
