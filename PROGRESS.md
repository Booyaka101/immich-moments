# PROGRESS

Status: v1.0.3 published. PyPI, GHCR and the GitHub repository are live.

## What exists

A Python 3.12 package, `immich-moments`, with five commands (`doctor`, `index`, `search`,
`relabel`, `serve`) and a single-page web UI. It talks to Immich over the public REST API and to Immich's
own machine-learning container. It does not fork Immich, touch its database, re-encode video,
or call any cloud service.

Layout matches the brief: `cli.py`, `immich.py`, `ml.py`, `scenes.py`, `faces.py`,
`transcribe.py`, `store.py`, `search.py`, `writeback.py`, `web/`, `tests/`, plus `config.py`,
`errors.py`, `indexer.py`, `labels.py` and `media.py` that fell out of the work.

## Verified against a live stack

An Immich v3.2.2 docker-compose at `D:\tmp\immich-stack` with 16 real videos, 31 minutes of
footage, 8 named people.

| Check | Result |
| --- | --- |
| `doctor` | 11 checks pass, including a real `/predict` round trip |
| full `index` from an empty data dir | 16 videos, 210 scenes, 118 transcript segments |
| per-phase timings | discover 0.0s, people 1.8s, visual 500.5s, speech 20.8s |
| determinism | a second full index reproduced 210 scenes and 118 segments exactly |
| three hand-written queries | correct video and moment for all three, at the default weight |
| timestamp accuracy | 1.2s, 1.0s and 0s off the true moment, checked against the frame and a separate transcription of the raw audio, not against the index |
| `--write-back --dry-run` | 16 assets planned, zero write requests sent |
| real `--write-back` | 7 tags, 7 links, 16 descriptions |
| second `--write-back` | nothing to change |
| Immich description search | finds the written block, `/api/search/metadata` returns the video |
| Immich smart search | does not find it, which is the gap this closes |
| clean wheel install | `doctor`, `search` and the date range all work from a fresh venv |
| `--since` / `--until` | drops the videos outside the range and rescales the scores against what is left |
| `--like` and "more like this" | 0.984 top cosine against scene 103, and the UI round-trips `?like=103&person=Martin` |
| `relabel` | the default vocabulary moves nothing, a six-phrase one moves 209 of 210 scenes and back |
| Docker image | builds, runs, `doctor` passes on Immich's own compose network |
| renaming a person | renamed Martin to "Martin Selby" in Immich, one `index` run moved 8 faces onto the new name with the visual phase doing no work, and renaming back moved them back |
| `--prune` | trashed a video in Immich, `index --prune` dropped it by name, its scene left the search and its thumbnail left the disk (209 of 210). Restoring it and re-indexing put all 210 back |
| `--prune --limit` | refused, exit 2, before any request is sent |
| `doctor` on a failed video | lists it by name with the ffmpeg error, and still exits 0 |
| `--album` and the album menu | made two albums in Immich over 6 and 2 of the 16 videos, one `index` run picked both up in 0.1s, `search --album "Night shoots"` returned only that album's scenes, and the UI round-trips `?q=...&album=Night+shoots` |

Test suite: 283 passed, including the two slow tests that really run Whisper. The fast subset
also passes from an unpacked sdist in a clean 3.12 venv, which is what CI checks.

## Measured at scale

The live verification above is 16 videos. `tools/scale_bench.py` builds a synthetic index
through the real `Store` write path and times real searches against it, so the schema, the FTS
index and the vector file are what a real run would produce. Only the query embedding is
stubbed, because that is Immich's work and a network call would hide ours behind it. Run on
this box, 512-dim vectors, 14 scenes per video.

| scenes | videos | blended | 9% date filter | text only | more like this | vectors | peak RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5,000 | 357 | 9.8 ms | 5.4 ms | 1.2 ms | 10.6 ms | 9.8 MB | 78 MB |
| 25,000 | 1,785 | 71 ms | 38 ms | 4.3 ms | 81 ms | 48.8 MB | 202 MB |
| 100,000 | 7,142 | 290 ms | 151 ms | 19 ms | 350 ms | 195 MB | 662 MB |
| 250,000 | 17,857 | 726 ms | 380 ms | 46 ms | 892 ms | 488 MB | 1,581 MB |

Latency is linear in scene count, about 2.9 ms per thousand scenes, and search stays usable
well past any plausible home library. The text channel is cheap throughout, 46 ms at 250,000
scenes, so FTS5 is not the constraint. The visual channel is all of it.

Two things are worth fixing, neither done yet.

Peak RSS settles at about 3.2x the vector file. `_candidates` calls `VectorFile.read_all`,
which is a `np.fromfile` of the whole matrix, and then fancy-indexes it into a second array
even when the selection is every row in id order. That is two full copies per query. The
working set drops straight back afterwards, so it is transient rather than a leak, but a
container with a memory cap sees the peak, not the average.

Narrowing the search barely helps. A date range that keeps 9% of the library still costs
about half a full query at every size, because the read happens before the filter is applied
and so pays for the whole file either way. The filter only saves the matmul.

Memory-mapping the vector file would fix both: the read becomes reclaimable page cache rather
than anonymous memory, and a narrow filter would touch only the pages it needs. Skipping the
fancy index when the filters select everything removes the second copy on the common path.

## Known limits, written down rather than hidden

- The blend has no relevance judgements behind it. The visual channel is a cosine margin over
  the per-query library average and the text channel is BM25 scaled against its own best
  match, combined with a fixed weight. That is defensible, not optimal. There is no labelled
  query set, so there is no number that says whether 0.65 beats 0.55.
- `VISUAL_DECISIVE_MARGIN` is 0.10 of cosine, which is ten logits at CLIP's logit scale of
  100. It holds across the OpenCLIP variants Immich ships but has only been measured on
  `ViT-B-32__openai`.
- Scene labels come from a fixed vocabulary of about 120 English phrases. They caption, they
  do not classify.
- Faces only match people already named in Immich. Nothing is created or renamed. Naming
  someone later does reach the scenes already indexed, because the face embeddings are kept.
- A video whose pass failed (corrupt file, ML hiccup) is retried on every run, download
  included. The reason is kept and `doctor` shows it, but nothing counts the attempts, so a
  permanently broken file costs a download on every run.
- Pruned scenes leave dead rows in the vector file. Nothing reads them, but only `--reindex`
  compacts the file.
- Whisper transcribes but does not diarise, so the transcript never says who spoke.
- Write-back never removes a tag it once added.
- There is no auth on the web UI.

## Review round after 1.0.0 was cut

A full read of every module found nine real defects, all fixed with a test each before any of
them was believed.

- `discover --limit` checkpointed a truncated enumeration, so every video the limit never
  reached would have been invisible to the next `--since auto` run, permanently.
- The store's SQLite connection was bound to its creating thread, so a search from the web app
  would have raised as soon as it left the event loop.
- `serve` and `search` never checked the stored CLIP model against the configured one. Only
  `index` did, so an index built with another model scored happily against nonsense.
- Vectors are appended before the transaction that points at them. A run killed in between
  left rows nothing references, and the next append would have written past them. The tail is
  now reclaimed at open, including a half-written vector.
- Searches ran the ML HTTP call and SQLite on the event loop, which froze the page and every
  thumbnail behind one query. They run in a thread pool now, one at a time.
- `person_thumbnail` raised on the 404 Immich answers until its own thumbnail job has run, so
  `doctor` failed on a library that was merely still working.
- Both retry loops slept after the final attempt, which is dead wait before an exception
  nobody can act on. The fast test suite got a third faster as a side effect.
- A full disk surfaced as an unhandled `OSError` traceback. It is exit code 7 now, with the
  note that progress is checkpointed.
- `--labels` kept lines whose comment marker was indented, so `  # like this` became a label.

## Enhancements after the review round

- `search --person NAME` and the who menu in the UI, both AND across names. Scores are
  relative to whatever was searched, so a filtered search rescales against what the filter
  left rather than against the whole library.
- `search --json`, sharing one serialiser with the web API so the two shapes cannot drift.
- `search --like SCENE_ID` and "more like this" in the UI, ranking against one scene's vector
  rather than a query. It shares the candidate selection with the visual channel, so the
  filters and the missing-vector handling are the same code.
- `search --since` / `--until` and the date boxes in the UI, bounding a search by the day the
  video was filmed. It is one more clause in the same `Filters` object, so it narrows the
  vectors, the transcript and "more like this" without any of them knowing about dates.
- `index --prune`, dropping the videos Immich no longer lists. A full walk of the library is
  cheap (one request per 250 assets), so the flag simply forces one and diffs the ids.
- Face embeddings stored per scene, with a re-match after every people refresh. A rename,
  a merge, a hide or a name given later reaches the scenes already indexed without a
  download. Faces are now detected on every scene, named people or not, which is one more
  `/predict` call per scene and is what makes naming later work.
- `--write-back` skips trashed videos instead of dying on the first 404.
- `doctor` lists what the last run could not index and why, reading the status the index
  already kept. The speech pass now records a failure the way the visual pass always did.
- `search --album NAME` and the album menu in the UI. Membership lives in its own table and is
  replaced wholesale on every run, since Immich never says that a video left an album. The
  album detail call stopped embedding its assets in Immich 3.2, so membership comes back
  through the same paged video search the walk already uses: one request per album.
- `relabel`, which re-scores the stored vectors against a new vocabulary. On the 16-video
  index the default vocabulary changes nothing, which is the check that the labels on disk
  still match the vectors they came from. A real run against a copy moved 209 of 210 scenes
  to a six-phrase vocabulary and back again, with every label identical afterwards and the
  scores within float32 epsilon.

## Final review pass

A line by line re-read of the brief against the build, after the enhancements above.

### Where the build differs from the brief, stated rather than hidden

- The stack line offers "PyAV or ffmpeg-python". Frames and audio go through `ffmpeg` and
  `ffprobe` as subprocesses instead. ffmpeg-python is a thin argv builder over the same call
  and has had no release since 2022, and PyAV would ship a second FFmpeg build beside the one
  OpenCV already carries for scene detection. The README lists both binaries as requirements
  and a missing one is exit code 5 with the binary named in the message.
- `doctor` round-trips a person thumbnail through `/predict` rather than "one known photo". It
  is a real image from the library, and it is the one image Immich hands back without
  downloading an entire original first.
- There is a fifth command, `relabel`, which the brief does not ask for. It only re-reads
  vectors that are already in the index, so it cannot invent anything the index does not have.
- The worked example is one 14-minute video with 31 scenes. The live run was 16 videos, 31
  minutes and 210 scenes, which answers the acceptance bar of at least ten videos rather than
  the example itself.

Everything else the brief names is present: the five REST endpoints, the three `/predict`
entry shapes, the schema tables (`assets`, `scenes`, `scene_faces`, `people_refs`,
`transcript_segments` as FTS5, `run_state`), the 0.65/0.35 default blend, the fenced
description block replaced rather than appended, every edge case in the EDGE CASES paragraph
with a test each, and every assertion the TESTS paragraph asks for.

### What is missing, and what was built because it was cheap

Built in this round, because each was a small diff against something that already existed:
the person filter, `--json`, `--like`, `relabel`, the `--since` / `--until` range, `--prune`,
the face re-match, the `doctor` list of what failed, and the album filter. The range and the
album are each one more clause on the same `Filters` object, so they narrow the vectors, the
transcript and "more like this" without any of them knowing that dates or albums exist.

Left out, with the reason, in the order I would build them next.

1. A relevance test set. Twenty hand-judged queries would turn the blend weight from a
   defensible number into a measured one. Nothing else here is guesswork, and this is.
2. OCR over the scene frame, for title cards and signs. Same `/predict` plumbing, but it is a
   second model the user has to have pulled.
3. Speaker diarisation, so a description could say who said a line. pyannote needs a Hugging
   Face token and a licence click, which conflicts with shipping no weights.
4. A deep link into Immich at a timestamp. Immich has no such URL today, so the link opens the
   asset and the timestamp is printed beside it.
5. A score floor. A query nothing matches still prints the library's best guess. The score
   column says so, and a badly chosen floor would hide the speech-led hits that land near
   0.35.
6. A retry budget. A file that fails every time is downloaded again on every run. Counting the
   attempts and backing off needs a column and a flag to force a retry anyway, and getting the
   backoff wrong would hide a video that a fixed ML container would now index.

## Presentation pass

Every surface a user actually sees, gone over once with the current platform rather than a
2018 one.

- The page is rebuilt on `light-dark()` over oklch tokens with `color-mix` for the soft
  variants, so one token block covers both themes and the `@supports` fallback covers the
  browsers that lack it. The toggle writes `data-theme` and an inline script applies it before
  first paint, so a reload never flashes the other theme.
- Searching happens as you type after a 350 ms pause, with an `AbortController` cancelling the
  superseded request. A skeleton grid only appears if the answer takes longer than 200 ms,
  below which a spinner reads as a glitch rather than as progress.
- Result swaps go through `document.startViewTransition` where it exists, and straight through
  where it does not or where the user asked for reduced motion.
- Each card says which side of the blend found it, with the raw cosine or BM25 in the tooltip.
  A blended contribution bar was considered and dropped: cosine and BM25 are not on comparable
  scales, so any split shown would have been invented.
- The empty result offers the narrowings in play as one-click removals, since dropping a
  filter is the usual way out of no results.
- `/`, `Esc`, `?` are bound, `?` works from the autofocused box as long as it is empty, and
  the panel is a native `popover` rather than a hand-rolled modal.
- The phone layout unsticks the header and drops to one column. A sticky header with wrapped
  controls ate most of a 390px viewport.
- `index` on a terminal is a `rich` progress bar per phase with a remaining-time estimate.
  Piped, it still prints one plain line per video, which is what the README captures and what
  a log file wants.

## Distribution

Published on the owner's instruction: `github.com/Booyaka101/immich-moments`,
`pypi.org/project/immich-moments`, and `ghcr.io/booyaka101/immich-moments` on the v1.0.3 tag.

1.0.0 and 1.0.1 were uploaded to PyPI with the account token, because trusted publishing needs
a pending publisher registered on pypi.org and that is a logged-in browser step. Registering it
against this repository, workflow `release.yml`, environment `pypi`, makes every later tag
publish itself from CI.

Announcing is still to do, and the single best first step is the Immich discussion that asks
for exactly this: `immich-app/immich` discussion 5936, 62 upvotes and still open. A reply there
with the compose snippet and one screenshot lands in front of the people who already know they
want it, which beats a cold post to r/selfhosted. r/selfhosted after that.

## Repository state

- Version 1.0.3 everywhere: `pyproject.toml`, `CHANGELOG.md`, the compose example.
- `ruff check` and `ruff format --check` clean.
- CI workflow covers 3.12 and 3.13 on Ubuntu and Windows, lint, packaging, and a Docker build.
- Release workflow is tag-triggered, re-verifies, and refuses a tag that disagrees with
  `pyproject.toml`.
