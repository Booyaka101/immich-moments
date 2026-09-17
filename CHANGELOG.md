# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## 1.2.1 - 2026-09-17

### Fixed

- Setting `whisper_device=cpu` was slower than letting the same CPU be picked automatically.
  The compute type was left to ctranslate2, which read float16 off the checkpoint and settled
  for float32, printing a warning on the way. It now follows the device it lands on, the way
  the automatic path and the GPU fallback already did. Over 28 minutes of audio on `small`
  that is 40 seconds instead of 74.
- `index --phase audio` before any visual pass did nothing and reported a zero. The visual
  pass is what writes the audio speech reads, and the run now says so.
- The example compose file published the web UI on every interface. It has no auth, which the
  README says, so it now binds to `127.0.0.1`.
- Seven settings existed and were documented nowhere: `whisper_compute_type`,
  `whisper_beam_size`, `host`, `request_timeout`, `download_timeout`, `max_retries` and
  `min_request_interval`. `host` is the one the README told you to use without naming.
- The README said Whisper on the CPU runs at "roughly a fifth of real time", which reads as
  either five times faster or five times slower than the video and was neither. It now gives
  the measured figure.
- A release cuts its own GitHub release notes from the CHANGELOG. 1.1.1 reached PyPI and
  GHCR while the Releases page still said 1.1.0.

## 1.2.0 - 2026-09-17

### Added

- Search says when your library has no answer. Ranking always has a top, so asking for a dog in
  a library with no dog in it returned a confident looking page of dark interiors and nothing
  said otherwise. It now measures what your library pays a query it has no answer for, using
  twenty mundane phrases nobody films, and prints "nothing in your library looks much like
  that" above the results when the query does not beat that. The number is a property of your
  library and your CLIP model rather than a constant, measured once and kept until either
  changes. The web API carries it as `nothing_close`.

  It is a hint and not a verdict, and nothing is ever hidden on the strength of it. Over 28
  queries on a 211 scene library it was right about 10 of the 12 that had an answer and 13 of
  the 16 that did not, and every threshold that catches the queries with no answer takes real
  ones with it.

### Fixed

- The README said a query with nothing to look at "scores low". It does not, and measuring that
  is what led to the note above: on a real library an absent "a car driving" outscored a present
  "a restaurant". The scoring section now says what the visual score actually is.
- The README claimed Immich's smart search returns "15 unrelated videos" for a query
  immich-moments answers exactly. It returns all 16 with the right one eighth, which is the
  comparison that was actually measured.
- The configuration table rendered as two tables, because the `immich_public_url` row added in
  1.1.1 went in after a blank line.

## 1.1.1 - 2026-09-16

### Fixed

- "Open in Immich" opens. Beside Immich in compose, which is how the README says to run it,
  `IMMICH_URL` is a service name like `http://immich-server:2283`. That is the right address
  for the API and an address no browser can follow, so every link on every card was dead. Set
  `IMMICH_PUBLIC_URL` to the address you use and links go there; leave it out and nothing
  changes. `doctor` says so when the link would only resolve inside the container network.

## 1.1.0 - 2026-09-16

### Changed

- `label_min_similarity` is now `label_min_zscore`, and `relabel --min-similarity` is
  `--min-zscore`. The old names are refused with a message rather than quietly ignored. Nothing
  else about your index changes and no reindex is needed.

### Fixed

- Scene labels survive a change of CLIP model. The floor a label had to clear was a raw cosine
  tuned for `ViT-B-32__openai`, so pointing Immich at a SigLIP model left every scene with no
  label at all and said nothing about it. The floor is now measured in standard deviations
  above the rest of the vocabulary, which is comparable across models: over the same 211 scenes
  the winning label scores a median 0.256 under ViT-B-32 and 0.066 under
  `ViT-L-16-SigLIP-384__webli`, and 3.26 under both once standardised. The default 2.3
  reproduces what 0.22 did under ViT-B-32 on that library, 210 scenes labelled against 209.
- A vocabulary too short to ever clear the floor is refused when it is read, instead of
  labelling nothing. The best of n labels can only stand sqrt(n-1) deviations above the rest.
- The vision slider travels in the URL, so a link to a result set reproduces the blend it was
  found with. It was the one control on the page that a shared link lost.

## 1.0.5 - 2026-09-16

### Fixed

- Whisper uses the GPU in the `-cuda` image, and anywhere `immich-moments[cuda]` is installed
  on Linux. The nvidia wheels keep their libraries inside site-packages, which the dynamic
  loader does not search, so ctranslate2 reported `libcublas.so.12` missing and every run fell
  back to the CPU with a warning. Windows has been handled since the first release; the same
  now happens elsewhere, by opening those libraries by path before faster-whisper asks for
  them. Three videos out of a test library transcribe in 2.2 seconds on a 4090 against 7.4 on
  the CPU.

## 1.0.4 - 2026-09-16

### Changed

- A search maps the vector file instead of reading it into memory twice. It used to load the
  whole matrix and then copy it again row by row, even when it was keeping every row, so peak
  memory ran at about three times the file and a container with a hard limit saw that peak. At
  250,000 scenes a query now takes 493 ms and peaks at 608 MB, against 755 ms and 1.6 GB
  before, and a date filter that keeps 9% of the library takes 205 ms rather than 400 ms.
  Results are unchanged.

## 1.0.3 - 2026-09-16

### Fixed

- `latest` points at the CPU image again. The GPU build carried the metadata action's
  automatic `latest` tag, which the `-cuda` suffix does not apply to, so it overwrote the
  plain image and a `docker pull ...:latest` fetched a CUDA build that wants a GPU. Both tag
  sets are now explicit about it. No code change.

## 1.0.2 - 2026-09-16

### Fixed

- The GPU image is published again. Its tag was built from `github.repository`, which carries
  the owner's capitalisation, and a registry only takes a lowercase repository name, so the
  `-cuda` variant failed to publish for 1.0.1 while the plain images went up. It now goes
  through the same metadata action as the rest. No code change.

## 1.0.1 - 2026-09-16

### Fixed

- Piping a command into something that stops reading, `immich-moments --help | head` being the
  usual one, no longer prints a disk error. Windows reports a write to a closed pipe as EINVAL
  rather than EPIPE, so the pipe is probed before the message is believed.

## 1.0.0 - 2026-09-16

First release.

### Added

- `index`: scene segmentation with PySceneDetect, one CLIP embedding per scene from the
  Immich ML container, face matching against the people already named in Immich, and Whisper
  transcription. Checkpointed per asset, so an interrupted run continues.
- `search`: cosine over scene vectors blended with FTS5 BM25 over the transcript, with a
  configurable weight (default 0.65 visual, 0.35 text). Speech hits report the second the
  words were said, not the start of the scene.
- `search --person NAME`: narrows either channel to the scenes that person is in. Repeatable,
  and two names mean both of them in one scene. With no query it lists their scenes newest
  video first, and prints the date where the score would go. `--json` prints the same shape
  the web API returns.
- `serve`: a single page on port 8099 with thumbnails, people chips, transcript quotes, a
  blend slider, a menu of the people in the index, and a link into Immich for every scene.
  Clicking a name on a result filters on that person, and the query and filters live in the
  URL.
- `doctor`: verifies the API key, reports the CLIP and face model names the server is actually
  configured with, and round-trips a real image from the library through `/predict`.
- `index --write-back`: tags under `moments/people/` and `moments/scene/`, plus one fenced
  block in the asset description that makes moments findable from the Description filter in
  Immich's own search. A second run replaces the block instead of appending. `--dry-run`
  prints the mutations and sends nothing.
- Scene labels from a built-in vocabulary of zero-shot CLIP phrases, replaceable with
  `--labels`.
- `relabel`: tries a different vocabulary against the vectors already in the index, so it
  costs one embedding pass over the word list instead of another pass over the videos.
  `--dry-run` shows what would move.
- `search --like SCENE_ID`, and "more like this" on every card in the UI: ranks the index
  against one scene's vector instead of a query. Filters still apply, and the score is a
  plain cosine rather than the blended one.
- `index --prune`: walks the whole library and drops the videos Immich no longer has, with
  their scenes, transcript, thumbnails and audio sidecar. It refuses `--limit`, since a
  partial walk cannot say what is gone.
- Face embeddings are kept per scene, and faces are detected even when nobody is named yet.
  Naming, renaming or merging a person in Immich reaches the scenes already indexed on the
  next `index` run, without downloading anything. People Immich no longer lists by name lose
  theirs in the index the same way.
- `--write-back` skips a video that has been trashed since it was indexed and says so,
  instead of ending the run on the first 404.
- `doctor` lists the videos the last run could not index, with the reason. A failed speech
  pass is now recorded like a failed visual one, and either is cleared by the run that works.
- `search --since` and `--until`, and the two date boxes in the UI: bounds the search by the
  day the video was filmed. Inclusive at both ends, and applied before either channel scores
  anything.
- `search --album NAME`, and the album menu in the UI: narrows the search to the videos in an
  Immich album. Membership is per video, repeatable, and re-read on every `index` run, so a
  video that moves between albums follows on the next one.
- The web UI searches as you type, cross-fades the results where the browser supports view
  transitions, follows the system light or dark setting with a toggle that overrides it, drops
  to one column on a phone, and says which side of the blend found each scene with the raw
  cosine or BM25 value in the tooltip. `/`, `Esc` and `?` are bound.
- `index` shows a progress bar per phase with a remaining-time estimate when it is attached to
  a terminal, and keeps the plain one-line-per-video output when it is piped.
- Configuration from environment variables or an `immich-moments.toml`.
- A Docker image and a compose snippet that sits beside an existing Immich stack.

### Notes

- Visual scores are the cosine margin over the average for the same query across the library,
  divided by 0.10. CLIP is trained at a logit scale of 100, so 0.10 of cosine is ten logits and
  counts as certain. Standard scores were tried first and are wrong for this: dividing by the
  per-query spread removes the very signal that says whether CLIP found anything.
- Scores are relative to the scenes that were searched, so a filtered search rescales against
  what the filter left rather than against the whole library.
- The visual channel normalises against every scene the filters allow, not against the best
  few hundred that get scored. Averaging the survivors instead made the baseline climb with
  the library, which drained the visual side of the blend as the index grew.
- The CLIP model name is read from `/api/system-config`; nothing is hardcoded. A model change
  that changes the vector dimension is detected and refuses to mix vector spaces, asking for
  `--reindex`.
- Videos with no audio track, no speech, HDR transfer functions, rotation matrices, or
  originals that 404 are all recorded and skipped rather than failing the run.
