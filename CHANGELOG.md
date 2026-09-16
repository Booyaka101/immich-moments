# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

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
