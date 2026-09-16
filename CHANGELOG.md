# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

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
- Configuration from environment variables or an `immich-moments.toml`.
- A Docker image and a compose snippet that sits beside an existing Immich stack.

### Notes

- Visual scores are the cosine margin over the average for the same query across the library,
  divided by 0.10. CLIP is trained at a logit scale of 100, so 0.10 of cosine is ten logits and
  counts as certain. Standard scores were tried first and are wrong for this: dividing by the
  per-query spread removes the very signal that says whether CLIP found anything.
- Scores are relative to the scenes that were searched, so a filtered search rescales against
  what the filter left rather than against the whole library.
- The CLIP model name is read from `/api/system-config`; nothing is hardcoded. A model change
  that changes the vector dimension is detected and refuses to mix vector spaces, asking for
  `--reindex`.
- Videos with no audio track, no speech, HDR transfer functions, rotation matrices, or
  originals that 404 are all recorded and skipped rather than failing the run.
