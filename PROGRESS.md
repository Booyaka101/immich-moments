# PROGRESS

Status: v1.0.0 complete and verified locally. Nothing has been published.

## What exists

A Python 3.12 package, `immich-moments`, with four commands (`doctor`, `index`, `search`,
`serve`) and a single-page web UI. It talks to Immich over the public REST API and to Immich's
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
| clean wheel install | `immich-moments doctor` and `search` work from a fresh venv |
| Docker image | builds, runs, `doctor` passes on Immich's own compose network |

Test suite: 229 passed, including the two slow tests that really run Whisper. The fast subset
also passes from an unpacked sdist in a clean 3.12 venv, which is what CI checks.

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
- Faces only match people already named in Immich. Nothing is created or renamed.
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
- `relabel`, which re-scores the stored vectors against a new vocabulary. On the 16-video
  index the default vocabulary changes nothing, which is the check that the labels on disk
  still match the vectors they came from. A real run against a copy moved 209 of 210 scenes
  to a six-phrase vocabulary and back again, with every label identical afterwards and the
  scores within float32 epsilon.

## Features considered and not built

Kept out of 1.0 deliberately. Each is a real want, none is needed to ship.

- Speaker diarisation, so the description could say who said a line. Needs pyannote, a
  Hugging Face token and a licence click, which conflicts with shipping no weights.
- OCR over scene frames, for title cards and signs. Cheap to add with the same `/predict`
  plumbing, but a second model download.
- A deep link into Immich at a timestamp. Immich has no such URL today.
- Album filters. Albums are the one thing the index does not carry; person filters shipped
  because the faces were already there.
- A relevance test set, so the blend weight could be tuned rather than argued.
- A score floor, so a query nothing matches prints nothing instead of the library's best
  guess at 0.25. The score column already says so, and a badly chosen floor would hide real
  speech-led hits, which land near 0.35.

## Distribution

Nothing is published and nothing will be published from here. The owner ships it.

Single best first step: the Immich discussion that asks for exactly this,
`immich-app/immich` discussion 5936, 62 upvotes and still open. A reply there with the compose
snippet and one screenshot lands in front of the people who already know they want it, which
beats a cold post to r/selfhosted on launch day.

Order after that: PyPI release (the release workflow is written and uses trusted publishing,
so the only manual step is creating the PyPI project and the GitHub environment), GHCR image
on the same tag, then r/selfhosted once there is a link to point at.

## Repository state

- Version 1.0.0 everywhere: `pyproject.toml`, `CHANGELOG.md`, the compose example.
- `ruff check` and `ruff format --check` clean.
- CI workflow covers 3.12 and 3.13 on Ubuntu and Windows, lint, packaging, and a Docker build.
- Release workflow is tag-triggered, re-verifies, and refuses a tag that disagrees with
  `pyproject.toml`.
