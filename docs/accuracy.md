# Measuring search accuracy locally

The automated tests check mechanics, not whether a real family recording is searchable.
Use this benchmark to measure that separately. It searches your existing local Moments UI;
it does not upload footage, index anything, write back to Immich, or require an API key.
Search may populate Moments' usual reference-score cache.

## Make a small test set

Choose recordings covering long unbroken takes, brief face appearances, camera movement,
low light, quiet speech, background music, and noisy rooms. Include a few events lasting
only a second or two. Keep your videos and labels private.

For each query, watch the recording first and record the asset ID and the time interval
where the event actually happens. Use seconds from the start of the video, not wall-clock
time. Include queries for things you know are absent from the entire indexed library.

Copy [benchmark.example.jsonl](benchmark.example.jsonl) outside the repository and replace
the placeholder IDs, queries and times. Each line is a separate JSON object:

```json
{"id":"candles","query":"blowing out candles","expected":[{"asset_id":"your-asset-id","start_seconds":42,"end_seconds":45}]}
{"id":"absent-whale","query":"a whale jumping out of the sea","expected":[]}
```

An event may have several acceptable occurrences in `expected`. Use unique case IDs.
Choose an initial set of around 20 answerable and 10 absent queries. Keep a separate set
for final evaluation so tuning on one set does not become evidence of general accuracy.

## Run it

Start `immich-moments serve` against an already indexed library, then run:

```sh
python -m immich_moments.benchmark private-cases.jsonl --output baseline.json
```

The default server is `http://127.0.0.1:8099`; `--url` accepts loopback origins only.
Redirects and environment HTTP proxies are not used. The report contains your queries,
asset IDs, returned timestamps and aggregate index statistics. Treat it as private too.
Do not commit either the cases or the report.

Defaults are `--top-k 5`, `--tolerance 2` seconds and `--weight 0.65`. These are recorded
in the report. Use `--tolerance 0` to require a timestamp strictly inside the labelled event.
Keep the library, query set and settings fixed while comparing runs. Index statistics are
checked before and after, but equal counts cannot detect every concurrent edit, so stop
indexing during a run.

## Read the results

- `asset_hit_at_k`: fraction of answerable queries with the right video in the top K.
- `moment_hit_at_k`: fraction with the right video **and returned timestamp** in the labelled
  interval, allowing the configured tolerance. Scene overlap alone does not count.
- `moment_mrr_at_k`: mean reciprocal rank of the first correct moment; misses score zero.
- `absent_warning_rate`: fraction of absent queries that receive the "nothing looks much
  like that" warning. Ranked results are expected even for absent queries.
- `answerable_warning_rate`: fraction of answerable queries incorrectly receiving that warning.

Higher is better except for `answerable_warning_rate`. A metric is `null` if there are no
cases in its category. Individual ranks and returned timestamps are retained for diagnosis.
These are retrieval and warning measurements, not face-recognition or transcription accuracy.

## Decide whether denser sampling helps

The existing default splits long takes into at most 20-second scenes and samples each midpoint.
First inspect misses: if the event falls between samples, repeat indexing into a **separate
data directory** with `IMMICH_MOMENTS_MAX_SCENE_SECONDS=5`, using the same Immich CLIP model.
Do not use `--reindex` against your normal data directory for this comparison.

Serve the comparison index on another loopback port and rerun the exact same cases and settings.
Record indexing elapsed time, scene count and index size alongside both reports. Lowering the
scene duration changes segmentation and score distributions as well as sample density; this is
a practical comparison, not an isolated test of a new multi-frame algorithm.

Adopt a denser setting only if held-out moment retrieval improves enough to justify its indexing
and storage cost. Synthetic cards and stub embeddings cannot establish that. No real-home-video
accuracy improvement is claimed by the shipped benchmark or its unit tests.
