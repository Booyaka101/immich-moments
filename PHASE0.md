# Phase 0: external resource verification

Run 2026-09-16. Every claim the brief depends on was fetched and checked against the
live source, not recalled.

| Resource | Status | Evidence |
|---|---|---|
| `machine-learning/immich_ml/main.py` | 200 | `main.py:167-170` `async def predict(entries: InferenceEntries = Depends(get_entries), image: bytes \| None = File(default=None), text: str \| None = Form(default=None))`; `main.py:132-134` `def get_entries(entries: str = Form())` / `request: PipelineRequest = orjson.loads(entries)` |
| `server/src/dtos/config.dto.ts` | 200 | line 629 `modelName: 'ViT-B-32__openai'` under `machineLearning.clip`; line 637 `modelName: 'buffalo_l'` under `machineLearning.facialRecognition` |
| `server/src/dtos/search.dto.ts` | 200 | line 369 `description: z.string().trim().optional()` in `MetadataSearchSchema`; filter surface has `id`, `type`, `personIds`, `tagIds`, `hasTags`, `takenAt`, `updatedAt` |
| `server/src/dtos/asset.dto.ts` | 200 | line 32 `description: z.string().optional().describe('Asset description')` in `UpdateAssetBaseSchema` |
| `server/src/controllers/tag.controller.ts` | 200 | `@Put()` → `upsertTags`, `@Put('assets')` → `bulkTagAssets`, `@Put(':id/assets')` → `tagAssets` |
| `docs.immich.app/FAQ/` | 200 | verbatim: "Immich's machine learning feature operates on the generated thumbnail. If a face is visible in the video's thumbnail it will be picked up by facial recognition." |
| `immich-app/immich` discussion #5936 | 200 | 62 upvotes, 48 comments, open. mertalev (maintainer, Dec 22 2023): "We're considering (1) for both smart search and facial recognition. … Additionally, for smart search, it would mean you could search for a specific scene in a video." CarlitoGrey (Aug 6 2026): "Not wanting to run a non-prod version of immich I decided to roll (with claude) a workaround which operates alongside immich taking advantage of the API." |

## Extra facts established in Phase 0 that changed the design

**The `/predict` response double-encodes every vector.** `immich_ml/models/transforms.py:74`
`serialize_np_array` returns `orjson.dumps(arr).decode()`, a `str`. `main.py:207`
then does `response[entry["task"]] = output`, so the JSON body is
`{"clip": "[0.1,0.2,…]"}`, a string inside JSON, not an array. Both the CLIP encoders
(`clip/visual.py:29-32`) and the face recognizer (`facial_recognition/recognition.py:68-74`,
`"embedding": serialize_np_array(embedding)`) go through it. Clients must `json.loads`
the value. `ml.py` handles both shapes so a future un-nesting does not break it.

**Response keys are the `ModelTask` values, not the request keys**: `clip` for search,
`facial-recognition` for faces (`schemas.py`, `ModelTask.SEARCH = "clip"`). Image requests
also come back with `imageHeight` / `imageWidth`.

**The flat search fields are deprecated but alive.** `search.dto.ts:249-252` marks `type`,
`page`, `personIds` etc. `DEPRECATED_FLAT_FIELD` (deprecated in v3.2.0), and
`withShapeExclusivity` only rejects them when combined with the new `filter`/`orderBy`/
`cursor` shape. `{"type":"VIDEO","withExif":false,"page":n}` is still valid on v3.2.x and
on every older release, so that is what the client sends; it falls back to the structured
`filter` shape if a server rejects it.

**Original download lives on a different controller.** `GET /api/assets/:id/original` is
`asset-media.controller.ts:92`, not `asset.controller.ts`.

**Person face embeddings are not exposed over REST.** `person.dto.ts` returns `id`, `name`,
`thumbnailPath`, `isHidden` and no vector. Matching therefore re-derives a reference
embedding per person by running `GET /api/people/:id/thumbnail` back through `/predict`,
which is what the `people_refs` table holds.

## Prior art

`github.com/CarlitoGrey/ImmichVideoFace` (the tool from the same discussion thread) was
cloned and read. 4103 lines, faces only: zero matches for `whisper`, `scenedetect`,
`transcri`, `fts5` anywhere in the tree. It reads Immich's `face_search` table directly
over Postgres and needs the library volume mounted. immich-moments does not touch Postgres
or the library volume (both explicit non-goals), and adds scene segmentation, CLIP scene
search, speech, and a search UI. Complementary, not overlapping.

## Cost

Nothing here is paid. Immich, its ML container and faster-whisper are all self-hosted and
local; there are no cloud API calls anywhere in the product. No account, key or hosting
the owner does not already have. Not cost-blocked.

## Environment confirmed on this box

Python 3.12.10, ffmpeg 8.1 (gyan full build), NVIDIA RTX 4090 24 GB (driver 610.88),
Docker 28.3.2. PyPI name `immich-moments` is unclaimed (404). Latest Immich release at
build time: v3.2.2 (2026-09-15).
