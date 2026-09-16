# immich-moments

Immich's own FAQ is straight about where video search stops:

> Immich's machine learning feature operates on the generated thumbnail. If a face is visible
> in the video's thumbnail it will be picked up by facial recognition.

One frame per video. A fourteen minute birthday tape is one thumbnail, so the moment someone
blows out the candles is not findable, and neither is anything anybody said.

immich-moments indexes the inside of your videos. It splits each one into scenes, embeds a
frame from each scene with the same CLIP model your Immich already runs, matches faces against
the people you have already named in Immich, transcribes the audio with Whisper, and puts it
all behind one search box.

It talks to Immich over the public REST API. No fork, no Postgres access, no re-encoding, and
nothing leaves your machine.

![the search UI](docs/search.png)

## What you get

```
$ immich-moments search "a train moving through the dark" --limit 3
                         3 scene(s) for 'a train moving through the dark'
┌───────┬───────┬─────────────────────┬─────────────────┬────────┬────────────────────────────────┐
│ score │    at │ video               │ scene           │ people │ said                           │
├───────┼───────┼─────────────────────┼─────────────────┼────────┼────────────────────────────────┤
│ 0.943 │ 00:04 │ mothersday-ep11.mp4 │ a train passing │ -      │ You're the first person I've   │
│       │       │                     │                 │        │ seen on this train…            │
│ 0.650 │ 00:46 │ mothersday-ep11.mp4 │ a train passing │ -      │                                │
│ 0.650 │ 01:08 │ mothersday-ep11.mp4 │ a train passing │ -      │                                │
└───────┴───────┴─────────────────────┴─────────────────┴────────┴────────────────────────────────┘
http://localhost:2283/photos/d625e56a-0254-4d24-ac08-d3cf2d297931
```

Each hit is a scene, not a file: a timestamp you can scrub to, the people in it, and the line
that was spoken there. `serve` puts the same thing in a browser with thumbnails, and every
result links back to the asset in Immich.

Optionally it writes what it found back into Immich, as tags and a fenced block in the asset
description, so the moments are findable from Immich's own search bar too.

## Requirements

- An Immich server (tested against v3.2.2) and an API key
- Immich's machine-learning container reachable on its own port, usually 3003
- ffmpeg and ffprobe on PATH
- Python 3.12 or newer
- A GPU is optional. Whisper on the CPU works, at roughly a fifth of real time on `small`.

## Install

```
pip install immich-moments
```

or without installing anything permanently:

```
uvx immich-moments doctor
```

or next to an existing Immich stack, with
[docker-compose.example.yml](docker-compose.example.yml) copied into the same folder as
Immich's own compose file:

```yaml
services:
  immich-moments:
    image: ghcr.io/booyaka101/immich-moments:1.0.0
    environment:
      IMMICH_URL: http://immich-server:2283
      IMMICH_API_KEY: ${IMMICH_API_KEY}
      IMMICH_ML_URL: http://immich-machine-learning:3003
    volumes:
      - immich-moments-data:/data
    ports:
      - 8099:8099
```

```
docker compose -f docker-compose.yml -f docker-compose.example.yml run --rm immich-moments index
docker compose -f docker-compose.yml -f docker-compose.example.yml up -d immich-moments
```

## Use it

Point it at your server. Make the key in Immich under Account Settings, API Keys.

```
export IMMICH_URL=http://localhost:2283
export IMMICH_API_KEY=...
export IMMICH_ML_URL=http://localhost:3003
```

### doctor

Checks the key, reports which models your server is actually configured with, and pushes a
real image from your own library through `/predict` before you wait on an index run.

```
$ immich-moments doctor
OK  Immich API       http://localhost:2283 (v3.2.2)
OK  CLIP model       ViT-B-32__openai
OK  Face model       buffalo_l
    Face thresholds  minScore 0.7, maxDistance 0.5
OK  Videos visible   16
OK  Named people     8
OK  ML container     http://localhost:3003 answered /ping
OK  CLIP text        512-dim vector
OK  CLIP image       round-tripped Martin's face thumbnail, 512-dim, cosine
                     +0.168 against the probe text
OK  Face pipeline    1 face(s) in Martin's face thumbnail
OK  Index            16 assets, 210 scenes, 118 transcript segments at
                     D:\tmp\moments-fresh\moments.sqlite3
```

Every line is a real request. A failure exits non-zero and says which one.

### index

```
$ immich-moments index
clip=ViT-B-32__openai (512-dim)  faces=buffalo_l  data=D:\tmp\moments-fresh
visual 1/16  umbra-short.mp4
visual 2/16  rotated_phone_clip.mp4
...
visual 16/16  day12-film.mp4
speech 1/16  umbra-short.mp4
...
speech 16/16  day12-film.mp4
discovered               16
people references        8
videos indexed (visual)  16
videos indexed (speech)  16
scenes                   210
transcript segments      118
no audio track           2
no speech found          5
                  timings
┌──────────┬────────┬─────────┬───────────┐
│ phase    │ assets │ seconds │ per asset │
├──────────┼────────┼─────────┼───────────┤
│ discover │     16 │     0.0 │      0.0s │
│ people   │      8 │     1.8 │      0.2s │
│ visual   │     16 │   500.5 │     31.3s │
│ speech   │     16 │    20.8 │      1.3s │
└──────────┴────────┴─────────┴───────────┘
```

Those numbers are a real run over 16 videos, 31 minutes of footage, on an RTX 4090 with
Immich's ML container on the same box. Most of the visual time is downloading originals and
decoding them, not the model.

It checkpoints after every asset, so an interrupted run continues where it stopped.
`--since auto` (the default) only looks at assets added since the last run. `--reindex` throws
the index away and rebuilds it, which is also what a change of CLIP model needs.

### search

```
$ immich-moments search "the drug was pulled off the market" --limit 2
                        2 scene(s) for 'the drug was pulled off the market'
┌───────┬───────┬─────────────────────┬────────────────────────┬────────┬─────────────────────────┐
│ score │    at │ video               │ scene                  │ people │ said                    │
├───────┼───────┼─────────────────────┼────────────────────────┼────────┼─────────────────────────┤
│ 0.350 │ 03:54 │ mothersday-ep09.mp4 │ autumn leaves          │ -      │ Memento was withdrawn   │
│       │       │                     │                        │        │ from the United States  │
│       │       │                     │                        │        │ ma…                     │
│ 0.349 │ 03:18 │ mothersday-ep09.mp4 │ a person smiling at    │ -      │ It's the drug. We all   │
│       │       │                     │ the camera             │        │ know it's the drug.     │
│       │       │                     │                        │        │ Foss,…                  │
└───────┴───────┴─────────────────────┴────────────────────────┴────────┴─────────────────────────┘
http://localhost:2283/photos/21dda574-e4f6-434c-a64f-e1b42ce88ae4

$ immich-moments search "something living behind the wall" --limit 2
                         2 scene(s) for 'something living behind the wall'
┌───────┬───────┬─────────────────────┬────────────────────────┬────────┬─────────────────────────┐
│ score │    at │ video               │ scene                  │ people │ said                    │
├───────┼───────┼─────────────────────┼────────────────────────┼────────┼─────────────────────────┤
│ 0.350 │ 00:40 │ mothersday-ep08.mp4 │ a dark indoor scene    │ -      │ there's something       │
│       │       │                     │                        │        │ behind the wall it's    │
│       │       │                     │                        │        │ been the…               │
│ 0.259 │ 01:52 │ exposure-film.mp4   │ a camera pointing at   │ -      │                         │
│       │       │                     │ the floor              │        │                         │
└───────┴───────┴─────────────────────┴────────────────────────┴────────┴─────────────────────────┘
http://localhost:2283/photos/8ae3a967-18b5-4626-b30c-f92e9ec13e9c
```

Both of those are speech hits at the default weight. The timestamp is where the words were
said when speech decided the hit, and where the cut is when the picture did. `--asset <id>`
searches inside one video.

`--weight` is the blend: 1 is vision only, 0 is speech only, the default is 0.65. Pull it down
towards 0.3 when you want the transcript to decide.

### serve

```
$ immich-moments serve
immich-moments on http://127.0.0.1:8099
```

One page, one search box, a slider for the blend, thumbnails, and a link into Immich for every
scene.

### write it back into Immich

`--write-back` adds tags (`moments/people/Anna`, `moments/scene/birthday cake`) and one fenced
block in the asset description. Everything else in the description is left alone, and a second
run replaces the block instead of appending another one.

```
$ immich-moments index --write-back --dry-run

planned mutations (nothing was sent)

mothersday-ep08.mp4 8ae3a967-18b5-4626-b30c-f92e9ec13e9c
  + tag moments/people/Sam
  + tag moments/scene/a black screen
  + tag moments/scene/a person talking to camera
  + tag moments/scene/a dark indoor scene
  + tag moments/scene/a sunrise
  + tag moments/scene/a video game on a television
  + tag moments/scene/baking in an oven
  ~ description
    Mothers Day series, episode 8.

    <!-- immich-moments:begin -->
    00:00 a black screen | "I can hear it in the walls. When everyone's asleep."
    00:04 a person talking to camera | "It's just the two of us tonight, Mayflower. Daddy's working late."
    00:09 a dark indoor scene | "I can hear it in the walls. When everyone's asleep."
    00:13 a dark indoor scene | "I started writing down the times."
    00:18 a person talking to camera (Sam) | "3, 14. 3, 14. Every night."
    00:29 a sunrise | "Maya?"
    00:33 a dark indoor scene | "Maya, baby, why are you out of bed? That's not Maya."
    00:40 a dark indoor scene | "there's something behind the wall it's been there the whole time it's not a"
    00:50 a dark indoor scene | "signal sigh it's an answer who are you talking to it's not a signal sigh it's"
    00:55 a video game on a television
    00:58 a dark indoor scene
    01:03 baking in an oven
    <!-- immich-moments:end -->

...

16 asset(s) would change. Re-run without --dry-run to apply.
```

"Mothers Day series, episode 8." was already in that description. It stays. Drop `--dry-run`
and the same plan is applied, and a second run finds nothing left to do:

```
$ immich-moments index --write-back
7 tag(s) upserted, 7 asset-tag link(s), 16 description(s) written.

$ immich-moments index --write-back
write-back
nothing to change: Immich already has everything this index knows.
```

Then Immich's own search finds the words:

```
$ curl -s -X POST -H "x-api-key: $IMMICH_API_KEY" -H "content-type: application/json" \
    -d '{"description":"something behind the wall","type":"VIDEO"}' \
    http://localhost:2283/api/search/metadata
1 result: mothersday-ep08.mp4
```

That is the endpoint behind the Description field in Immich's own search filters, so the same
words typed into Immich find the video. Immich's smart search does not: the same query through
`/api/search/smart` returns 15 unrelated videos, because it only ever saw one thumbnail per
file. The `moments/people/` and `moments/scene/` tags show up in Immich's tag browser as well.

## Configuration

Every setting is an environment variable or a key in `immich-moments.toml`, looked for in the
working directory and then in the data directory. The environment wins over the file, and a
command line flag wins over both.

| Setting | Variable | Default | What it does |
|---|---|---|---|
| `immich_url` | `IMMICH_URL` | none | Your Immich server |
| `immich_api_key` | `IMMICH_API_KEY` | none | API key from Account Settings |
| `ml_url` | `IMMICH_ML_URL` | `http://localhost:3003` | Immich's ML container |
| `data_dir` | `DATA_DIR` | `~/.local/share/immich-moments` | Index, thumbnails, vectors |
| `scene_threshold` | `IMMICH_MOMENTS_SCENE_THRESHOLD` | `27.0` | PySceneDetect content threshold |
| `min_scene_seconds` | `IMMICH_MOMENTS_MIN_SCENE_SECONDS` | `1.5` | Shorter cuts get merged |
| `max_scene_seconds` | `IMMICH_MOMENTS_MAX_SCENE_SECONDS` | `20.0` | Long takes get split |
| `face_min_score` | `IMMICH_MOMENTS_FACE_MIN_SCORE` | `0.7` | Detection confidence floor |
| `face_max_distance` | `IMMICH_MOMENTS_FACE_MAX_DISTANCE` | `0.5` | How close a face must be to count as that person |
| `whisper_model` | `IMMICH_MOMENTS_WHISPER_MODEL` | `small` | Any faster-whisper model name |
| `whisper_device` | `IMMICH_MOMENTS_WHISPER_DEVICE` | `auto` | `cuda`, `cpu` or `auto` |
| `whisper_language` | `IMMICH_MOMENTS_WHISPER_LANGUAGE` | detect | ISO code, e.g. `en` |
| `visual_weight` | `IMMICH_MOMENTS_VISUAL_WEIGHT` | `0.65` | Vision against speech in the blend |
| `label_min_similarity` | `IMMICH_MOMENTS_LABEL_MIN_SIMILARITY` | `0.22` | Below this a scene gets no label |
| `port` | `IMMICH_MOMENTS_PORT` | `8099` | Web UI port |

```toml
[immich_moments]
immich_url = "http://localhost:2283"
whisper_model = "medium"
visual_weight = 0.5
```

### Exit codes

Every command exits deliberately, so a cron job or a shell script can tell a broken key from a
full disk.

| Code | Meaning |
|---|---|
| 0 | Worked. Assets whose original had been deleted are skipped, not failed |
| 1 | Anything else that went wrong |
| 2 | Configuration: a missing URL or key, or a setting that will not parse |
| 3 | Immich refused or could not answer |
| 4 | The ML container refused or could not answer |
| 5 | ffmpeg or ffprobe could not read the file |
| 6 | The index was built with a different CLIP model. Re-run with `--reindex` |
| 7 | The data directory could not be read or written. Progress is checkpointed, so fix it and re-run |

## How it works

1. `GET /api/search/metadata` lists your videos, `GET /api/assets/:id/original` fetches one.
2. PySceneDetect cuts it into scenes. Long takes are split, very short cuts merged, and a clip
   shorter than one scene becomes a single scene spanning the file.
3. The mid frame of each scene goes to your Immich ML container's `/predict`, which returns a
   CLIP vector from whatever model your server is configured with. The model name is read from
   `/api/system-config`, never assumed.
4. The same frame goes through face detection, and each face is matched against reference
   embeddings derived from the thumbnails of the people you have named in Immich.
5. The audio track goes to faster-whisper, on the GPU when there is one.
6. Vectors land in a plain float32 matrix beside a SQLite database, transcripts in FTS5. No
   vector database, nothing to run.
7. A query is embedded once, scored against every scene vector by cosine, and blended with the
   BM25 score of the transcript. The two channels are normalised separately. The visual score
   is the cosine margin over the library average for that query, in cosine units, so a query
   with nothing to look at scores low instead of crowning whichever scene happened to come
   closest. The text score is scaled against the best match, because BM25 has no absolute
   meaning and the weakest match still matched.

## Limitations

- Scene labels come from a fixed vocabulary of about 120 English phrases
  (`--labels your-own.txt` replaces it). They are a caption, not a classifier.
- Faces are matched against people you have already named in Immich. It will not find people
  Immich does not know, and it never creates or renames anyone.
- The default weight of 0.65 favours vision. Speech-led queries still work at the default,
  but if you want the transcript to lead, `--weight 0.3` or the slider in the UI does it.
- The blend is a weighted sum of two separately normalised channels with no relevance
  judgements behind it. It is tuned to be defensible, not optimal.
- Whisper transcribes, it does not diarise. The transcript does not say who spoke.
- Write-back only touches tags under `moments/` and the fenced block. It will not remove tags
  for things you later drop from the index.
- Immich has no deep link to a timestamp inside a video, so a result links to the asset and
  tells you where to scrub to.
- The index is local and single user. There is no auth on the web UI, so bind it to localhost
  or put it behind whatever you already use.

## Development

```
git clone https://github.com/Booyaka101/immich-moments
cd immich-moments
pip install -e ".[dev]"
pytest -m "not slow and not live"
```

The slow tests download a Whisper model and run it against a generated fixture video:

```
pytest -m slow
```

## Licence

MIT.
