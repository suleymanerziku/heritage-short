# Heritage Short: The Patience of Buna

An automated, cache-aware pipeline that turns a single JSON storyboard into a ~77-second AI-assisted short film, mastered in both **16:9** and **9:16**, with Amharic narration and burned-in English subtitles.

> **The story.** On a rainy Kiremt afternoon in Addis Ababa, a boy waits impatiently while his mother roasts coffee over glowing charcoal. She answers with a proverb, *"ቀስ በቀስ እንቁላል በእግሩ ይሄዳል"* ("slowly, slowly, an egg learns to walk"), and the film closes on the grown son, still hearing her voice and smelling the roast when the world rushes him. Nine shots, one voice, about 77 seconds of sound, rain, and strings, built for phone screens and the Ethiopian diaspora family group chat. See [DESIGN.md](DESIGN.md) for the creative intent and user-test notes.

Everything the pipeline needs to know about the film (script, shot prompts, seeds, pacing, music brief, end card) lives in one file, [`storyboard.json`](storyboard.json), validated against [`schema/storyboard.schema.json`](schema/storyboard.schema.json). Five stages read it and write artifacts to `outputs/`:

```text
storyboard.json ──► 1 narration ──► 2 keyframes ──► 3 shots ──► 4 music ──► 5 assemble ──► outputs/final/*.mp4
   (contract)        Gemini TTS       FLUX / Gemini     Wan 2.2 or        score +           FFmpeg mix,
                                      image models      Ken Burns         loudnorm          subtitles, end card
```

Every stage hashes its inputs, skips work whose output already exists, retries flaky APIs with exponential backoff, and appends a record to [`runs.jsonl`](runs.jsonl). Details are in [ARCHITECTURE.md](ARCHITECTURE.md).

### Current state of this repository

Read this before you judge the output. It is what is committed today, not what the pipeline can do.

- `outputs/` is committed, so a fresh clone starts with a populated cache. The two masters (`outputs/final/*_16x9.mp4`, `*_9x16.mp4`) are 76.8 s each at about −14.1 LUFS.
- **All 9 shots are FFmpeg Ken Burns fallback clips** (pan/zoom over a still). The Wan 2.2 image-to-video notebook exists but has not produced shipped clips yet.
- **The score is a procedural placeholder pad**, not generated music. No `outputs/music/heritage_theme.mp3` exists. A real ACE-Step 1.5 notebook is ready to replace it (see [Optional: a real score](#optional-a-real-score-with-ace-step-15)).
- **Keyframes come from two different models:** `shot_01`–`shot_03` from NVIDIA FLUX 3 Image Klein, `shot_04`–`shot_09` from Cloudflare FLUX.2 klein 9B (a result of free-tier quota juggling, recorded in `runs.jsonl`). Expect some style drift between those groups.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.10+** | A virtual environment is recommended. |
| **FFmpeg + ffprobe** on `PATH` | Needs `libx264`, `libass` (burned subtitles), `loudnorm`, `sidechaincompress`, and AAC. A standard full build has them. `imageio-ffmpeg` (in `requirements.txt`) is a fallback for `ffmpeg` only, not `ffprobe`. |
| **GNU make** | Linux/macOS: preinstalled or via your package manager. Windows: Git Bash, Chocolatey, or Scoop. Without `make`, run the underlying commands from the [`Makefile`](Makefile) by hand, in order. |
| **`GEMINI_API_KEY`** | Free key from [Google AI Studio](https://aistudio.google.com/). Used for narration (Gemini TTS). |
| **`NVIDIA_API_KEY`** | Free key from [build.nvidia.com](https://build.nvidia.com/). **Required by `make film` as shipped**, because `keyframes.py` defaults to `nvidia-flux3`. It is *not* listed in `.env.example`; you must add it (see Quickstart). |
| **`CF_ACCOUNT_ID` + `CF_API_TOKEN`** | Cloudflare Workers AI. Needed only if you pick a Cloudflare keyframe model (`flux1`, `flux2`, `flux9b`) or run `make smoke`. |
| *Optional:* a Kaggle account | For the Wan 2.2 image-to-video stage on a free GPU. Without it, shots fall back to Ken Burns automatically. |

Free tiers have daily quotas; `runs.jsonl` records the 429 errors hit during development. If one model's quota is gone, pick another with `--model`.

---

## Quickstart

```bash
git clone https://github.com/suleymanerziku/heritage-short.git
cd heritage-short

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env                 # Windows: copy .env.example .env
# Edit .env: set GEMINI_API_KEY, and ADD a line:  NVIDIA_API_KEY=your-key-here

make film                            # validate -> narration -> keyframes -> shots -> music -> assemble
```

Final films land in `outputs/final/`:
`heritage_short__the_patience_of_buna_16x9.mp4` (1280×720) and `…_9x16.mp4` (720×1280).

Run every command from the repository root; several scripts resolve `schema/` and `outputs/` relative to the current directory.

### What to expect on a fresh clone

Because `outputs/` is committed, `make film` is mostly cache hits:

1. **Narration:** all 9 cached. No Gemini call is made.
2. **Keyframes:** `shot_01`–`shot_03` are cached; **`shot_04`–`shot_09` will regenerate** (6 API calls, about 11 s each with FLUX 3 Klein), because they were made with a different model than the default. The cache key includes the model name. Without `NVIDIA_API_KEY` the script exits immediately with a `ValueError`, even if everything were cached.
3. **Shots:** all 9 cached, **including the six whose keyframes just changed.** The shot cache does not look at keyframe pixels. After keyframes regenerate, refresh the clips:
   ```bash
   python pipeline/shots.py --force
   make music assemble
   ```
4. **Music** and **assemble** run; assemble always re-renders both masters.

Want one consistent look from scratch? Delete `outputs/keyframes/*.png` and run `python pipeline/keyframes.py --model flux9b` (or any alias; `--help` lists them), then `python pipeline/shots.py --force`.

### Other make targets

| Target | What it does |
|---|---|
| `make validate` | Validate `storyboard.json` against the schema. |
| `make smoke` | Zero-cost API check: Gemini text, Gemini TTS (Amharic), Cloudflare FLUX. |
| `make radio` | Narration plus a black-screen "radio cut" for audio-pacing review. |
| `make narration` / `keyframes` / `shots` / `music` / `assemble` | One stage at a time. |
| `make costs` | Regenerates [COSTS.md](COSTS.md) from `runs.jsonl` (per stage and model, list-price equivalents, observed free-tier limits, cost per finished minute). |

---

## Regenerating a single shot

Edit one field in `storyboard.json`, run `make film` again, and only the affected work reruns. Cached stages print `[CACHE HIT]`.

Example: change `shots[4].narration_text` (`shot_05`).

```bash
make film
# narration  : shot_05 regenerates (1 Gemini TTS call); the other 8 are [CACHE HIT]
# keyframes  : all [CACHE HIT]
# shots      : shot_05 re-renders, because narration length changed its target duration
# music      : [CACHE HIT]
# assemble   : always re-renders both masters (FFmpeg only, no API calls)
```

What each edit triggers:

| You edit (per shot) | Reruns | Extra step needed? |
|---|---|---|
| `narration_text`, `tts_voice`, `tts_style` | narration, then shots (new duration), then assemble | No |
| `pad_after_s` | shots, then assemble | No |
| `motion_prompt`, `ken_burns_motion` | shots, then assemble | No |
| `seed` | keyframe **and** shot, then assemble | No |
| `image_prompt` | keyframe only | **Yes:** `python pipeline/shots.py --shot shot_05 --force`, because the shot cache ignores keyframe pixels |
| `subtitle_en` | assemble only | No |
| `project.style_block_short` | **all** keyframes | Yes, same as above for every shot |
| `project.language` | **all** narration (and shots) | No |

Targeted runs: `python pipeline/narration.py --shot shot_05 [--force]`, `python pipeline/keyframes.py --shot shot_05 --model <alias>`, `python pipeline/shots.py --shot shot_05 [--force]`. `keyframes.py` has no `--force`; delete the PNG to force it. The exact hash fields per stage are in [ARCHITECTURE.md](ARCHITECTURE.md#2-content-addressed-caching).

### Optional: a real score with ACE-Step 1.5

`notebooks/acestep_score.ipynb` generates instrumental candidates with the real ACE-Step 1.5 model (DiT-only, fits a free 16 GB T4). Upload it and `storyboard.json` to Kaggle with a GPU on, then:

1. Run with `MODE = "draft"` (turbo, 8 steps) to audition three prompt variants: **A** masenqo + krar in the Tizita mode, **B** krar + cello + washint with faint rain, **C** slow 1970s-style Ethio-jazz. Variant A is the caption now in `storyboard.json`.
2. Pick the closest to real Ethiopian instrumentation, then re-run with `MODE = "final"` (SFT model, 32 steps) for that variant only.
3. Download the zip, extract into `outputs/`, copy your favourite to `outputs/music/heritage_theme.wav`, set `project.music.selected_file` to that path, and run `make music assemble`.

The notebook was written against ACE-Step's documented API and source (pinned to commit `ca1e85f`) and tested with a stand-in model, **not on a real GPU**. If the model cannot render masenqo or krar convincingly, a human-recorded licensed track is the better route: drop it in as `selected_file`.

### Optional: real motion with Wan 2.2 (Kaggle GPU)

```bash
python pipeline/shots.py --export-manifest      # writes outputs/shots/wan22_manifest.json
```

Upload the manifest and `outputs/keyframes/` to a Kaggle notebook, run `notebooks/wan22_batch.ipynb` (set `DRAFT_MODE = False` for 720p), download `wan22_shots_bundle.zip`, unzip into `outputs/shots/`, then `python pipeline/shots.py && make assemble`. A clip is accepted only if its sidecar hash matches the current storyboard and narration; otherwise that shot quietly stays on Ken Burns.

---

## Pipeline at a glance

| # | Stage | Script | Model / tool | Licence | Measured (from `runs.jsonl`) |
|---|---|---|---|---|---|
| 0 | Validate | `validate_storyboard.py` | `jsonschema` | MIT (library) | instant |
| 1 | Narration | `narration.py` | Gemini 3.8 Flash TTS (then a 4-model fallback chain: Flash-Lite TTS, 3.1 Flash TTS preview, and two text models) | Google Generative AI terms (hosted API) | 23 ok calls, mean 8.1 s each |
| 2 | Keyframes | `keyframes.py` | Default: FLUX 3 Image Klein (NVIDIA NIM). Alternatives: FLUX.2 klein 4B/9B and FLUX.1 schnell (Cloudflare), FLUX.1 dev (NVIDIA), Gemini image models | FLUX 3 Klein: FLUX Kommunity License · klein 9B: FLUX Non-Commercial · klein 4B, schnell: Apache 2.0 | klein 9B mean 20.7 s, FLUX 3 Klein mean 11.5 s per image |
| 3 | Shots | `shots.py`, `notebooks/wan22_batch.*` | Wan 2.2 TI2V-5B on Kaggle (optional), else FFmpeg Ken Burns | Wan 2.2: Apache 2.0 (per ETHICS.md) · FFmpeg: LGPL/GPL | Ken Burns: 9 clips, mean 7.0 s each. Wan: no shipped clips |
| 4 | Music | `music.py`, `notebooks/acestep_score.{ipynb,py}` | ACE-Step 1.5 on a Kaggle/Colab GPU (DiT-only). Shipped film still uses the procedural pad until you pick a candidate | ACE-Step 1.5: MIT · procedural pad: original code | loudnorm pass 9.8 s |
| 5 | Assemble | `assemble.py` | FFmpeg: crop, duck, two-pass loudnorm, ASS subtitles, end card | LGPL/GPL | 76.8 s masters, −14.1 LUFS (latency not logged) |

**Total measured cost: $0.00 recorded, 0.41 GPU-minutes recorded.** Treat this carefully. [COSTS.md](COSTS.md) is generated by `make costs` and holds the full breakdown, including list-price equivalents at stated benchmark rates. The numbers above are the sums of `est_cost_usd` and `gpu_min` over the records in `runs.jsonl` at the time of writing (86 records: 60 ok, 26 errors; rerun `make costs` for current totals). Every call site hard-codes `est_cost_usd = 0.0`, so $0.00 reflects "free tiers were used", not metered billing, and the 0.41 GPU-minutes come from a single Wan 2.2 test record. Failed attempts, mostly free-tier 429s, are included in the counts but cost nothing.

**Licence caveats worth knowing:**

- **Old music candidates.** An earlier version of `acestep_score.ipynb` was labelled ACE-Step 1.5 but actually loaded `facebook/musicgen-small`, whose model card lists **CC-BY-NC-4.0**, and fell back to a synthetic pad on failure. The notebook now runs real ACE-Step 1.5 (MIT). Do not ship any `score_cand_*.wav` generated by the old version, and ignore any `"ACE-Step-1.5"` records in `runs.jsonl` from before this change.
- **Keyframe models.** [ETHICS.md](ETHICS.md) lists FLUX.2 klein 4B and FLUX.1 schnell (Apache 2.0). The shipped keyframes used klein **9B** (FLUX Non-Commercial License) and FLUX 3 Image Klein (FLUX Kommunity License). Distribution is currently non-commercial, but check these terms before any commercial use.

Licence sources: [FLUX.2 klein 9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-nvfp4), [BFL klein announcement](https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence), [NVIDIA FLUX 3 Image Klein terms](https://catalog.ngc.nvidia.com/orgs/nim/black-forest-labs/containers/flux-3-image-klein/-), [MusicGen small model card](https://huggingface.co/facebook/musicgen-small), [ACE-Step 1.5 model card](https://huggingface.co/ACE-Step/acestep-v15-base/blob/main/README.md).

---

## Repository layout

```text
storyboard.json            The contract: script, prompts, seeds, pacing, music brief
schema/                    storyboard.schema.json (strict: unknown fields are rejected)
pipeline/                  One script per stage, plus utils.py (hashing, caching, retries, logging)
notebooks/                 wan22_batch.{ipynb,py} and acestep_score.{ipynb,py} (Kaggle GPU: video clips, music candidates)
outputs/                   narration/ keyframes/ shots/ music/ final/  (each artifact has a .meta.json sidecar)
runs.jsonl                 Append-only log of every model call
Makefile                   validate, smoke, radio, film, per-stage targets
```

`storyboard_v3.json` is a snapshot of an earlier revision. `pipeline/ceiling_multilingual.py` is an experiment (not part of `make film`) that probes Gemini TTS in Afaan Oromo, Tigrinya, and Somali.

## Documentation

| Doc | What's in it | State |
|---|---|---|
| [DESIGN.md](DESIGN.md) | Logline, audience, safe areas, audio direction, user-test log | Written |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Data flow, caching, retries, degradation, logging contract | Written |
| [ETHICS.md](ETHICS.md) | Consent record, licence audit, AI-labelling plan | Written (some licences out of date, see above) |
| [EXPERIMENTS.md](EXPERIMENTS.md) | Model comparisons and benchmarks | Outline only |
| [FAILURES.md](FAILURES.md) | Failure taxonomy and incident log | Outline only |
| [COSTS.md](COSTS.md) | Generated cost report (`make costs`) | Generated; only the "Exact limits" block is hand-edited |

## AI disclosure

The film is AI-assisted: visuals, narration, and (eventually) score are machine-generated from human-written direction and family memory. The assembled film ends with a 2.5-second end card stating this. Follow the platform labelling steps in [ETHICS.md](ETHICS.md) when publishing.

## Licence

No `LICENSE` file has been added yet, so all rights are reserved by default. Choose one before accepting contributions.
