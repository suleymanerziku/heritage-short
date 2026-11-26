#!/usr/bin/env python3
"""Aggregate runs.jsonl into a clean, complete COSTS.md  (invoked by `make costs`).

What it reports
---------------
* per (stage, model): calls, cache hits, failures / 429s / retries (when logged),
  mean + total latency, GPU minutes, logged out-of-pocket cost, and a list-price
  equivalent computed from clearly stated benchmark rates (CLI-overridable);
* "Observed Free-Tier Limits": what runs.jsonl proves (peak calls/day, peak
  calls/minute, 429 counts per provider) plus a hand-edited block with
  placeholders for the exact limits.  The hand-edited block is PRESERVED when the
  file is regenerated, so filling it in once is enough;
* "Cost per Finished Minute of Film" at the bottom.

Everything is derived from the log; nothing is invented.  Where the log cannot
answer a question (retries per call, cache hits) the report says so instead of
printing a misleading zero.

Notebook logs (Colab/Kaggle) are merged automatically when found at outputs/music/music_runs.jsonl or
outputs/shots/kaggle_runs.jsonl, or explicitly via --extra-log [colab|kaggle=]PATH.

Usage:  python pipeline/report_costs.py [--log runs.jsonl] [--out COSTS.md] [--stdout]
Stdlib only (Python 3.10+).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------- #
# Benchmark rates (list prices used ONLY for the "list-price equivalent" columns)
# --------------------------------------------------------------------------- #
DEFAULT_RATES = {
    "t4_gpu_hr": 0.35,       # USD per T4 GPU-hour (typical on-demand cloud list price)
    "tts_per_1m_chars": 15.00,  # USD per 1M input characters, commercial neural TTS
    "image_each": 0.03,      # USD per generated image, commercial image generation
    "video_per_sec": 0.05,   # USD per generated second, commercial video generation
}

STAGE_ORDER = [
    "narration", "keyframes", "shots", "music", "assemble", "radio_cut",
    "ceiling_translate", "ceiling_multilingual",
]
MANUAL_BEGIN = "<!-- BEGIN MANUAL: free-tier-limits (edit freely; preserved by `make costs`) -->"
MANUAL_END = "<!-- END MANUAL: free-tier-limits -->"
SHOT_RE = re.compile(r"^shot_\d+$")

MANUAL_TEMPLATE = f"""{MANUAL_BEGIN}
Fill in the cells marked `TODO` from each provider's dashboard or documentation and
add the date you checked. This block is kept as-is every time `make costs` regenerates
this file.

| Provider / resource | Limit type | Exact limit | Source (URL or dashboard) | Date checked |
|---|---|---|---|---|
| Cloudflare Workers AI (FLUX.2 klein) | Free daily allocation | TODO (e.g. neurons/day) | TODO | TODO |
| Cloudflare Workers AI (FLUX.2 klein) | Images per day at 1024x576 | TODO | TODO | TODO |
| Gemini API: TTS models | Requests per minute (RPM) | TODO | TODO | TODO |
| Gemini API: TTS models | Requests per day (RPD) | TODO | TODO | TODO |
| Gemini API: image models | RPM / RPD | TODO / TODO | TODO | TODO |
| Gemini API: text models (translation) | RPM / RPD | TODO / TODO | TODO | TODO |
| NVIDIA API catalog (FLUX.1 / FLUX 3) | Credits or requests | TODO | TODO | TODO |
| Kaggle | Free GPU quota per week | TODO (hours) | TODO | TODO |
| Kaggle T4 | Wan 2.2 TI2V-5B throughput | TODO s per clip, TODO clips per hour | TODO (measure with `notebooks/wan22_batch`) | TODO |
| Kaggle / Colab T4 | ACE-Step 1.5 turbo throughput | TODO s per 82 s of audio | TODO | TODO |
{MANUAL_END}"""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def parse_ts(value: Any) -> Optional[datetime]:
    if not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def load_records(path: str, platform: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Return (records, number_of_unparseable_lines). `platform` labels records lacking one."""
    records: List[Dict[str, Any]] = []
    bad = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if not isinstance(rec, dict) or "stage" not in rec:
                bad += 1
                continue
            rec["_dt"] = parse_ts(rec.get("timestamp"))
            if not isinstance(rec.get("params"), dict):
                rec["params"] = {}
            rec["_src"] = os.path.basename(path)
            rec["_platform"] = rec["params"].get("platform") or rec.get("platform") or platform
            records.append(rec)
    return records, bad


# --------------------------------------------------------------------------- #
# Classification and pricing
# --------------------------------------------------------------------------- #
def provider_of(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("@cf/"):
        return "Cloudflare Workers AI"
    if "gemini" in m:
        return "Google Gemini API"
    if m.startswith("black-forest-labs/"):
        return "NVIDIA API catalog"
    if "wan" in m or "ace-step" in m or "acestep" in m:
        return "Kaggle / Colab GPU"
    return "Local (FFmpeg / CPU)"


def is_remote_api(provider: str) -> bool:
    return provider in ("Cloudflare Workers AI", "Google Gemini API", "NVIDIA API catalog")


def category_of(rec: Dict[str, Any]) -> str:
    """tts | image | video | gpu | none  -> which benchmark rate applies."""
    stage, model = rec.get("stage"), (rec.get("model") or "").lower()
    if stage in ("narration", "ceiling_multilingual"):
        return "tts"
    if stage == "keyframes":
        return "image"
    if stage == "shots":
        return "video" if ("wan" in model and "ffmpeg" not in model) else "none"
    if stage == "music":
        return "gpu" if float(rec.get("gpu_min") or 0) > 0 else "none"
    return "none"


def tts_chars(rec: Dict[str, Any]) -> int:
    p = rec["params"]
    return len(p.get("narration_text") or p.get("text") or "")


def priced(rec: Dict[str, Any], rates: Dict[str, float]) -> Tuple[float, float, str]:
    """(api_usd, quantity, unit) for ONE successful call; (0, 0, '') when unpriced."""
    cat = category_of(rec)
    if cat == "tts":
        n = tts_chars(rec)
        return n / 1_000_000 * rates["tts_per_1m_chars"], n, "chars"
    if cat == "image":
        return rates["image_each"], 1, "images"
    if cat == "video":
        secs = float(rec["params"].get("target_duration_s") or 0)
        return secs * rates["video_per_sec"], secs, "s video"
    return 0.0, 0, ""


def gpu_usd(gpu_min: float, rates: Dict[str, float]) -> float:
    return gpu_min / 60.0 * rates["t4_gpu_hr"]


def is_ok(rec: Dict[str, Any]) -> bool:
    return rec.get("status") == "ok"


def is_cache_hit(rec: Dict[str, Any]) -> bool:
    return bool(rec.get("cache_hit")) or rec.get("status") == "cached"


def is_429(rec: Dict[str, Any]) -> bool:
    return "429" in str(rec.get("error") or "")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
class Row:
    def __init__(self, stage: str, model: str) -> None:
        self.stage, self.model = stage, model
        self.calls = self.ok = self.fail = self.rate_limited = self.cache_hits = 0
        self.retries = 0
        self.retries_logged = False
        self.lat_ok: List[float] = []
        self.lat_total = 0.0
        self.gpu_min = 0.0
        self.oop = 0.0
        self.api_usd = 0.0
        self.qty: Dict[str, float] = defaultdict(float)
        self.equiv_usd = 0.0

    def add(self, rec: Dict[str, Any], rates: Dict[str, float]) -> None:
        self.calls += 1
        lat = float(rec.get("latency_s") or 0)
        self.lat_total += lat
        self.gpu_min += float(rec.get("gpu_min") or 0)
        self.oop += float(rec.get("est_cost_usd") or 0)
        if is_cache_hit(rec):
            self.cache_hits += 1
        if "attempts" in rec["params"]:
            self.retries_logged = True
            self.retries += max(int(rec["params"].get("attempts") or 1) - 1, 0)
        if is_ok(rec):
            self.ok += 1
            self.lat_ok.append(lat)
            api, qty, unit = priced(rec, rates)
            self.api_usd += api
            if unit:
                self.qty[unit] += qty
            g = gpu_usd(float(rec.get("gpu_min") or 0), rates)
            self.equiv_usd += api if qty > 0 else g
        elif not is_cache_hit(rec):
            self.fail += 1
            if is_429(rec):
                self.rate_limited += 1

    @property
    def mean_lat(self) -> Optional[float]:
        return sum(self.lat_ok) / len(self.lat_ok) if self.lat_ok else None


def aggregate(records: List[Dict[str, Any]], rates: Dict[str, float]) -> List[Row]:
    rows: Dict[Tuple[str, str], Row] = {}
    for r in records:
        key = (str(r.get("stage")), str(r.get("model")))
        rows.setdefault(key, Row(*key)).add(r, rates)

    def order(row: Row) -> Tuple[int, str, str]:
        s = row.stage
        return (STAGE_ORDER.index(s) if s in STAGE_ORDER else len(STAGE_ORDER), s, row.model)

    return sorted(rows.values(), key=order)


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def usd(x: float) -> str:
    if x == 0:
        return "$0.00"
    return f"${x:,.2f}" if abs(x) >= 0.01 else f"${x:,.4f}"


def num(x: Optional[float], nd: int = 1) -> str:
    return "n/a" if x is None else f"{x:,.{nd}f}"


def units(qty: Dict[str, float]) -> str:
    if not qty:
        return "—"
    out = []
    for unit, v in qty.items():
        out.append(f"{v:,.0f} {unit}" if unit != "s video" else f"{v:,.1f} {unit}")
    return "; ".join(out)


def md_table(header: List[str], body: List[List[str]], align: Optional[List[str]] = None) -> str:
    align = align or ["l"] * len(header)
    sep = {"l": "---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(sep[a] for a in align) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in body]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Sections
# --------------------------------------------------------------------------- #
def table_stage_model(rows: List[Row]) -> str:
    header = ["Stage", "Model", "Calls", "Cache hits", "Failures (429)", "Retries",
              "Mean lat. ok (s)", "Total lat. (s)", "GPU min", "Out-of-pocket",
              "Billable units", "List-price equiv."]
    body: List[List[str]] = []
    for r in rows:
        retries = str(r.retries) if r.retries_logged else "n/l"
        body.append([
            r.stage, f"`{r.model}`", str(r.calls), str(r.cache_hits),
            f"{r.fail} ({r.rate_limited})", retries, num(r.mean_lat), num(r.lat_total),
            num(r.gpu_min, 2), usd(r.oop), units(r.qty), usd(r.equiv_usd),
        ])
    tot = lambda f: sum(f(r) for r in rows)  # noqa: E731
    body.append([
        "**Total**", "", f"**{tot(lambda r: r.calls)}**", f"**{tot(lambda r: r.cache_hits)}**",
        f"**{tot(lambda r: r.fail)} ({tot(lambda r: r.rate_limited)})**", "",
        "", f"**{num(tot(lambda r: r.lat_total))}**", f"**{num(tot(lambda r: r.gpu_min), 2)}**",
        f"**{usd(tot(lambda r: r.oop))}**", "", f"**{usd(tot(lambda r: r.equiv_usd))}**",
    ])
    return md_table(header, body, ["l", "l"] + ["r"] * 10)


def table_by_stage(rows: List[Row]) -> str:
    agg: Dict[str, Row] = {}
    for r in rows:
        a = agg.setdefault(r.stage, Row(r.stage, "*"))
        a.calls += r.calls; a.ok += r.ok; a.fail += r.fail; a.rate_limited += r.rate_limited
        a.cache_hits += r.cache_hits; a.lat_total += r.lat_total; a.gpu_min += r.gpu_min
        a.oop += r.oop; a.equiv_usd += r.equiv_usd; a.lat_ok += r.lat_ok
    header = ["Stage", "Calls", "OK", "Failed", "Success rate", "Mean lat. ok (s)",
              "GPU min", "Out-of-pocket", "List-price equiv."]
    body = []
    for a in agg.values():
        rate = f"{100 * a.ok / a.calls:.0f}%" if a.calls else "n/a"
        body.append([a.stage, str(a.calls), str(a.ok), str(a.fail), rate, num(a.mean_lat),
                     num(a.gpu_min, 2), usd(a.oop), usd(a.equiv_usd)])
    return md_table(header, body, ["l"] + ["r"] * 8)


def observed_limits(records: List[Dict[str, Any]]) -> str:
    by_prov: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        prov = provider_of(str(r.get("model")))
        if is_remote_api(prov):
            by_prov[prov].append(r)

    header = ["Provider", "Calls", "OK", "HTTP 429", "First 429 (UTC)", "Active days",
              "Peak OK calls / UTC day", "Peak calls / 60 s"]
    body: List[List[str]] = []
    for prov in sorted(by_prov):
        recs = by_prov[prov]
        oks = [r for r in recs if is_ok(r)]
        limited = sorted((r["_dt"] for r in recs if is_429(r) and r["_dt"]))
        per_day = Counter(r["_dt"].date().isoformat() for r in oks if r["_dt"])
        days = {r["_dt"].date().isoformat() for r in recs if r["_dt"]}
        stamps = sorted(r["_dt"] for r in recs if r["_dt"])
        peak60, j = 0, 0
        for i, t in enumerate(stamps):
            while (t - stamps[j]).total_seconds() > 60:
                j += 1
            peak60 = max(peak60, i - j + 1)
        body.append([
            prov, str(len(recs)), str(len(oks)), str(sum(1 for r in recs if is_429(r))),
            limited[0].strftime("%Y-%m-%d %H:%M") if limited else "none",
            str(len(days)), str(max(per_day.values()) if per_day else 0), str(peak60),
        ])
    auto = md_table(header, body, ["l"] + ["r"] * 7) if body else "_No remote-API calls logged yet._"

    per_model: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        if is_remote_api(provider_of(str(r.get("model")))) and is_429(r):
            per_model[(provider_of(str(r.get("model"))), str(r.get("model")))].append(r)
    detail = ""
    if per_model:
        det_rows = [[p, f"`{m}`", str(len(rs)), ", ".join(sorted({r["stage"] for r in rs}))]
                    for (p, m), rs in sorted(per_model.items())]
        detail = ("\n\nModels that returned HTTP 429 (rate limited or quota exhausted):\n\n"
                  + md_table(["Provider", "Model", "429 responses", "Stage(s)"], det_rows, ["l", "l", "r", "l"]))

    kaggle = [r for r in records if provider_of(str(r.get("model"))) == "Kaggle / Colab GPU" and is_ok(r)]
    gpu_note = ""
    if kaggle:
        real = [r for r in kaggle if r.get("shot_id") and not str(r["shot_id"]).startswith("test")]
        gpu_note = (f"\n\nGPU runs logged: {len(kaggle)} successful record(s), "
                    f"{sum(float(r.get('gpu_min') or 0) for r in kaggle):.2f} GPU-min in total"
                    + ("" if real else "; **all of them are test/merge records, so there is no measured "
                       "Kaggle T4 throughput for real shots yet**")
                    + ".")

    return (
        "These figures are what `runs.jsonl` proves, i.e. **lower bounds** on what the free tiers allow "
        "(a peak of N calls in a day means N succeeded, not that N+1 would fail). They are not the "
        "providers' published limits; those go in the table below.\n\n"
        + auto + detail + gpu_note
        + "\n\n#### Exact limits (fill in)\n\n"
    )


def final_takes(records: List[Dict[str, Any]], stage: str) -> Dict[str, Dict[str, Any]]:
    """Latest successful, non-cached record per real shot for `stage`."""
    out: Dict[str, Dict[str, Any]] = {}
    for r in sorted((x for x in records if x["stage"] == stage and is_ok(x) and not is_cache_hit(x)),
                    key=lambda x: x["_dt"] or datetime.min.replace(tzinfo=timezone.utc)):
        if SHOT_RE.match(str(r.get("shot_id"))):
            out[r["shot_id"]] = r
    return out


def film_seconds(records: List[Dict[str, Any]], override: Optional[float]) -> Tuple[Optional[float], str]:
    if override:
        return override, "--film-seconds"
    masters = sorted((r for r in records if r["stage"] == "assemble" and is_ok(r)
                      and r["params"].get("duration_s")),
                     key=lambda x: x["_dt"] or datetime.min.replace(tzinfo=timezone.utc))
    if not masters:
        return None, "no successful assemble record"
    last = masters[-1]
    return float(last["params"]["duration_s"]), f"latest assemble record ({last.get('shot_id')}, {last.get('timestamp')})"


def per_minute_section(records: List[Dict[str, Any]], rows: List[Row], rates: Dict[str, float],
                       override: Optional[float]) -> str:
    secs, src = film_seconds(records, override)
    if not secs:
        return (f"Cannot compute: {src}. Run `make assemble` or pass `--film-seconds`.\n")
    minutes = secs / 60.0

    oop = sum(r.oop for r in rows)
    equiv_all = sum(r.equiv_usd for r in rows)

    narr = final_takes(records, "narration")
    keys = final_takes(records, "keyframes")
    shots = final_takes(records, "shots")
    chars = sum(tts_chars(r) for r in narr.values())
    tts_usd = chars / 1e6 * rates["tts_per_1m_chars"]
    img_usd = len(keys) * rates["image_each"]
    clip_secs = sum(float(r["params"].get("target_duration_s") or 0) for r in shots.values())
    vid_usd = clip_secs * rates["video_per_sec"]
    music = sorted((r for r in records if r["stage"] == "music" and category_of(r) == "gpu" and is_ok(r)),
                   key=lambda x: x["_dt"] or datetime.min.replace(tzinfo=timezone.utc))
    # a clean pass needs ONE successful score render (the latest), not every audition candidate
    music_gpu_min = float(music[-1].get("gpu_min") or 0) if music else 0.0
    music_usd = gpu_usd(music_gpu_min, rates)
    reference_total = tts_usd + img_usd + vid_usd + music_usd

    # one clean pass of what was actually priced (video via Ken Burns is $0 of API spend)
    wan_final = {k: r for k, r in shots.items() if category_of(r) == "video"}
    clean_actual = tts_usd + img_usd + sum(
        float(r["params"].get("target_duration_s") or 0) * rates["video_per_sec"] for r in wan_final.values()
    ) + music_usd

    body = [
        ["Finished film length", f"{secs:.1f} s ({minutes:.2f} min)", "—", f"Source: {src}"],
        ["Out-of-pocket, every logged call", usd(oop), usd(oop / minutes),
         "Sum of `est_cost_usd` in the log."],
        ["List-price equivalent, every logged call (all experiments and re-takes)",
         usd(equiv_all), usd(equiv_all / minutes),
         "Whole experiment history, not just the final film. Failed calls are priced at $0."],
        ["List-price equivalent, one clean pass as actually built",
         usd(clean_actual), usd(clean_actual / minutes),
         f"{len(narr)} narration takes ({chars:,} chars), {len(keys)} keyframes, "
         f"{len(wan_final)} Wan clips, one score render {music_gpu_min:.2f} GPU-min. "
         "Ken Burns clips are FFmpeg, so $0 of video spend."],
        ["List-price equivalent, **all-commercial reference** (same film, every shot bought as video)",
         usd(reference_total), usd(reference_total / minutes),
         f"{len(narr)} TTS takes ({chars:,} chars) + {len(keys)} images + {clip_secs:.1f} s of shot video "
         f"({len(shots)} clips) + one score render {music_gpu_min:.2f} GPU-min on a T4."],
    ]
    table = md_table(["Basis", "Total", "Per finished minute", "What it counts"], body, ["l", "r", "r", "l"])
    parts = [
        table,
        "",
        "**Formula:** cost per finished minute = total USD / (finished film seconds / 60).",
        "",
        "Component breakdown of the all-commercial reference:",
        "",
        md_table(["Component", "Quantity", "Rate", "USD"], [
            ["Narration (TTS)", f"{chars:,} chars", f"${rates['tts_per_1m_chars']:.2f} / 1M chars", usd(tts_usd)],
            ["Keyframes", f"{len(keys)} images", f"${rates['image_each']:.2f} / image", usd(img_usd)],
            ["Shot video", f"{clip_secs:.1f} s", f"${rates['video_per_sec']:.2f} / s", usd(vid_usd)],
            ["Music (own GPU, one render)", f"{music_gpu_min:.2f} GPU-min", f"${rates['t4_gpu_hr']:.2f} / T4-h", usd(music_usd)],
            ["**Total**", "", "", f"**{usd(reference_total)}**"],
        ], ["l", "r", "r", "r"]),
    ]
    if not music:
        parts += ["", "_No successful ACE-Step run is logged, so the placeholder pad's cost is $0 "
                      "and a real score would add roughly its T4 GPU time._"]
    else:
        parts += ["", f"_Music is one draft-quality ACE-Step render ({music_gpu_min:.2f} GPU-min of generation, setup "
                      "excluded). A higher-quality final pass uses more diffusion steps and costs more._"]
    return "\n".join(parts) + "\n"


def data_notes(records: List[Dict[str, Any]], bad_lines: int, rows: List[Row],
               merged: Optional[Dict[str, int]] = None) -> List[str]:
    notes: List[str] = []
    for src, n in (merged or {}).items():
        notes.append(f"{n} record(s) were merged in from `{src}` (notebook logs that were never appended to "
                     "`runs.jsonl`). Records already present in both are counted once.")
    if any(platform_of(r) == UNKNOWN_PLATFORM for r in records
           if float(r.get("gpu_min") or 0) > 0 or provider_of(str(r.get("model"))) == "Kaggle / Colab GPU"):
        notes.append("Some GPU records carry no `platform`, so Kaggle and Colab time cannot be told apart. Newer "
                     "ACE-Step runs record it in `params.platform`; pass `--extra-log colab=PATH` or "
                     "`kaggle=PATH` to label an older log.")
    if bad_lines:
        notes.append(f"{bad_lines} line(s) in the log could not be parsed and were skipped.")
    if not any(is_cache_hit(r) for r in records):
        notes.append("**Cache hits are never logged** (`cache_hit` is always `false`; the pipeline only logs "
                     "real model calls). The cache-hit column therefore shows 0, which means 'not recorded', "
                     "not 'none occurred'.")
    if not any("attempts" in r["params"] for r in records if r["stage"] != "ceiling_translate"):
        notes.append("**Retries are not logged** for narration, keyframes or shots: `with_retries` records only "
                     "the final outcome, so 'n/l' appears in the Retries column. Only `ceiling_multilingual` "
                     "records an `attempts` count.")
    if all(float(r.get("est_cost_usd") or 0) == 0 for r in records):
        notes.append("`est_cost_usd` is hard-coded to 0.0 by the pipeline, so the out-of-pocket column "
                     "reflects the free-tier design, not a measured bill.")
    tests = [r for r in records if provider_of(str(r.get("model"))) == "Kaggle / Colab GPU"
             and str(r.get("shot_id") or "").startswith("test")]
    if tests:
        notes.append(f"{len(tests)} GPU record(s) have a `test*` shot id (e.g. `{tests[0]['shot_id']}`) with empty "
                     "params. They exercise the log-merge path and are not real shot generations, so GPU "
                     "minutes and Wan latency are **not** representative.")
    ken = [r for r in records if r["stage"] == "shots" and "ffmpeg" in str(r.get("model")).lower()]
    if ken:
        notes.append(f"{len(ken)} shot clip(s) came from the FFmpeg Ken Burns fallback and are priced $0 as video.")
    if any(r["stage"] == "music" and category_of(r) == "none" for r in records):
        notes.append("Music records for the loudnorm step carry no GPU time; a real ACE-Step run adds "
                     "`gpu_min` and shows up in the GPU columns.")
    failed_img = [r for r in records if r["stage"] == "keyframes" and not is_ok(r)]
    if failed_img:
        notes.append(f"{len(failed_img)} failed keyframe call(s) are priced at $0. A commercial API may bill some "
                     "of them (e.g. 'no usable image' responses), so real list-price spend could be slightly higher.")
    return notes


UNKNOWN_PLATFORM = "GPU notebook (platform not logged)"
DEFAULT_EXTRA_LOGS = ["outputs/music/music_runs.jsonl", "outputs/shots/kaggle_runs.jsonl"]


def record_key(rec: Dict[str, Any]) -> Tuple[Any, ...]:
    p = rec["params"]
    return (rec.get("timestamp"), rec.get("stage"), rec.get("model"), rec.get("shot_id"),
            rec.get("seed"), p.get("output_file") or p.get("variant"), rec.get("latency_s"))


def merge_records(main: List[Dict[str, Any]], extras: List[List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], int]:
    """Append extra-log records that are not already in the main log. Returns (all, added)."""
    seen = {record_key(r) for r in main}
    added = 0
    out = list(main)
    for batch in extras:
        for r in batch:
            k = record_key(r)
            if k in seen:
                continue
            seen.add(k)
            out.append(r)
            added += 1
    return out, added


def platform_of(rec: Dict[str, Any]) -> str:
    return rec.get("_platform") or UNKNOWN_PLATFORM


def gpu_by_platform(records: List[Dict[str, Any]], rates: Dict[str, float]) -> str:
    gpu_recs = [r for r in records
                if float(r.get("gpu_min") or 0) > 0 or provider_of(str(r.get("model"))) == "Kaggle / Colab GPU"]
    if not gpu_recs:
        return "_No GPU runs logged._"
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in gpu_recs:
        groups[platform_of(r)].append(r)
    header = ["Platform", "Records", "OK", "Failed", "Logged GPU min", "T4 list price", "Workloads", "Source log(s)"]
    body = []
    for plat in sorted(groups):
        rs = groups[plat]
        mins = sum(float(r.get("gpu_min") or 0) for r in rs)
        wl = sorted({f"{r['stage']}: {r.get('model')}" for r in rs})
        body.append([plat, str(len(rs)), str(sum(1 for r in rs if is_ok(r))),
                     str(sum(1 for r in rs if not is_ok(r) and not is_cache_hit(r))),
                     f"{mins:.2f}", usd(gpu_usd(mins, rates)),
                     "<br>".join(f"`{w}`" for w in wl), ", ".join(sorted({r["_src"] for r in rs}))])
    tot = sum(float(r.get("gpu_min") or 0) for r in gpu_recs)
    body.append(["**Total**", str(len(gpu_recs)), "", "", f"**{tot:.2f}**", f"**{usd(gpu_usd(tot, rates))}**", "", ""])
    return (md_table(header, body, ["l", "r", "r", "r", "r", "r", "l", "l"])
            + "\n\n**Logged GPU minutes cover model generation only.** Session setup (installing dependencies, "
              "downloading weights, loading the model) and idle time are not logged, so the real Colab or Kaggle "
              "usage is higher. Check Colab's compute-unit counter or Kaggle's quota page for the true figure.")


# --------------------------------------------------------------------------- #
# Document assembly
# --------------------------------------------------------------------------- #
def read_manual_block(out_path: str) -> str:
    if os.path.exists(out_path):
        text = open(out_path, "r", encoding="utf-8").read()
        m = re.search(re.escape(MANUAL_BEGIN) + r".*?" + re.escape(MANUAL_END), text, flags=re.S)
        if m:
            return m.group(0)
        # tolerate a header-only edit of the marker line
        m = re.search(r"<!-- BEGIN MANUAL: free-tier-limits.*?-->.*?" + re.escape(MANUAL_END), text, flags=re.S)
        if m:
            return m.group(0)
    return MANUAL_TEMPLATE


def build_document(records: List[Dict[str, Any]], bad_lines: int, rates: Dict[str, float],
                   manual: str, film_override: Optional[float], log_name: str,
                   merged: Optional[Dict[str, int]] = None) -> str:
    rows = aggregate(records, rates)
    stamps = sorted(r["_dt"] for r in records if r["_dt"])
    span = (f"{stamps[0].strftime('%Y-%m-%d %H:%M')} to {stamps[-1].strftime('%Y-%m-%d %H:%M')} UTC"
            if stamps else "n/a")
    oop = sum(r.oop for r in rows)
    equiv = sum(r.equiv_usd for r in rows)
    gpu = sum(r.gpu_min for r in rows)
    calls = sum(r.calls for r in rows)
    fail = sum(r.fail for r in rows)

    rate_table = md_table(["Benchmark", "Rate", "Applied to"], [
        ["GPU time", f"${rates['t4_gpu_hr']:.2f} per T4-hour", "`gpu_min` from Kaggle/Colab runs"],
        ["Commercial TTS", f"${rates['tts_per_1m_chars']:.2f} per 1M characters", "narration and ceiling TTS: characters of the text sent"],
        ["Commercial image generation", f"${rates['image_each']:.2f} per image", "keyframes: each successful call"],
        ["Commercial video generation", f"${rates['video_per_sec']:.2f} per second", "Wan 2.2 shots: `target_duration_s` of each successful call"],
    ], ["l", "r", "l"])

    notes = data_notes(records, bad_lines, rows, merged)
    notes_md = "\n".join(f"- {n}" for n in notes) if notes else "- No data-quality issues detected."

    doc = f"""# Costs

> Generated by `make costs` (`pipeline/report_costs.py`) from `{log_name}`. **Do not edit by hand**
> except inside the marked "Exact limits" block, which is preserved on regeneration.
> Data: {len(records)} records, {span}{(" (incl. " + ", ".join(f"{n} from {k}" for k, n in merged.items()) + ")") if merged else ""}.

## Summary

| Measure | Value |
|---|---|
| Out-of-pocket cost (logged) | **{usd(oop)}** |
| List-price equivalent of everything logged | {usd(equiv)} |
| Logged calls | {calls} ({calls - fail} ok, {fail} failed) |
| GPU minutes (Kaggle/Colab) | {gpu:.2f} |

Everything in this project runs on free tiers, so the real bill is zero. The list-price columns answer a
different question: what the same work would cost on commercial services at the benchmark rates below. Change
a rate with a flag, for example `python pipeline/report_costs.py --rate-image 0.04`.

## Benchmark rates (list-price equivalents)

{rate_table}

Rates are round benchmark figures chosen for comparison, not quotes from a specific vendor. A row's list-price
equivalent uses the commercial rate when one applies to its billable unit and falls back to GPU time on the
T4 rate otherwise (for example an ACE-Step run). FFmpeg stages are CPU-only and cost $0.

## Aggregate by stage and model

Calls count every logged record. *Failures* are non-ok, non-cached records, with HTTP 429 responses in
brackets. *Mean lat. ok* averages successful calls only; *Total lat.* sums all calls. `n/l` means the pipeline
does not log that value (see the data-quality notes).

{table_stage_model(rows)}

### By stage

{table_by_stage(rows)}

### GPU usage by platform

{gpu_by_platform(records, rates)}

## Observed Free-Tier Limits

{observed_limits(records)}{manual}

## Cost per Finished Minute of Film

{per_minute_section(records, rows, rates, film_override)}
## Methodology and data-quality notes

Cost = quantity x benchmark rate, summed over successful calls only. Quantities come from each record's
`params` (characters of narration text, one per image, `target_duration_s` for Wan clips) and from `gpu_min`.
Failed calls are priced at $0.

{notes_md}
"""
    return doc


def main(argv: Optional[Iterable[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Aggregate runs.jsonl into COSTS.md")
    ap.add_argument("--log", default="runs.jsonl")
    ap.add_argument("--out", default="COSTS.md")
    ap.add_argument("--stdout", action="store_true", help="print instead of writing the file")
    ap.add_argument("--extra-log", action="append", default=None, metavar="[PLATFORM=]PATH",
                    help="additional runs log to merge, e.g. colab=outputs/music/music_runs.jsonl "
                         "(repeatable; duplicates of runs.jsonl records are ignored). Default: auto-discover "
                         + ", ".join(DEFAULT_EXTRA_LOGS))
    ap.add_argument("--film-seconds", type=float, default=None,
                    help="override the finished film length (default: latest successful assemble record)")
    ap.add_argument("--rate-t4-hr", type=float, default=DEFAULT_RATES["t4_gpu_hr"])
    ap.add_argument("--rate-tts-1m-chars", type=float, default=DEFAULT_RATES["tts_per_1m_chars"])
    ap.add_argument("--rate-image", type=float, default=DEFAULT_RATES["image_each"])
    ap.add_argument("--rate-video-sec", type=float, default=DEFAULT_RATES["video_per_sec"])
    args = ap.parse_args(list(argv) if argv is not None else None)

    if not os.path.exists(args.log):
        print(f"error: {args.log} not found. Run a pipeline stage first.", file=sys.stderr)
        return 1
    records, bad = load_records(args.log)
    if not records:
        print(f"error: {args.log} has no usable records.", file=sys.stderr)
        return 1

    specs: List[Tuple[Optional[str], str]] = []
    for item in (args.extra_log if args.extra_log is not None else []):
        label, _, path = item.partition("=") if "=" in item and not re.match(r"^[A-Za-z]:[\\/]", item) else (None, "", item)
        specs.append((label or None, path))
    if args.extra_log is None:  # auto-discover next to the main log
        base = os.path.dirname(os.path.abspath(args.log))
        specs = [(None, os.path.join(base, d)) for d in DEFAULT_EXTRA_LOGS if os.path.exists(os.path.join(base, d))]
    extras: List[List[Dict[str, Any]]] = []
    for label, path in specs:
        if not os.path.exists(path):
            print(f"warning: extra log {path} not found, skipped", file=sys.stderr)
            continue
        batch, nbad = load_records(path, label)
        bad += nbad
        extras.append(batch)
    before = {id(r) for r in records}
    records, _ = merge_records(records, extras)
    merged: Dict[str, int] = defaultdict(int)
    for r in records:
        if id(r) not in before:
            merged[r["_src"]] += 1

    rates = {"t4_gpu_hr": args.rate_t4_hr, "tts_per_1m_chars": args.rate_tts_1m_chars,
             "image_each": args.rate_image, "video_per_sec": args.rate_video_sec}
    doc = build_document(records, bad, rates, read_manual_block(args.out), args.film_seconds,
                         os.path.basename(args.log), dict(merged))
    if args.stdout:
        sys.stdout.buffer.write(doc.encode("utf-8"))
        return 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(doc)
    print(f"Wrote {args.out} from {len(records)} records in {args.log}"
          + (f" ({bad} unparseable line(s) skipped)" if bad else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
