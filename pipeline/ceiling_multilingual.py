"""Ceiling test: how far does Gemini 3.8 Flash TTS go beyond the primary language?

The film's primary language is Amharic (storyboard project.language == "am").
This script probes three neighbouring languages that Gemini may or may not
officially support for speech: Afaan Oromo (om), Tigrinya (ti) and Somali (so).

Steps
  1. Translate narration_text of shot_01 / shot_05 / shot_08 from the storyboard
     into each target language with Gemini 3.8 Flash
     -> outputs/ceiling/translations.json
  2. Synthesize each translated line with Gemini 3.8 Flash TTS (same voice as the
     Amharic baseline, NO model fallback chain: a ceiling test must measure the
     named model, not whichever fallback happened to work)
     -> outputs/ceiling/<lang>/<shot_id>.wav
     -> one runs.jsonl record per (language, shot) with stage="ceiling_multilingual"
        (latency, error, audio duration, words/sec live in the record / params)
  3. Compare speaking rate (words/sec), latency and error rate with the Amharic
     baseline and write a paste-ready Markdown table
     -> outputs/ceiling/summary.md  (also printed)

Native-speaker intelligibility scores (1-5) are human input: fill in
outputs/ceiling/intelligibility.json (a template is created on first run), then
run with --report-only to regenerate the table with the scores included.

Usage
  python pipeline/ceiling_multilingual.py                    # all 3 languages x 3 shots
  python pipeline/ceiling_multilingual.py --langs ti,so
  python pipeline/ceiling_multilingual.py --dry-run          # plan only, no API calls
  python pipeline/ceiling_multilingual.py --report-only      # rebuild summary.md
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import requests
from dotenv import load_dotenv

from pipeline.narration import ensure_wav_format
from pipeline.utils import (
    check_cache,
    compute_input_hash,
    log_run,
    probe_audio_duration,
    validate_storyboard,
    with_retries,
    write_cache_meta,
)

STAGE = "ceiling_multilingual"          # TTS records (what the experiment asked for)
TRANSLATE_STAGE = "ceiling_translate"   # kept separate so TTS error rates stay clean

DEFAULT_SHOTS = ["shot_01", "shot_05", "shot_08"]
DEFAULT_TRANSLATE_MODEL = "gemini-3.8-flash"
DEFAULT_TTS_MODEL = "gemini-3.8-flash-tts"

LANGUAGES: Dict[str, Dict[str, str]] = {
    "om": {"name": "Afaan Oromo", "script": "latin", "orthography": "standard Qubee (Latin-script) orthography"},
    "ti": {"name": "Tigrinya", "script": "ethiopic", "orthography": "standard Ge'ez (Ethiopic) script"},
    "so": {"name": "Somali", "script": "latin", "orthography": "standard Latin-script Somali orthography"},
}
SOURCE_NAMES = {"am": "Amharic"}

API_ROOT = "https://generativelanguage.googleapis.com/v1beta/models"
PROMPT_VERSION = 1

# Audio shorter/longer than this multiple of the Amharic duration for the same
# shot is flagged "suspect" (likely truncated, looping or garbled speech that the
# API still returned as HTTP 200).
SUSPECT_RATIO_RANGE = (0.4, 2.5)

WORD_RE = re.compile(r"[^\W\d_]+(?:['’ʼ][^\W\d_]+)*", re.UNICODE)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_words(text: str) -> int:
    """Unicode-letter word count (works for Ethiopic and Latin script alike)."""
    return len(WORD_RE.findall(text))


def ethiopic_ratio(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    eth = sum(
        1
        for c in letters
        if 0x1200 <= ord(c) <= 0x139F or 0x2D80 <= ord(c) <= 0x2DDF or 0xAB00 <= ord(c) <= 0xAB2F
    )
    return eth / len(letters)


def load_json(path: str, default: Any) -> Any:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# HTTP layer (REST only: exact status codes, key kept out of URLs and logs)
# --------------------------------------------------------------------------- #
class APIError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None):
        super().__init__(message)
        self.status = status


class RetryableAPIError(APIError):
    """Transient failure (timeouts, 5xx, short 429s) worth retrying."""


def _retry_delay_s(body: str) -> float:
    m = re.search(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"', body)
    return float(m.group(1)) if m else 0.0


def _post(model: str, payload: Dict[str, Any], api_key: str, timeout: float) -> Dict[str, Any]:
    url = f"{API_ROOT}/{model}:generateContent"
    try:
        res = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json", "x-goog-api-key": api_key},
            timeout=timeout,
        )
    except (requests.Timeout, requests.ConnectionError) as exc:
        raise RetryableAPIError(f"{type(exc).__name__} calling {model}") from None

    if res.status_code == 200:
        return res.json()

    message = f"HTTP {res.status_code} from {model}: {res.text[:600]}"
    # A 429 asking for a long wait is a quota wall (e.g. free-tier daily limit): retrying is pointless.
    if res.status_code in (429, 500, 502, 503, 504) and _retry_delay_s(res.text) <= 30:
        raise RetryableAPIError(message, status=res.status_code)
    raise APIError(message, status=res.status_code)


def timed_call(fn: Callable[[], Any], max_retries: int) -> Tuple[Any, Optional[str], Dict[str, Any]]:
    """Run fn with retries on transient errors. Never raises.

    Returns (result, error_message, stats) where stats has attempts, last_attempt_s
    (latency of the final attempt) and wall_s (total incl. backoff sleeps).
    """
    stats: Dict[str, Any] = {"attempts": 0, "last_attempt_s": 0.0, "wall_s": 0.0}

    def _once() -> Any:
        stats["attempts"] += 1
        t = time.perf_counter()
        try:
            return fn()
        finally:
            stats["last_attempt_s"] = time.perf_counter() - t

    t0 = time.perf_counter()
    try:
        out = with_retries(_once, max_retries, 4.0, (RetryableAPIError,))
        err = None
    except Exception as exc:  # noqa: BLE001 - we want every failure recorded, not raised
        out, err = None, f"{type(exc).__name__}: {exc}"
    stats["wall_s"] = time.perf_counter() - t0
    return out, err, stats


# --------------------------------------------------------------------------- #
# Step 1: translation
# --------------------------------------------------------------------------- #
def build_translation_prompt(text: str, source_name: str, lang: str) -> str:
    meta = LANGUAGES[lang]
    return (
        f"Translate the following {source_name} voice-over narration line into {meta['name']}.\n"
        "Rules:\n"
        f"- It will be read aloud by a text-to-speech voice, so write natural spoken {meta['name']}, "
        "not a word-for-word calque.\n"
        "- Keep the meaning, the warm reflective tone and roughly the same sentence structure.\n"
        f"- Use {meta['orthography']}. Do not romanize or transliterate the source script.\n"
        "- Keep quoted speech inside quotation marks.\n"
        "- Output ONLY the translated line: no notes, no alternatives, no quotes around the whole line.\n\n"
        f"{source_name} line:\n{text}"
    )


def _extract_text(data: Dict[str, Any]) -> str:
    cands = data.get("candidates") or []
    if not cands:
        raise APIError(f"No candidates returned. promptFeedback={json.dumps(data.get('promptFeedback'))[:300]}")
    parts = (cands[0].get("content") or {}).get("parts") or []
    text = "".join(p.get("text", "") for p in parts).strip()
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
    if not text:
        raise APIError(f"Empty translation. finishReason={cands[0].get('finishReason')}")
    return text


def translate_shots(
    shots: List[Dict[str, Any]],
    langs: List[str],
    source_lang: str,
    api_key: str,
    model: str,
    path: str,
    log_path: str,
    force: bool,
    sleep_s: float,
) -> Dict[str, Any]:
    doc = load_json(path, {})
    doc.setdefault("translations", {})
    doc.update({"source_language": source_lang, "translate_model": model, "updated_at": now_iso()})
    source_name = SOURCE_NAMES.get(source_lang, source_lang)

    for lang in langs:
        bucket = doc["translations"].setdefault(lang, {})
        for shot in shots:
            sid, src = shot["id"], shot["narration_text"]
            h = compute_input_hash(
                {"model": model, "lang": lang, "text": src, "prompt_version": PROMPT_VERSION}
            )
            prev = bucket.get(sid)
            if not force and prev and prev.get("status") == "ok" and prev.get("input_hash") == h:
                print(f"[CACHE HIT] translate {lang}/{sid}")
                continue

            payload = {
                "contents": [{"parts": [{"text": build_translation_prompt(src, source_name, lang)}]}],
                "generationConfig": {"temperature": 0.2},
            }
            print(f"[TRANSLATE] {sid} -> {LANGUAGES[lang]['name']} ({model})...")
            out, err, st = timed_call(lambda: _extract_text(_post(model, payload, api_key, 60)), 2)

            entry: Dict[str, Any] = {
                "source_text": src,
                "translated_text": out,
                "input_hash": h,
                "status": "ok" if out else "error",
                "error": err,
                "attempts": st["attempts"],
                "latency_s": round(st["last_attempt_s"], 3),
                "script_ok": None,
            }
            if out:
                eth = ethiopic_ratio(out)
                entry["script_ok"] = eth > 0.9 if LANGUAGES[lang]["script"] == "ethiopic" else eth < 0.1
                if not entry["script_ok"]:
                    print(f"  [WARN] {lang}/{sid}: unexpected script mix (Ethiopic ratio {eth:.2f})")
            bucket[sid] = entry
            save_json(path, doc)  # incremental: partial progress survives a crash

            log_run(
                stage=TRANSLATE_STAGE,
                shot_id=sid,
                model=model,
                params={"language": lang, "source_language": source_lang, "attempts": st["attempts"]},
                seed=None,
                latency_s=st["last_attempt_s"],
                status=entry["status"],
                error=err,
                log_path=log_path,
            )
            print(f"  [{'+' if out else '!'}] {out if out else err}")
            time.sleep(sleep_s)

    save_json(path, doc)
    return doc


# --------------------------------------------------------------------------- #
# Step 2: TTS
# --------------------------------------------------------------------------- #
def _extract_audio(data: Dict[str, Any]) -> Tuple[bytes, str]:
    cands = data.get("candidates") or []
    if not cands:
        raise APIError(f"No candidates returned. promptFeedback={json.dumps(data.get('promptFeedback'))[:300]}")
    for part in (cands[0].get("content") or {}).get("parts") or []:
        inline = part.get("inlineData") or part.get("inline_data")
        if inline and inline.get("data"):
            return base64.b64decode(inline["data"]), inline.get("mimeType", "audio/pcm;rate=24000")
    raise APIError(f"HTTP 200 but no audio part. finishReason={cands[0].get('finishReason')}")


def tts_shot(
    lang: str,
    shot_id: str,
    text: str,
    voice: str,
    model: str,
    api_key: str,
    out_dir: str,
    log_path: str,
    am_duration_s: Optional[float],
    force: bool,
    prior: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    wav_path = os.path.join(out_dir, lang, f"{shot_id}.wav")
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    h = compute_input_hash({"model": model, "voice": voice, "lang": lang, "text": text})

    if not force and prior and prior.get("input_hash") == h and prior.get("status") == "ok" and check_cache(wav_path, h):
        print(f"[CACHE HIT] tts {lang}/{shot_id} (reusing earlier measurement)")
        return prior

    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}},
        },
    }
    print(f"[TTS] {lang}/{shot_id} ({model}, voice={voice})...")
    out, err, st = timed_call(lambda: _extract_audio(_post(model, payload, api_key, 90)), 2)

    words = count_words(text)
    rec: Dict[str, Any] = {
        "language": lang,
        "shot_id": shot_id,
        "text": text,
        "words": words,
        "model": model,
        "voice": voice,
        "input_hash": h,
        "status": "error",
        "error": err,
        "attempts": st["attempts"],
        "latency_s": round(st["last_attempt_s"], 3),
        "wall_s": round(st["wall_s"], 3),
        "duration_s": None,
        "words_per_s": None,
        "duration_vs_am": None,
        "suspect": False,
        "suspect_reason": None,
        "audio_path": None,
        "measured_at": now_iso(),
    }

    if out is not None:
        raw, mime = out
        try:
            with open(wav_path, "wb") as f:
                f.write(ensure_wav_format(raw, mime_type=mime))
            dur = probe_audio_duration(wav_path)
            rec.update(
                status="ok",
                error=None,
                duration_s=round(dur, 3),
                words_per_s=round(words / dur, 3) if dur > 0 else None,
                audio_path=os.path.relpath(wav_path, REPO_ROOT),
            )
            if am_duration_s:
                ratio = dur / am_duration_s
                rec["duration_vs_am"] = round(ratio, 3)
                lo, hi = SUSPECT_RATIO_RANGE
                if dur <= 0.3:
                    rec.update(suspect=True, suspect_reason="near-empty audio")
                elif not lo <= ratio <= hi:
                    rec.update(suspect=True, suspect_reason=f"duration {ratio:.2f}x the Amharic take")
            write_cache_meta(wav_path, h, extra={"duration_s": rec["duration_s"], "model": model, "lang": lang})
        except Exception as exc:  # noqa: BLE001
            rec.update(status="error", error=f"post-processing failed: {type(exc).__name__}: {exc}")

    log_run(
        stage=STAGE,
        shot_id=shot_id,
        model=model,
        params={
            "language": lang,
            "voice": voice,
            "text": text,
            "words": words,
            "attempts": rec["attempts"],
            "wall_s": rec["wall_s"],
            "duration_s": rec["duration_s"],
            "words_per_s": rec["words_per_s"],
            "suspect": rec["suspect"],
            "suspect_reason": rec["suspect_reason"],
        },
        seed=None,
        latency_s=st["last_attempt_s"],
        status=rec["status"],
        error=rec["error"],
        log_path=log_path,
    )
    if rec["status"] == "ok":
        flag = f"  [SUSPECT: {rec['suspect_reason']}]" if rec["suspect"] else ""
        print(f"  [+] {rec['duration_s']:.2f}s audio, {rec['words_per_s']} w/s, latency {rec['latency_s']:.2f}s{flag}")
    else:
        print(f"  [!] {rec['error']}")
    return rec


# --------------------------------------------------------------------------- #
# Step 3: baseline + report
# --------------------------------------------------------------------------- #
def load_baseline(storyboard: Dict[str, Any], shots: List[Dict[str, Any]], log_path: str) -> Dict[str, Any]:
    """Amharic numbers for the same shots, from outputs/narration/*.meta.json and runs.jsonl."""
    per_shot: Dict[str, Any] = {}
    for shot in shots:
        sid = shot["id"]
        wav = os.path.join(REPO_ROOT, "outputs", "narration", f"{sid}.wav")
        meta = load_json(f"{wav}.meta.json", {})
        dur = meta.get("duration_s")
        if not dur and os.path.exists(wav):
            try:
                dur = probe_audio_duration(wav)
            except Exception:
                dur = None
        words = count_words(shot["narration_text"])
        per_shot[sid] = {
            "words": words,
            "duration_s": dur,
            "words_per_s": round(words / dur, 3) if dur else None,
            "latency_s": meta.get("latency_s"),
            "model": meta.get("model"),
        }

    ok = err = 0
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("stage") == "narration" and not r.get("cache_hit"):
                    if r.get("status") == "ok":
                        ok += 1
                    elif r.get("status") == "error":
                        err += 1
    return {"per_shot": per_shot, "errors": err, "total": ok + err}


def _f(x: Any, nd: int = 2) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _mean(xs: List[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _score_cell(scores: Dict[str, Any], lang: str, shot_ids: List[str]) -> str:
    vals = [scores.get(lang, {}).get(s) for s in shot_ids]
    rated = [v for v in vals if isinstance(v, (int, float))]
    return f"{sum(rated) / len(rated):.1f} ({len(rated)}/{len(shot_ids)})" if rated else "TBD"


def build_report(
    results: Dict[str, Any],
    baseline: Dict[str, Any],
    scores: Dict[str, Any],
    shot_ids: List[str],
    source_lang: str,
    translate_model: str,
    tts_model: str,
) -> str:
    base = baseline["per_shot"]
    base_dur = [base[s]["duration_s"] for s in shot_ids if base[s]["duration_s"]]
    base_words = sum(base[s]["words"] for s in shot_ids if base[s]["duration_s"])
    base_wps = base_words / sum(base_dur) if base_dur else None
    n_err, n_tot = baseline["errors"], baseline["total"]

    lines: List[str] = [
        f"### Ceiling test: multilingual TTS ({tts_model})",
        "",
        f"Shots: {', '.join(shot_ids)} | translation: `{translate_model}` | TTS: `{tts_model}` | "
        f"baseline: {SOURCE_NAMES.get(source_lang, source_lang)} (`{source_lang}`)",
        "",
        "| Language | Shots OK | Error rate | Suspect audio | Mean latency (s) | Total audio (s) | Words/sec | Duration vs am | Native intelligibility (1–5) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    base_err = f"{n_err}/{n_tot} ({100 * n_err / n_tot:.0f}%)" if n_tot else "n/a"
    lines.append(
        f"| {SOURCE_NAMES.get(source_lang, source_lang)} (baseline) | {len(base_dur)}/{len(shot_ids)} | {base_err} | — | "
        f"{_f(_mean([base[s]['latency_s'] for s in shot_ids]))} | {_f(sum(base_dur))} | {_f(base_wps)} | 1.00 | "
        f"{_score_cell(scores, source_lang, shot_ids)} |"
    )

    detail: List[str] = [
        "",
        "Per shot:",
        "",
        "| Language | Shot | Words | Audio (s) | Words/sec | Latency (s) | Attempts | Status | Intelligibility (1–5) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s in shot_ids:
        b = base[s]
        detail.append(
            f"| {source_lang} | {s} | {b['words']} | {_f(b['duration_s'])} | {_f(b['words_per_s'])} | "
            f"{_f(b['latency_s'], 1)} | — | baseline | {scores.get(source_lang, {}).get(s) or 'TBD'} |"
        )

    for lang, name in ((l, LANGUAGES[l]["name"]) for l in LANGUAGES if l in results):
        recs = [results[lang][s] for s in shot_ids if s in results[lang]]
        if not recs:
            continue
        ok = [r for r in recs if r["status"] == "ok"]
        failed = len(recs) - len(ok)
        suspect = sum(1 for r in ok if r.get("suspect"))
        tot_dur = sum(r["duration_s"] for r in ok)
        tot_words = sum(r["words"] for r in ok)
        wps = tot_words / tot_dur if tot_dur else None
        lines.append(
            f"| {name} (`{lang}`) | {len(ok)}/{len(shot_ids)} | {failed}/{len(recs)} ({100 * failed / len(recs):.0f}%) | "
            f"{suspect}/{len(ok)} | {_f(_mean([r['latency_s'] for r in ok]))} | {_f(tot_dur if ok else None)} | "
            f"{_f(wps)} | {_f(_mean([r['duration_vs_am'] for r in ok]))} | {_score_cell(scores, lang, shot_ids)} |"
        )
        for s in shot_ids:
            r = results[lang].get(s)
            if not r:
                continue
            status = r["status"] + ("*" if r.get("suspect") else "")
            if r["status"] == "error":
                status = "error"
            detail.append(
                f"| {lang} | {s} | {r['words']} | {_f(r['duration_s'])} | {_f(r['words_per_s'])} | "
                f"{_f(r['latency_s'], 1)} | {r['attempts']} | {status} | {scores.get(lang, {}).get(s) or 'TBD'} |"
            )

    errors = [(l, s, r["error"]) for l in results for s, r in results[l].items() if r["status"] == "error"]
    notes = [
        "",
        "Notes:",
        "",
        "- `*` = HTTP 200 but audio duration is outside "
        f"{SUSPECT_RATIO_RANGE[0]}–{SUSPECT_RATIO_RANGE[1]}x the Amharic take for the same shot (possible truncated or garbled speech). "
        "Treat it as a failure unless a native speaker says otherwise.",
        "- Words/sec is a rough comparison only: Amharic and Tigrinya (Ethiopic script, Semitic) pack more meaning into each word "
        "than Somali and Afaan Oromo (Latin script, agglutinative). Use *Duration vs am* for what matters to the film's timing.",
        "- Latency is the final successful call only (no retry backoff). The Amharic baseline latency comes from the earlier "
        "`narration.py` run, which could include fallback-model attempts, so treat the comparison as approximate.",
        "- Amharic error rate is over every non-cached `narration` entry in `runs.jsonl`; the ceiling rows are only these shots.",
        "- The TTS API takes no language parameter: the model infers the language from the text. "
        "The same voice as the Amharic baseline is used for every language.",
        "- Intelligibility is a human score (1 = unintelligible, 5 = fully natural/intelligible). `TBD` = not yet rated; "
        "fill `outputs/ceiling/intelligibility.json` and re-run with `--report-only`.",
    ]
    if errors:
        notes += ["", "Errors:", ""] + [f"- `{l}/{s}`: {str(e)[:220]}" for l, s, e in errors]

    return "\n".join(lines + detail + notes) + "\n"


def ensure_scores_template(path: str, langs: List[str], shot_ids: List[str], source_lang: str) -> Dict[str, Any]:
    scores = load_json(path, {})
    changed = not os.path.exists(path)
    for lang in [source_lang] + langs:
        bucket = scores.setdefault(lang, {})
        for s in shot_ids:
            if s not in bucket:
                bucket[s] = None
                changed = True
    if changed:
        save_json(path, scores)
    return scores


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> int:
    load_dotenv(os.path.join(REPO_ROOT, ".env"))
    p = argparse.ArgumentParser(description="Gemini 3.8 Flash TTS ceiling test beyond the primary language.")
    p.add_argument("--langs", default=",".join(LANGUAGES), help="Comma list from: " + ",".join(LANGUAGES))
    p.add_argument("--shots", default=",".join(DEFAULT_SHOTS))
    p.add_argument("--storyboard", default=os.path.join(REPO_ROOT, "storyboard.json"))
    p.add_argument("--schema", default=os.path.join(REPO_ROOT, "schema", "storyboard.schema.json"))
    p.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "outputs", "ceiling"))
    p.add_argument("--log-path", default=os.path.join(REPO_ROOT, "runs.jsonl"))
    p.add_argument("--translate-model", default=DEFAULT_TRANSLATE_MODEL)
    p.add_argument("--tts-model", default=DEFAULT_TTS_MODEL)
    p.add_argument("--voice", default=None, help="Override voice (default: each shot's tts_voice, as in the baseline).")
    p.add_argument("--sleep", type=float, default=2.0, help="Pause between API calls (rate-limit courtesy).")
    p.add_argument("--force", action="store_true", help="Ignore caches and redo translation + TTS.")
    p.add_argument("--translate-only", action="store_true")
    p.add_argument("--report-only", action="store_true", help="Skip API calls; rebuild summary.md from saved results.")
    p.add_argument("--dry-run", action="store_true", help="Print the plan and exit; no API calls, no writes.")
    args = p.parse_args()

    langs = [l.strip() for l in args.langs.split(",") if l.strip()]
    bad = [l for l in langs if l not in LANGUAGES]
    if bad:
        print(f"[FATAL] Unknown language code(s): {bad}. Choose from {list(LANGUAGES)}.", file=sys.stderr)
        return 2
    shot_ids = [s.strip() for s in args.shots.split(",") if s.strip()]

    try:
        storyboard = validate_storyboard(args.storyboard, args.schema)
    except Exception as exc:  # noqa: BLE001
        print(f"[FATAL] Storyboard validation failed: {exc}", file=sys.stderr)
        return 1
    source_lang = storyboard.get("project", {}).get("language", "am")
    by_id = {s["id"]: s for s in storyboard["shots"]}
    missing = [s for s in shot_ids if s not in by_id]
    if missing:
        print(f"[FATAL] Shot(s) not in storyboard: {missing}", file=sys.stderr)
        return 1
    shots = [by_id[s] for s in shot_ids]

    translations_path = os.path.join(args.out_dir, "translations.json")
    results_path = os.path.join(args.out_dir, "results.json")
    scores_path = os.path.join(args.out_dir, "intelligibility.json")
    summary_path = os.path.join(args.out_dir, "summary.md")

    if args.dry_run:
        print(f"Source language: {source_lang} | shots: {shot_ids}")
        print(f"Translate model: {args.translate_model} | TTS model: {args.tts_model} (no fallback chain)")
        for l in langs:
            print(f"  {l} ({LANGUAGES[l]['name']}): {len(shots)} translations + {len(shots)} TTS calls "
                  f"-> {args.out_dir}/{l}/<shot>.wav")
        print(f"Outputs: {translations_path}, {results_path}, {summary_path}; log -> {args.log_path} (stage='{STAGE}')")
        return 0

    results_doc = load_json(results_path, {"results": {}})
    results: Dict[str, Dict[str, Any]] = results_doc.setdefault("results", {})

    if not args.report_only:
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            print("[FATAL] GEMINI_API_KEY is not set (.env or environment).", file=sys.stderr)
            return 1

        trans = translate_shots(shots, langs, source_lang, api_key, args.translate_model,
                                translations_path, args.log_path, args.force, args.sleep)

        if not args.translate_only:
            baseline_dur = {
                s["id"]: load_json(os.path.join(REPO_ROOT, "outputs", "narration", f"{s['id']}.wav.meta.json"), {}).get("duration_s")
                for s in shots
            }
            for lang in langs:
                bucket = results.setdefault(lang, {})
                for shot in shots:
                    sid = shot["id"]
                    t = trans["translations"].get(lang, {}).get(sid, {})
                    if t.get("status") != "ok":
                        print(f"[SKIP] tts {lang}/{sid}: translation missing ({t.get('error')})")
                        continue
                    voice = args.voice or shot.get("tts_voice", "Puck")
                    bucket[sid] = tts_shot(lang, sid, t["translated_text"], voice, args.tts_model, api_key,
                                           args.out_dir, args.log_path, baseline_dur.get(sid),
                                           args.force, bucket.get(sid))
                    results_doc.update({"tts_model": args.tts_model, "updated_at": now_iso()})
                    save_json(results_path, results_doc)
                    time.sleep(args.sleep)

    scores = ensure_scores_template(scores_path, langs, shot_ids, source_lang)
    baseline = load_baseline(storyboard, shots, args.log_path)
    report_langs = {l: results[l] for l in LANGUAGES if l in results}
    report = build_report(report_langs, baseline, scores, shot_ids, source_lang,
                          args.translate_model, args.tts_model)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"Summary written to {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
