"""Generates per-shot narration audio from storyboard.json using Gemini Speech / TTS.

Features:
- Validates storyboard.json against schema/storyboard.schema.json.
- Computes deterministic SHA-256 input hash for caching.
- Calls Gemini TTS with exponential retries.
- Saves audio as 24kHz/48kHz mono WAV.
- Probes exact audio duration using probe_audio_duration.
- Logs API calls and latencies to runs.jsonl.
- Prints structured summary table with duration warning (60-90s).
- Supports single-shot generation via --shot CLI flag.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import requests
from dotenv import load_dotenv

from pipeline.utils import (
    check_cache,
    compute_input_hash,
    log_run,
    probe_audio_duration,
    validate_storyboard,
    with_retries,
)

# Optional google-genai SDK
try:
    from google import genai
    from google.genai import types

    HAS_GENAI_SDK = True
except Exception:
    HAS_GENAI_SDK = False


def ensure_wav_format(audio_bytes: bytes, mime_type: str = "audio/wav") -> bytes:
    """Ensures audio data has a valid standard RIFF/WAVE header (24kHz/48kHz 16-bit mono)."""
    # If already a valid WAV container with RIFF/WAVE header, return as-is
    if len(audio_bytes) >= 12 and audio_bytes[:4] == b"RIFF" and audio_bytes[8:12] == b"WAVE":
        return audio_bytes

    # Parse sample rate from mimeType if available (e.g. "audio/pcm;rate=24000")
    sample_rate = 24000
    if "rate=" in mime_type:
        try:
            extracted_rate = int(mime_type.split("rate=")[1].split(";")[0].strip())
            if extracted_rate in (16000, 22050, 24000, 44100, 48000):
                sample_rate = extracted_rate
        except Exception:
            sample_rate = 24000

    # Wrap raw PCM in 16-bit mono WAV container
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)

    return buf.getvalue()


def call_gemini_tts(
    api_key: str,
    text: str,
    voice_name: str,
    tts_style: str,
    language: str,
) -> Tuple[bytes, str, str]:
    """Invokes Gemini speech generation API with fallback across candidate models.

    Candidate models:
    - gemini-3.8-flash-tts
    - gemini-3.8-flash-lite-tts
    - gemini-3.1-flash-tts-preview
    - gemini-3.8-flash
    - gemini-2.5-flash

    Returns:
        (raw_audio_bytes, mime_type, model_used)
    """
    candidate_models = [
        "gemini-3.8-flash-tts",
        "gemini-3.8-flash-lite-tts",
        "gemini-3.1-flash-tts-preview",
        "gemini-3.8-flash",
        "gemini-2.5-flash",
    ]

    last_error: Optional[str] = None

    for model in candidate_models:
        # 1. Try google-genai SDK if available
        if HAS_GENAI_SDK:
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model=model,
                    contents=text,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_name or "Puck"
                                )
                            )
                        ),
                    ),
                )
                if response and response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "inline_data") and part.inline_data:
                            data = part.inline_data.data
                            mime = getattr(part.inline_data, "mime_type", "audio/wav")
                            return data, mime, model
            except Exception as sdk_exc:
                last_error = f"SDK failed for {model}: {sdk_exc}"

        # 2. REST API execution (Pure text content without instruction wrappers)
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": text
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": voice_name or "Puck"
                        }
                    }
                }
            }
        }

        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)

            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        inline_data = part.get("inlineData") or part.get("inline_data")
                        if inline_data and "data" in inline_data:
                            raw_b64 = inline_data["data"]
                            audio_bytes = base64.b64decode(raw_b64)
                            mime = inline_data.get("mimeType", "audio/wav")
                            return audio_bytes, mime, model

                last_error = f"HTTP 200 from {model} but no inline audio data found: {data}"
            else:
                last_error = f"HTTP {res.status_code} from {model}: {res.text}"

        except Exception as exc:
            last_error = f"Exception connecting to {model}: {exc}"

    raise RuntimeError(f"All Gemini TTS candidate models failed. Last error: {last_error}")


def generate_single_narration(
    shot: Dict[str, Any],
    language: str,
    output_dir: str = "outputs/narration",
    api_key: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Generates and caches audio for a single shot."""
    shot_id = shot["id"]
    output_path = os.path.join(output_dir, f"{shot_id}.wav")
    os.makedirs(output_dir, exist_ok=True)

    narration_text = shot["narration_text"]
    tts_style = shot.get("tts_style", "warm, calm")
    tts_voice = shot.get("tts_voice", "Puck")
    pad_after_s = float(shot.get("pad_after_s", 0.4))

    # Compute deterministic cache hash
    input_payload = {
        "language": language,
        "narration_text": narration_text,
        "tts_style": tts_style,
        "tts_voice": tts_voice,
    }
    input_hash = compute_input_hash(input_payload)

    # Check cache
    if not force and check_cache(output_path, input_hash):
        meta_path = f"{output_path}.meta.json"
        duration_s = 0.0
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    duration_s = float(meta.get("duration_s", 0.0))
            except Exception:
                duration_s = 0.0

        if duration_s <= 0.0 and os.path.exists(output_path):
            duration_s = probe_audio_duration(output_path)

        print(f"[CACHE HIT] {shot_id:<8} | Duration: {duration_s:.2f}s | Path: {output_path}")
        return {
            "shot_id": shot_id,
            "duration_s": duration_s,
            "pad_after_s": pad_after_s,
            "total_shot_s": round(duration_s + pad_after_s, 2),
            "output_path": output_path,
            "cache_hit": True,
            "model": "cached",
        }

    # Cache miss - API invocation required
    if not api_key:
        raise ValueError(
            f"GEMINI_API_KEY is required to generate narration for {shot_id}. "
            "Please configure it in .env or your environment."
        )

    print(f"[GENERATING] {shot_id:<8} | Voice: {tts_voice} | Style: {tts_style}...")
    start_time = time.time()

    def _invoke() -> Tuple[bytes, str, str]:
        return call_gemini_tts(
            api_key=api_key,
            text=narration_text,
            voice_name=tts_voice,
            tts_style=tts_style,
            language=language,
        )

    raw_audio, mime_type, model_used = with_retries(_invoke, max_retries=3, base_delay=2.0)
    latency_s = time.time() - start_time

    # Standardize to 24kHz/48kHz WAV
    wav_bytes = ensure_wav_format(raw_audio, mime_type=mime_type)
    with open(output_path, "wb") as f:
        f.write(wav_bytes)

    # Probe duration
    duration_s = probe_audio_duration(output_path)

    # Write cache metadata
    from pipeline.utils import write_cache_meta

    write_cache_meta(
        output_path=output_path,
        input_hash=input_hash,
        extra={
            "duration_s": round(duration_s, 3),
            "pad_after_s": pad_after_s,
            "model": model_used,
            "sample_rate": 24000,
            "channels": 1,
            "latency_s": round(latency_s, 2),
        },
    )

    # Log to runs.jsonl
    log_run(
        stage="narration",
        shot_id=shot_id,
        model=model_used,
        params=input_payload,
        seed=shot.get("seed"),
        latency_s=latency_s,
        status="ok",
        est_cost_usd=0.0,
        cache_hit=False,
    )

    print(f"  [+] Saved: {output_path} ({duration_s:.2f}s, latency {latency_s:.2f}s via {model_used})")

    return {
        "shot_id": shot_id,
        "duration_s": duration_s,
        "pad_after_s": pad_after_s,
        "total_shot_s": round(duration_s + pad_after_s, 2),
        "output_path": output_path,
        "cache_hit": False,
        "model": model_used,
    }


def print_narration_summary(results: List[Dict[str, Any]]) -> None:
    """Renders structured summary table and checks total film runtime."""
    print("\n" + "=" * 80)
    print(f"{'NARRATION GENERATION SUMMARY':^80}")
    print("=" * 80)
    print(
        f"{'Shot ID':<10} {'Audio (s)':<12} {'Pad (s)':<10} {'Shot Total (s)':<16} "
        f"{'Source':<12} {'Output File':<20}"
    )
    print("-" * 80)

    total_narration_s = 0.0
    total_film_s = 0.0

    for r in results:
        dur = r["duration_s"]
        pad = r["pad_after_s"]
        shot_tot = r["total_shot_s"]
        total_narration_s += dur
        total_film_s += shot_tot
        source_str = "Cache Hit" if r["cache_hit"] else r.get("model", "API")

        print(
            f"{r['shot_id']:<10} {dur:<12.2f} {pad:<10.2f} {shot_tot:<16.2f} "
            f"{source_str:<12} {os.path.basename(r['output_path']):<20}"
        )

    print("=" * 80)
    print(f"Total Spoken Narration Duration: {total_narration_s:.2f} seconds")
    print(f"Total Estimated Film Runtime   : {total_film_s:.2f} seconds (narration + breathing padding)")
    print("=" * 80)

    # Runtime check (60 to 90 seconds)
    if total_film_s < 60.0:
        print(
            f"[WARNING] Total estimated runtime ({total_film_s:.1f}s) is below target minimum (60.0s). "
            "Consider slightly increasing shot pad_after_s or expanding narration text."
        )
    elif total_film_s > 90.0:
        print(
            f"[WARNING] Total estimated runtime ({total_film_s:.1f}s) exceeds target maximum (90.0s). "
            "Consider tightening narration text to keep the film under 90 seconds."
        )
    else:
        print(f"[OK] Total film runtime ({total_film_s:.1f}s) sits comfortably within the 60-90s target window!")
    print()


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    parser = argparse.ArgumentParser(description="Generate per-shot narration audio for heritage-short.")
    parser.add_argument(
        "--shot",
        type=str,
        default=None,
        help="Target a specific shot_id (e.g. shot_04) to generate or test.",
    )
    parser.add_argument(
        "--storyboard",
        type=str,
        default="storyboard.json",
        help="Path to storyboard.json file.",
    )
    parser.add_argument(
        "--schema",
        type=str,
        default="schema/storyboard.schema.json",
        help="Path to schema/storyboard.schema.json file.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass cache and force audio regeneration.",
    )
    args = parser.parse_args()

    # 1. Validate storyboard first
    print("Validating storyboard before narration generation...")
    try:
        storyboard = validate_storyboard(args.storyboard, args.schema)
    except Exception as val_err:
        print(f"[FATAL] Storyboard validation failed: {val_err}", file=sys.stderr)
        return 1

    language = storyboard.get("project", {}).get("language", "am")
    shots = storyboard.get("shots", [])

    # Filter for single shot if requested
    if args.shot:
        matched_shots = [s for s in shots if s["id"] == args.shot]
        if not matched_shots:
            print(f"[ERROR] Specified shot '{args.shot}' not found in {args.storyboard}.", file=sys.stderr)
            return 1
        shots_to_process = matched_shots
    else:
        shots_to_process = shots

    print(f"Processing narration for {len(shots_to_process)} shot(s) (Language: '{language}')...\n")

    results: List[Dict[str, Any]] = []
    has_errors = False

    for shot in shots_to_process:
        try:
            res = generate_single_narration(
                shot=shot,
                language=language,
                output_dir=os.path.join("outputs", "narration"),
                api_key=api_key,
                force=args.force,
            )
            results.append(res)
        except Exception as exc:
            has_errors = True
            print(f"[ERROR] Failed generating narration for {shot['id']}: {exc}", file=sys.stderr)
            log_run(
                stage="narration",
                shot_id=shot["id"],
                model="gemini-tts",
                params={"narration_text": shot.get("narration_text", "")},
                seed=shot.get("seed"),
                latency_s=0.0,
                status="error",
                error=str(exc),
                est_cost_usd=0.0,
            )

    if results:
        print_narration_summary(results)

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
