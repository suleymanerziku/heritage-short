"""Smoke test script for verifying zero-cost API stack in heritage-short pipeline.

Verifies:
1. Gemini API (Text Generation)
2. Gemini TTS / Speech Generation (Amharic greeting -> outputs/narration/smoke_test.wav)
3. Cloudflare Workers AI (FLUX image generation -> outputs/keyframes/smoke_test.png)

Records latencies and results to runs.jsonl using pipeline.utils.log_run.
"""

from __future__ import annotations

import base64
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import requests
from dotenv import load_dotenv
from PIL import Image

# Import shared utilities
from pipeline.utils import log_run

# Optional SDK import with graceful REST fallback
try:
    from google import genai
    from google.genai import types

    HAS_GENAI_SDK = True
except Exception as sdk_err:
    HAS_GENAI_SDK = False
    GENAI_SDK_ERR = str(sdk_err)


def run_gemini_text_smoke(api_key: str) -> Tuple[bool, str, float, Optional[str]]:
    """Tests Gemini text generation API.

    Candidate models tested in order:
    - gemini-3.8-flash
    - gemini-2.5-flash
    - gemini-2.0-flash
    - gemini-1.5-flash

    Returns:
        (success, model_used, latency_s, error_message)
    """
    candidate_models = [
        "gemini-3.8-flash",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash",
    ]
    prompt = "State one historical fact about the rock-hewn churches of Lalibela in one concise sentence."
    last_error: Optional[str] = None

    for model in candidate_models:
        print(f"[*] Testing Gemini Text API with model: '{model}'...")
        start_time = time.time()
        try:
            # 1. Try with google-genai SDK if available
            if HAS_GENAI_SDK:
                try:
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=model,
                        contents=prompt,
                    )
                    text = response.text.strip() if response and response.text else ""
                    if text:
                        latency = time.time() - start_time
                        print(f"  [+] Success via SDK ({model}) in {latency:.2f}s:")
                        print(f"      \"{text}\"")
                        log_run(
                            stage="smoke_test",
                            shot_id="smoke_text",
                            model=model,
                            params={"prompt": prompt, "method": "sdk"},
                            seed=None,
                            latency_s=latency,
                            status="ok",
                            est_cost_usd=0.0,
                        )
                        return True, model, latency, None
                except Exception as sdk_call_err:
                    last_error = f"SDK call failed for {model}: {sdk_call_err}"

            # 2. REST API fallback
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=30)
            latency = time.time() - start_time

            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        output_text = parts[0]["text"].strip()
                        print(f"  [+] Success via REST ({model}) in {latency:.2f}s:")
                        print(f"      \"{output_text}\"")
                        log_run(
                            stage="smoke_test",
                            shot_id="smoke_text",
                            model=model,
                            params={"prompt": prompt, "method": "rest"},
                            seed=None,
                            latency_s=latency,
                            status="ok",
                            est_cost_usd=0.0,
                        )
                        return True, model, latency, None

            error_msg = f"HTTP {res.status_code}: {res.text}"
            last_error = error_msg
            print(f"  [-] Model '{model}' unavailable: {error_msg}")

        except Exception as exc:
            latency = time.time() - start_time
            last_error = f"Exception: {exc}"
            print(f"  [-] Error testing '{model}': {exc}")

    # All models failed
    log_run(
        stage="smoke_test",
        shot_id="smoke_text",
        model=candidate_models[0],
        params={"prompt": prompt},
        seed=None,
        latency_s=0.0,
        status="error",
        error=last_error,
        est_cost_usd=0.0,
    )
    return False, candidate_models[0], 0.0, last_error


def run_gemini_tts_smoke(api_key: str) -> Tuple[bool, str, float, Optional[str]]:
    """Tests Gemini Speech / TTS API with an Amharic greeting.

    Saves output to outputs/narration/smoke_test.wav and verifies file size > 0.

    Candidate models tested:
    - gemini-3.8-flash-tts
    - gemini-2.5-flash-tts
    - gemini-2.5-flash
    - gemini-2.0-flash

    Returns:
        (success, model_used, latency_s, error_message)
    """
    output_path = os.path.join("outputs", "narration", "smoke_test.wav")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    amharic_greeting = "ጤና ይስጥልኝ፣ ይህ ለቅርስ አጭር ፊልም ፕሮጀክት የተዘጋጀ የመጀመሪያው የድምፅ ሙከራ ነው።"
    candidate_models = [
        "gemini-3.8-flash-tts",
        "gemini-2.5-flash-tts",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
    ]
    last_error: Optional[str] = None

    for model in candidate_models:
        print(f"[*] Testing Gemini TTS with model: '{model}'...")
        start_time = time.time()

        # Try REST API with audio responseModality
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {
                "parts": [
                    {
                        "text": "You are a professional voice actor. Speak only the user's provided text verbatim with natural pacing. Do not read instructions or add introductory remarks."
                    }
                ]
            },
            "contents": [
                {
                    "parts": [
                        {
                            "text": amharic_greeting
                        }
                    ]
                }
            ],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {
                            "voiceName": "Puck"
                        }
                    }
                }
            }
        }

        try:
            res = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=45)
            latency = time.time() - start_time

            if res.status_code == 200:
                data = res.json()
                candidates = data.get("candidates", [])
                audio_bytes: Optional[bytes] = None

                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    for part in parts:
                        inline_data = part.get("inlineData") or part.get("inline_data")
                        if inline_data and "data" in inline_data:
                            audio_bytes = base64.b64decode(inline_data["data"])
                            break

                if audio_bytes and len(audio_bytes) > 0:
                    with open(output_path, "wb") as f:
                        f.write(audio_bytes)

                    file_size = os.path.getsize(output_path)
                    print(f"  [+] TTS audio saved to '{output_path}' ({file_size} bytes, {latency:.2f}s)")
                    log_run(
                        stage="smoke_test",
                        shot_id="smoke_tts",
                        model=model,
                        params={"text": amharic_greeting, "output_path": output_path},
                        seed=None,
                        latency_s=latency,
                        status="ok",
                        est_cost_usd=0.0,
                    )
                    return True, model, latency, None

                last_error = f"HTTP 200 returned but no audio payload found in candidates: {data}"
                print(f"  [-] {last_error}")
            else:
                last_error = f"HTTP {res.status_code}: {res.text}"
                print(f"  [-] TTS failed on model '{model}': {last_error}")

        except Exception as exc:
            latency = time.time() - start_time
            last_error = f"Exception: {exc}"
            print(f"  [-] Error testing TTS with '{model}': {exc}")

    log_run(
        stage="smoke_test",
        shot_id="smoke_tts",
        model=candidate_models[0],
        params={"text": amharic_greeting},
        seed=None,
        latency_s=0.0,
        status="error",
        error=last_error,
        est_cost_usd=0.0,
    )
    return False, candidate_models[0], 0.0, last_error


def run_cloudflare_flux_smoke(account_id: str, api_token: str) -> Tuple[bool, str, float, Optional[str]]:
    """Tests Cloudflare Workers AI image generation with FLUX.

    Saves output to outputs/keyframes/smoke_test.png and verifies image validity.

    Candidate models:
    - @cf/black-forest-labs/flux-2-klein-4b
    - @cf/black-forest-labs/flux-1-schnell

    Returns:
        (success, model_used, latency_s, error_message)
    """
    output_path = os.path.join("outputs", "keyframes", "smoke_test.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    candidate_models = [
        "@cf/black-forest-labs/flux-2-klein-4b",
        "@cf/black-forest-labs/flux-1-schnell",
    ]
    prompt = "A majestic stone-hewn church in northern Ethiopia at golden hour, cinematic lighting, photorealistic 8k."
    last_error: Optional[str] = None

    for model in candidate_models:
        print(f"[*] Testing Cloudflare Workers AI with model: '{model}'...")
        start_time = time.time()

        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
        headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "prompt": prompt,
        }

        try:
            res = requests.post(url, headers=headers, json=payload, timeout=60)
            latency = time.time() - start_time

            if res.status_code == 200:
                image_bytes: Optional[bytes] = None
                content_type = res.headers.get("Content-Type", "")

                if "image" in content_type or "octet-stream" in content_type:
                    image_bytes = res.content
                else:
                    try:
                        data = res.json()
                        if "result" in data and isinstance(data["result"], dict) and "image" in data["result"]:
                            image_bytes = base64.b64decode(data["result"]["image"])
                        elif "result" in data and isinstance(data["result"], str):
                            image_bytes = base64.b64decode(data["result"])
                    except Exception:
                        image_bytes = res.content

                if image_bytes and len(image_bytes) > 0:
                    with open(output_path, "wb") as f:
                        f.write(image_bytes)

                    # Verify that image is valid via Pillow
                    try:
                        with Image.open(output_path) as img:
                            img.verify()
                        print(f"  [+] Keyframe image successfully generated and verified at '{output_path}' ({latency:.2f}s)")
                        log_run(
                            stage="smoke_test",
                            shot_id="smoke_keyframe",
                            model=model,
                            params={"prompt": prompt, "output_path": output_path},
                            seed=None,
                            latency_s=latency,
                            status="ok",
                            est_cost_usd=0.0,
                        )
                        return True, model, latency, None
                    except Exception as img_err:
                        last_error = f"File written but Pillow verification failed: {img_err}"
                        print(f"  [-] {last_error}")
                else:
                    last_error = "HTTP 200 returned but response contained empty image data"
                    print(f"  [-] {last_error}")
            else:
                last_error = f"HTTP {res.status_code}: {res.text}"
                print(f"  [-] Cloudflare Workers AI call failed on '{model}': {last_error}")

        except Exception as exc:
            latency = time.time() - start_time
            last_error = f"Exception: {exc}"
            print(f"  [-] Error testing Cloudflare model '{model}': {exc}")

    log_run(
        stage="smoke_test",
        shot_id="smoke_keyframe",
        model=candidate_models[0],
        params={"prompt": prompt},
        seed=None,
        latency_s=0.0,
        status="error",
        error=last_error,
        est_cost_usd=0.0,
    )
    return False, candidate_models[0], 0.0, last_error


def print_summary_table(results: List[Dict[str, Any]]) -> None:
    """Prints a structured PASS/FAIL summary table for all smoke tests."""
    print("\n" + "=" * 80)
    print(f"{'SMOKE TEST RESULTS SUMMARY':^80}")
    print("=" * 80)
    print(f"{'Provider / Service':<26} {'Model Used':<34} {'Status':<8} {'Latency':<10}")
    print("-" * 80)

    passed_count = 0
    for r in results:
        status_str = "PASS" if r["passed"] else "FAIL"
        if r["passed"]:
            passed_count += 1
        latency_str = f"{r['latency']:.2f}s" if r["latency"] > 0 else "N/A"
        print(f"{r['service']:<26} {r['model']:<34} {status_str:<8} {latency_str:<10}")
        if not r["passed"] and r.get("error"):
            # Indent and print exact error for documentation in FAILURES.md
            print(f"   └──> Error: {r['error']}")

    print("=" * 80)
    print(f"Total: {len(results)} | Passed: {passed_count} | Failed: {len(results) - passed_count}")
    print("=" * 80 + "\n")


def main() -> int:
    """Main smoke test orchestration."""
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()
    cf_account_id = os.getenv("CF_ACCOUNT_ID", "").strip()
    cf_api_token = os.getenv("CF_API_TOKEN", "").strip()

    print("=" * 80)
    print("Starting heritage-short Zero-Cost API Stack Smoke Test")
    print("=" * 80)

    summary_records: List[Dict[str, Any]] = []

    # 1. Gemini Text API
    print("\n[Stage 1/3] Gemini Text Generation API")
    if not gemini_api_key:
        err = "GEMINI_API_KEY is missing or empty in .env"
        print(f"  [-] SKIPPED: {err}")
        log_run(
            stage="smoke_test",
            shot_id="smoke_text",
            model="gemini-flash",
            params={},
            seed=None,
            latency_s=0.0,
            status="error",
            error=err,
        )
        summary_records.append({
            "service": "Gemini Text Generation",
            "model": "gemini-flash",
            "passed": False,
            "latency": 0.0,
            "error": err,
        })
    else:
        try:
            ok, model_used, latency, error = run_gemini_text_smoke(gemini_api_key)
            summary_records.append({
                "service": "Gemini Text Generation",
                "model": model_used,
                "passed": ok,
                "latency": latency,
                "error": error,
            })
        except Exception as e:
            print(f"  [-] Unhandled exception in Gemini Text: {e}")
            summary_records.append({
                "service": "Gemini Text Generation",
                "model": "unknown",
                "passed": False,
                "latency": 0.0,
                "error": str(e),
            })

    # 2. Gemini TTS Speech API
    print("\n[Stage 2/3] Gemini TTS / Speech Generation API (Amharic)")
    if not gemini_api_key:
        err = "GEMINI_API_KEY is missing or empty in .env"
        print(f"  [-] SKIPPED: {err}")
        log_run(
            stage="smoke_test",
            shot_id="smoke_tts",
            model="gemini-tts",
            params={},
            seed=None,
            latency_s=0.0,
            status="error",
            error=err,
        )
        summary_records.append({
            "service": "Gemini TTS Audio",
            "model": "gemini-tts",
            "passed": False,
            "latency": 0.0,
            "error": err,
        })
    else:
        try:
            ok, model_used, latency, error = run_gemini_tts_smoke(gemini_api_key)
            summary_records.append({
                "service": "Gemini TTS Audio",
                "model": model_used,
                "passed": ok,
                "latency": latency,
                "error": error,
            })
        except Exception as e:
            print(f"  [-] Unhandled exception in Gemini TTS: {e}")
            summary_records.append({
                "service": "Gemini TTS Audio",
                "model": "unknown",
                "passed": False,
                "latency": 0.0,
                "error": str(e),
            })

    # 3. Cloudflare Workers AI FLUX Keyframe Generation
    print("\n[Stage 3/3] Cloudflare Workers AI FLUX Image Generation")
    if not cf_account_id or not cf_api_token:
        err = "CF_ACCOUNT_ID or CF_API_TOKEN is missing or empty in .env"
        print(f"  [-] SKIPPED: {err}")
        log_run(
            stage="smoke_test",
            shot_id="smoke_keyframe",
            model="cf-flux",
            params={},
            seed=None,
            latency_s=0.0,
            status="error",
            error=err,
        )
        summary_records.append({
            "service": "Cloudflare Workers AI",
            "model": "cf-flux",
            "passed": False,
            "latency": 0.0,
            "error": err,
        })
    else:
        try:
            ok, model_used, latency, error = run_cloudflare_flux_smoke(cf_account_id, cf_api_token)
            summary_records.append({
                "service": "Cloudflare Workers AI",
                "model": model_used,
                "passed": ok,
                "latency": latency,
                "error": error,
            })
        except Exception as e:
            print(f"  [-] Unhandled exception in Cloudflare FLUX: {e}")
            summary_records.append({
                "service": "Cloudflare Workers AI",
                "model": "unknown",
                "passed": False,
                "latency": 0.0,
                "error": str(e),
            })

    # Print summary table
    print_summary_table(summary_records)

    all_passed = all(r["passed"] for r in summary_records)
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
