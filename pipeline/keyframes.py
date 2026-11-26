"""Generates keyframe images from storyboard.json using Cloudflare Workers AI or NVIDIA NIM API.

Supported Models:
- Cloudflare:
  - flux2: @cf/black-forest-labs/flux-2-klein-4b (distilled 4B parameter model)
  - flux9b: @cf/black-forest-labs/flux-2-klein-9b (9B parameter model)
  - flux1: @cf/black-forest-labs/flux-1-schnell (fast 4-step generation)
- NVIDIA NIM:
  - nvidia-flux3: black-forest-labs/flux-3-image-klein (latest state-of-the-art Klein)
  - nvidia-flux1-dev: black-forest-labs/flux.1-dev (full 12B guidance model)

Features:
- Validates storyboard.json against schema/storyboard.schema.json.
- Computes deterministic SHA-256 input hash for content-addressed caching.
- Checks existing cache before making any API calls ([CACHE HIT]).
- Supports --ref-sheet flag generating 4 candidate style/character reference images.
- Supports --shot <shot_id> and --model <model_alias> CLI flags for A/B testing.
- Logs all calls, latencies, and metadata to runs.jsonl.
- Automatically handles retry logic via with_retries().
"""

from __future__ import annotations

import argparse
import base64
import io
import json
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

from pipeline.utils import (
    check_cache,
    compute_input_hash,
    log_run,
    validate_storyboard,
    with_retries,
    write_cache_meta,
)

# Load environment variables
load_dotenv()

# Cloudflare model catalog mapping
CF_MODEL_MAP: Dict[str, str] = {
    "flux2": "@cf/black-forest-labs/flux-2-klein-4b",
    "flux9b": "@cf/black-forest-labs/flux-2-klein-9b",
    "flux1": "@cf/black-forest-labs/flux-1-schnell",
}

# NVIDIA model catalog mapping
NVIDIA_MODEL_MAP: Dict[str, str] = {
    "nvidia-flux3": "black-forest-labs/flux-3-image-klein",
    "nvidia-flux1-dev": "black-forest-labs/flux.1-dev",
}

# Gemini model catalog mapping
GEMINI_MODEL_MAP: Dict[str, str] = {
    "gemini-3.1-flash-lite-image": "gemini-3.1-flash-lite-image",
    "gemini-3.1-flash-image": "gemini-3.1-flash-image",
    "gemini-3.1-flash-image-preview": "gemini-3.1-flash-image-preview",
    "gemini-2.5-flash-image": "gemini-2.5-flash-image",
    "gemini-3-pro-image-preview": "gemini-3-pro-image-preview",
    "imagen3": "imagen-3.0-generate-002",
}

MODEL_MAP: Dict[str, str] = {**CF_MODEL_MAP, **NVIDIA_MODEL_MAP, **GEMINI_MODEL_MAP}


def get_cf_credentials() -> Tuple[str, str]:
    """Retrieves Cloudflare Account ID and API Token from environment variables."""
    account_id = os.getenv("CF_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.getenv("CF_API_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN")

    if not account_id:
        raise ValueError("Missing Cloudflare Account ID: Set CF_ACCOUNT_ID or CLOUDFLARE_ACCOUNT_ID in .env")
    if not api_token:
        raise ValueError("Missing Cloudflare API Token: Set CF_API_TOKEN or CLOUDFLARE_API_TOKEN in .env")

    return account_id, api_token


def get_nvidia_api_key() -> str:
    """Retrieves NVIDIA API Key from environment variables."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        raise ValueError("Missing NVIDIA API Key: Set NVIDIA_API_KEY in .env")
    return api_key


def get_gemini_api_key() -> str:
    """Retrieves Gemini API key from environment variables."""
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise ValueError("Missing Gemini API Key: Set GEMINI_API_KEY in .env")
    return key


def call_gemini_image(
    api_key: str,
    model_alias: str,
    prompt: str,
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 1024,
    timeout: int = 90,
) -> bytes:
    """Calls Gemini REST API for text-to-image generation.

    Supports:
    - imagen3: imagen-3.0-generate-002 (:predict endpoint)
    - gemini-3.1-flash-lite-image, gemini-3.1-flash-image, gemini-2.5-flash-image (:generateContent endpoint)
    """
    model_name = GEMINI_MODEL_MAP.get(model_alias, model_alias)

    # 1. Imagen 3 predict endpoint
    if "imagen" in model_name:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:predict?key={api_key}"
        payload: Dict[str, Any] = {
            "instances": [{"prompt": prompt}],
            "parameters": {
                "sampleCount": 1,
                "aspectRatio": "1:1",
                "outputOptions": {"mimeType": "image/png"},
            },
        }
        res = requests.post(url, json=payload, timeout=timeout)
        if res.status_code != 200:
            raise RuntimeError(f"Gemini Imagen API returned HTTP {res.status_code}: {res.text}")
        data = res.json()
        predictions = data.get("predictions", [])
        if predictions and "bytesBase64Encoded" in predictions[0]:
            return base64.b64decode(predictions[0]["bytesBase64Encoded"])
        raise ValueError(f"Unrecognized response structure from Imagen API: {data}")

    # 2. Gemini Multimodal generateContent endpoint
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
        },
    }
    if seed is not None:
        payload["generationConfig"]["seed"] = seed

    res = requests.post(url, json=payload, timeout=timeout)
    if res.status_code != 200:
        raise RuntimeError(f"Gemini API returned HTTP {res.status_code} for {model_name}: {res.text}")

    data = res.json()
    candidates = data.get("candidates", [])
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        for part in parts:
            if "inlineData" in part and "data" in part["inlineData"]:
                return base64.b64decode(part["inlineData"]["data"])

    raise ValueError(f"No inline image data found in Gemini response: {data}")


def call_nvidia_flux(
    api_key: str,
    model_alias: str,
    prompt: str,
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 1024,
    timeout: int = 120,
) -> bytes:
    """Calls NVIDIA NIM API for FLUX 3 Image Klein or FLUX.1 dev."""
    model_name = NVIDIA_MODEL_MAP.get(model_alias, model_alias)
    url = f"https://ai.api.nvidia.com/v1/genai/{model_name}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "width": width,
        "height": height,
    }

    if "flux.1-dev" in model_name:
        payload["mode"] = "base"
        payload["cfg_scale"] = 3.5
        payload["steps"] = 30
        if seed is not None:
            payload["seed"] = seed

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code != 200:
        raise RuntimeError(
            f"NVIDIA API returned HTTP {response.status_code} for {model_name}: {response.text}"
        )

    data = response.json()
    artifacts = data.get("artifacts") or []
    if not artifacts:
        raise ValueError(f"No artifacts from {model_name}: {str(data)[:500]}")
    art = artifacts[0]
    meta = {k: v for k, v in art.items() if k != "base64"}
    reason = str(art.get("finishReason") or art.get("finish_reason") or "").upper()
    if reason and reason not in ("SUCCESS", "STOP"):
        raise RuntimeError(f"{model_name} returned no usable image: {meta}")
    raw = base64.b64decode(art.get("base64") or "")
    if len(raw) < 1000:
        raise RuntimeError(f"{model_name} returned {len(raw)} bytes, not an image: {meta}")
    return raw


def call_cloudflare_flux(
    account_id: str,
    api_token: str,
    model_alias: str,
    prompt: str,
    seed: Optional[int] = None,
    width: int = 1024,
    height: int = 1024,
    timeout: int = 90,
) -> bytes:
    """Calls Cloudflare Workers AI FLUX text-to-image endpoint.

    Handles protocol differences:
    - flux2/flux9b: requires multipart/form-data
    - flux1: standard JSON body

    Returns:
        Decoded image bytes (PNG/JPEG).
    """
    model_name = CF_MODEL_MAP.get(model_alias, model_alias)
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model_name}"
    headers = {"Authorization": f"Bearer {api_token}"}

    if "flux-2" in model_name or model_alias in ("flux2", "flux9b"):
        # FLUX.2 klein expects multipart/form-data
        files: Dict[str, Tuple[Optional[str], str]] = {
            "prompt": (None, prompt),
        }
        if seed is not None:
            files["seed"] = (None, str(seed))
        if width:
            files["width"] = (None, str(width))
        if height:
            files["height"] = (None, str(height))

        response = requests.post(url, headers=headers, files=files, timeout=timeout)
    else:
        # FLUX.1 schnell accepts standard JSON
        json_payload: Dict[str, Any] = {"prompt": prompt}
        response = requests.post(url, headers=headers, json=json_payload, timeout=timeout)

    if response.status_code != 200:
        raise RuntimeError(
            f"Cloudflare Workers AI returned HTTP {response.status_code} for {model_name}: {response.text}"
        )

    content_type = response.headers.get("Content-Type", "")
    if "image" in content_type or "octet-stream" in content_type:
        return response.content

    # JSON response containing base64 image
    try:
        data = response.json()
        if "result" in data and isinstance(data["result"], dict) and "image" in data["result"]:
            return base64.b64decode(data["result"]["image"])
        elif "result" in data and isinstance(data["result"], str):
            return base64.b64decode(data["result"])
    except Exception as e:
        raise ValueError(f"Failed to parse base64 image from Cloudflare response: {e}")

    raise ValueError("No recognizable image data returned from Cloudflare Workers AI.")


def save_and_verify_image(image_bytes: bytes, output_path: str) -> None:
    """Saves image bytes and verifies that it can be opened as a valid image."""
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    # Convert/verify with PIL and save as PNG
    with Image.open(io.BytesIO(image_bytes)) as img:
        img.verify()

    with Image.open(io.BytesIO(image_bytes)) as img:
        img.save(output_path, "PNG")


def generate_single_keyframe(
    shot_id: str,
    full_prompt: str,
    seed: int,
    model_alias: str,
    output_path: str,
    cf_credentials: Optional[Tuple[str, str]] = None,
    nvidia_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    width: int = 1024,
    height: int = 1024,
) -> bool:
    """Generates a keyframe with deterministic hashing and caching.

    Returns:
        True if generated via API, False if served from cache.
    """
    model_name = MODEL_MAP.get(model_alias, model_alias)

    # Deterministic input hash
    hash_payload = {
        "full_prompt": full_prompt,
        "seed": seed,
        "model": model_name,
        "width": width,
        "height": height,
    }
    input_hash = compute_input_hash(hash_payload)

    if check_cache(output_path, input_hash):
        print(f"[CACHE HIT] {shot_id:<8} | Hash: {input_hash} | Path: {output_path}")
        return False

    print(f"[GENERATING] {shot_id:<8} | Model: {model_alias} ({model_name}) | Seed: {seed}...")
    start_time = time.time()

    def _call() -> bytes:
        if model_alias in GEMINI_MODEL_MAP:
            if not gemini_key:
                raise ValueError("Gemini API key required for this model")
            return call_gemini_image(
                api_key=gemini_key,
                model_alias=model_alias,
                prompt=full_prompt,
                seed=seed,
                width=width,
                height=height,
            )
        elif model_alias in NVIDIA_MODEL_MAP:
            if not nvidia_key:
                raise ValueError("NVIDIA API key required for this model")
            return call_nvidia_flux(
                api_key=nvidia_key,
                model_alias=model_alias,
                prompt=full_prompt,
                seed=seed,
                width=width,
                height=height,
            )
        else:
            if not cf_credentials:
                raise ValueError("Cloudflare credentials required for this model")
            account_id, api_token = cf_credentials
            return call_cloudflare_flux(
                account_id=account_id,
                api_token=api_token,
                model_alias=model_alias,
                prompt=full_prompt,
                seed=seed,
                width=width,
                height=height,
            )

    try:
        image_bytes = with_retries(_call, max_retries=3, base_delay=2.0)
        latency_s = time.time() - start_time

        save_and_verify_image(image_bytes, output_path)

        # Write cache meta sidecar
        write_cache_meta(
            output_path=output_path,
            input_hash=input_hash,
            extra={
                "shot_id": shot_id,
                "model": model_name,
                "seed": seed,
                "full_prompt": full_prompt,
                "width": width,
                "height": height,
                "latency_s": round(latency_s, 3),
            },
        )

        # Log run
        log_run(
            stage="keyframes",
            shot_id=shot_id,
            model=model_name,
            params={
                "prompt": full_prompt,
                "output_path": output_path,
                "width": width,
                "height": height,
            },
            seed=seed,
            latency_s=latency_s,
            status="ok",
            est_cost_usd=0.0,
            cache_hit=False,
        )

        print(f"  [+] Saved: {output_path} ({latency_s:.2f}s)")
        return True

    except Exception as e:
        latency_s = time.time() - start_time
        log_run(
            stage="keyframes",
            shot_id=shot_id,
            model=model_name,
            params={"prompt": full_prompt, "output_path": output_path},
            seed=seed,
            latency_s=latency_s,
            status="error",
            error=str(e),
            est_cost_usd=0.0,
            cache_hit=False,
        )
        print(f"  [-] Failed generating {shot_id}: {e}", file=sys.stderr)
        raise


def run_reference_sheet(
    storyboard: Dict[str, Any],
    model_alias: str,
    cf_credentials: Optional[Tuple[str, str]] = None,
    nvidia_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
) -> None:
    """Generates 4 candidate character & style reference images for consistency benchmarking."""
    project = storyboard.get("project", {})
    style_short = project.get("style_block_short", "")
    character_bible = project.get("character_bible", "")

    ref_prompt = (
        f"{character_bible}, {style_short}, "
        f"character reference sheet, front and three-quarter view, consistent lighting"
    )

    seeds = [101, 202, 303, 404]
    out_dir = os.path.join("outputs", "keyframes")
    os.makedirs(out_dir, exist_ok=True)

    print("================================================================================")
    print("           GENERATING CHARACTER & STYLE REFERENCE SHEET (4 SEEDS)               ")
    print(f"Prompt: {ref_prompt}")
    print("================================================================================")

    for i, seed in enumerate(seeds):
        shot_id = f"ref_sheet_{i}"
        output_path = os.path.join(out_dir, f"{shot_id}.png")
        generate_single_keyframe(
            shot_id=shot_id,
            full_prompt=ref_prompt,
            seed=seed,
            model_alias=model_alias,
            output_path=output_path,
            cf_credentials=cf_credentials,
            nvidia_key=nvidia_key,
            gemini_key=gemini_key,
        )

    print("================================================================================")
    print("Reference sheets generated in outputs/keyframes/ref_sheet_0..3.png")
    print("Pick the Golden Reference and record findings in EXPERIMENTS.md.")
    print("================================================================================")


def main() -> None:
    all_model_aliases = list(MODEL_MAP.keys())
    parser = argparse.ArgumentParser(
        description="Generate keyframes via Cloudflare Workers AI, NVIDIA NIM, or Google Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Model aliases:
  Cloudflare (CF_ACCOUNT_ID + CF_API_TOKEN):
    flux2                    @cf/black-forest-labs/flux-2-klein-4b
    flux9b                   @cf/black-forest-labs/flux-2-klein-9b
    flux1                    @cf/black-forest-labs/flux-1-schnell

  NVIDIA NIM (NVIDIA_API_KEY):
    nvidia-flux3             black-forest-labs/flux-3-image-klein
    nvidia-flux1-dev         black-forest-labs/flux.1-dev

  Google Gemini (GEMINI_API_KEY):
    gemini-3.1-flash-lite-image   Fast, quota-efficient lite model
    gemini-3.1-flash-image        Standard flash image model
    gemini-3.1-flash-image-preview Preview channel flash image model
    gemini-2.5-flash-image        High-quality flash image model
    gemini-3-pro-image-preview    Pro-tier image preview model
    imagen3                       imagen-3.0-generate-002 (highest quality)
""",
    )
    parser.add_argument(
        "--storyboard",
        default="storyboard.json",
        help="Path to storyboard JSON file (default: storyboard.json)",
    )
    parser.add_argument(
        "--model",
        choices=all_model_aliases,
        default="nvidia-flux3",
        help="Model alias to use. See list below. (default: nvidia-flux3)",
    )
    parser.add_argument(
        "--shot",
        help="Generate keyframe only for a specific shot ID (e.g. shot_04)",
    )
    parser.add_argument(
        "--ref-sheet",
        action="store_true",
        help="Generate 4 candidate character & style reference sheets (ref_sheet_0..3.png)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels (default: 1024)",
    )

    args = parser.parse_args()

    # 1. Validate storyboard.json first
    print("Validating storyboard before keyframe generation...")
    storyboard = validate_storyboard(args.storyboard)
    print(f"Storyboard validation passed: '{storyboard['project']['title']}'")

    cf_credentials: Optional[Tuple[str, str]] = None
    nvidia_key: Optional[str] = None
    gemini_key: Optional[str] = None

    if args.model in GEMINI_MODEL_MAP:
        gemini_key = get_gemini_api_key()
        print(f"[*] Using Gemini backend: {GEMINI_MODEL_MAP[args.model]}")
    elif args.model in NVIDIA_MODEL_MAP:
        nvidia_key = get_nvidia_api_key()
        print(f"[*] Using NVIDIA NIM backend: {NVIDIA_MODEL_MAP[args.model]}")
    else:
        cf_credentials = get_cf_credentials()
        print(f"[*] Using Cloudflare Workers AI backend: {CF_MODEL_MAP[args.model]}")

    # 2. Ref-sheet mode
    if args.ref_sheet:
        run_reference_sheet(
            storyboard=storyboard,
            model_alias=args.model,
            cf_credentials=cf_credentials,
            nvidia_key=nvidia_key,
            gemini_key=gemini_key,
        )
        return

    # 3. Normal keyframe generation mode
    project = storyboard["project"]
    shots: List[Dict[str, Any]] = storyboard["shots"]

    if args.shot:
        filtered = [s for s in shots if s.get("id") == args.shot]
        if not filtered:
            print(f"[ERROR] Shot '{args.shot}' not found in {args.storyboard}", file=sys.stderr)
            sys.exit(1)
        shots = filtered

    out_dir = os.path.join("outputs", "keyframes")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing keyframes for {len(shots)} shot(s) using model '{args.model}'...")

    generated_count = 0
    cached_count = 0

    for shot in shots:
        shot_id = shot["id"]
        seed = shot.get("seed", 42000)
        image_prompt = shot["image_prompt"].strip()
        style_short = project.get("style_block_short", "").strip()
        full_prompt = f"{image_prompt.rstrip('.')}. {style_short}" if style_short else image_prompt
        output_path = os.path.join(out_dir, f"{shot_id}.png")

        was_generated = generate_single_keyframe(
            shot_id=shot_id,
            full_prompt=full_prompt,
            seed=seed,
            model_alias=args.model,
            output_path=output_path,
            cf_credentials=cf_credentials,
            nvidia_key=nvidia_key,
            gemini_key=gemini_key,
            width=args.width,
            height=args.height,
        )

        if was_generated:
            generated_count += 1
        else:
            cached_count += 1

    print("\n================================================================================")
    print("                           KEYFRAMES SUMMARY                                    ")
    print("================================================================================")
    print(f"Total Shots Processed: {len(shots)}")
    print(f"Generated via API    : {generated_count}")
    print(f"Cache Hits (Skipped) : {cached_count}")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
