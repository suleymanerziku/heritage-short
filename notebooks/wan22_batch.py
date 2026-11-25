"""Wan 2.2 TI2V-5B Batch Generation Script for Kaggle (2x T4 or single GPU).

Reads wan22_manifest.json and input keyframes, runs Wan 2.2 Image-to-Video in
fp16 with CPU offloading, writes per-shot MP4 + .meta.json sidecars immediately,
logs run metrics to kaggle_runs.jsonl, and zips results.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone

# ==============================================================================
# CONFIGURATION & TOGGLES
# ==============================================================================
DRAFT_MODE = True  # True: 480p fast draft (saves GPU quota); False: 720p final

MANIFEST_PATH = "wan22_manifest.json"
KEYFRAMES_DIR = "keyframes"
OUTPUT_DIR = "outputs/shots"
KAGGLE_RUNS_LOG = os.path.join(OUTPUT_DIR, "kaggle_runs.jsonl")
ZIP_OUTPUT = "wan22_shots_bundle.zip"

# Resolution and steps based on mode
if DRAFT_MODE:
    RESOLUTION = (832, 480)  # Width, Height (approx 480p 16:9 safe)
    NUM_INFERENCE_STEPS = 20
    GUIDANCE_SCALE = 5.0
else:
    RESOLUTION = (1280, 720) # 720p
    NUM_INFERENCE_STEPS = 35
    GUIDANCE_SCALE = 6.0


def install_dependencies():
    """Installs required packages on Kaggle environment."""
    print("[*] Installing minimal dependencies for Wan 2.2 TI2V-5B...")
    pkgs = [
        "diffusers>=0.32.0",
        "transformers>=4.48.0",
        "accelerate>=0.34.0",
        "torchvision",
        "imageio",
        "imageio-ffmpeg",
        "sentencepiece",
    ]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q"] + pkgs, check=True)
    print("[+] Dependencies successfully installed.")


def load_pipeline():
    """Initializes Wan 2.2 TI2V pipeline with memory optimizations for T4 (16GB)."""
    import torch
    from diffusers import AutoencoderKLWan, WanImageToVideoPipeline

    print(f"[*] Loading Wan-AI/Wan2.2-TI2V-5B (torch {torch.__version__}, CUDA: {torch.cuda.is_available()})...")

    # Use diffusers pipeline with CPU offload & fp16
    pipe = WanImageToVideoPipeline.from_pretrained(
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        torch_dtype=torch.float16,
    )

    # Enable aggressive memory optimizations for 16GB T4 GPU
    try:
        pipe.enable_model_cpu_offload()
        print("[+] Model CPU offload enabled.")
    except Exception as e:
        print(f"[!] CPU offload fallback: {e}")
        pipe.to("cuda")

    try:
        pipe.enable_vae_slicing()
        pipe.enable_vae_tiling()
        print("[+] VAE slicing & tiling enabled.")
    except Exception:
        pass

    return pipe


def process_batch(pipe):
    """Processes shots described in wan22_manifest.json."""
    import torch
    from diffusers.utils import export_to_video
    from PIL import Image

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    shots = manifest.get("shots", [])
    negative_prompt = manifest.get("negative_prompt_video", "")
    fps = manifest.get("fps", 24)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n[*] Processing {len(shots)} shots. DRAFT_MODE={DRAFT_MODE}, Resolution={RESOLUTION}...")

    completed_count = 0

    for idx, shot in enumerate(shots, 1):
        shot_id = shot["shot_id"]
        motion_prompt = shot["motion_prompt"]
        seed = int(shot.get("seed", 42000))
        target_duration_s = float(shot.get("target_duration_s", 5.0))
        input_hash = shot.get("input_hash", "")

        # Look for keyframe image in KEYFRAMES_DIR or local manifest path
        keyframe_filename = f"{shot_id}.png"
        candidates = [
            os.path.join(KEYFRAMES_DIR, keyframe_filename),
            shot.get("keyframe_path", ""),
            os.path.basename(shot.get("keyframe_path", "")),
        ]
        image_path = next((c for c in candidates if os.path.exists(c)), None)

        if not image_path:
            print(f"[!] Keyframe not found for {shot_id}. Searched {candidates}. Skipping.")
            continue

        output_mp4 = os.path.join(OUTPUT_DIR, f"{shot_id}.mp4")
        meta_json = f"{output_mp4}.meta.json"

        # Check existing output cache
        if os.path.exists(output_mp4) and os.path.exists(meta_json):
            try:
                with open(meta_json, "r", encoding="utf-8") as mf:
                    meta = json.load(mf)
                if meta.get("input_hash") == input_hash:
                    print(f"[{idx}/{len(shots)}] [CACHE HIT] {shot_id} already generated.")
                    completed_count += 1
                    continue
            except Exception:
                pass

        print(f"[{idx}/{len(shots)}] [GENERATING] {shot_id} | Seed: {seed} | Duration: {target_duration_s}s...")
        start_time = time.time()

        try:
            image = Image.open(image_path).convert("RGB")
            # Calculate num_frames based on target duration
            # Wan generates at fixed frame step (e.g., 49 or 81 frames)
            num_frames = max(17, int(round(target_duration_s * 16)))
            if num_frames % 4 != 1:
                num_frames = ((num_frames // 4) * 4) + 1

            generator = torch.Generator(device="cuda").manual_seed(seed)

            output = pipe(
                image=image,
                prompt=motion_prompt,
                negative_prompt=negative_prompt,
                height=RESOLUTION[1],
                width=RESOLUTION[0],
                num_frames=num_frames,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            )

            frames = output.frames[0]
            export_to_video(frames, output_mp4, fps=fps)
            latency_s = time.time() - start_time
            gpu_min = latency_s / 60.0

            # Write immediate sidecar metadata
            meta_payload = {
                "input_hash": input_hash,
                "shot_id": shot_id,
                "model": "Wan-AI/Wan2.2-TI2V-5B",
                "seed": seed,
                "motion_prompt": motion_prompt,
                "target_duration_s": target_duration_s,
                "resolution": f"{RESOLUTION[0]}x{RESOLUTION[1]}",
                "draft_mode": DRAFT_MODE,
                "latency_s": round(latency_s, 2),
                "gpu_min": round(gpu_min, 4),
            }
            with open(meta_json, "w", encoding="utf-8") as mf:
                json.dump(meta_payload, mf, indent=2)

            # Log to kaggle_runs.jsonl immediately
            log_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stage": "shots",
                "shot_id": shot_id,
                "model": "Wan-AI/Wan2.2-TI2V-5B",
                "params": {
                    "draft_mode": DRAFT_MODE,
                    "resolution": f"{RESOLUTION[0]}x{RESOLUTION[1]}",
                    "num_inference_steps": NUM_INFERENCE_STEPS,
                    "guidance_scale": GUIDANCE_SCALE,
                    "target_duration_s": target_duration_s,
                },
                "seed": seed,
                "latency_s": round(latency_s, 4),
                "status": "ok",
                "gpu_min": round(gpu_min, 4),
                "est_cost_usd": 0.0,
                "cache_hit": False,
                "error": None,
            }
            with open(KAGGLE_RUNS_LOG, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_record, ensure_ascii=False) + "\n")

            print(f"  [+] Saved {output_mp4} ({latency_s:.1f}s, {gpu_min:.2f} GPU min)")
            completed_count += 1

        except Exception as e:
            latency_s = time.time() - start_time
            gpu_min = latency_s / 60.0
            print(f"  [-] Failed generating {shot_id}: {e}")

            err_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "stage": "shots",
                "shot_id": shot_id,
                "model": "Wan-AI/Wan2.2-TI2V-5B",
                "params": {"draft_mode": DRAFT_MODE, "resolution": f"{RESOLUTION[0]}x{RESOLUTION[1]}"},
                "seed": seed,
                "latency_s": round(latency_s, 4),
                "status": "error",
                "gpu_min": round(gpu_min, 4),
                "est_cost_usd": 0.0,
                "cache_hit": False,
                "error": str(e),
            }
            with open(KAGGLE_RUNS_LOG, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(err_record, ensure_ascii=False) + "\n")

    print(f"\n[+] Batch complete. Generated/Cached {completed_count}/{len(shots)} shots.")

    # Create downloadable zip
    print(f"[*] Packaging outputs into '{ZIP_OUTPUT}'...")
    with zipfile.ZipFile(ZIP_OUTPUT, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, os.path.dirname(OUTPUT_DIR))
                zipf.write(file_path, arcname=rel_path)

    print(f"[+] All clips & logs packaged successfully into: {ZIP_OUTPUT}")
    print("    Download this zip and extract directly into 'outputs/shots/' on your local machine.")


if __name__ == "__main__":
    install_dependencies()
    pipe = load_pipeline()
    process_batch(pipe)
