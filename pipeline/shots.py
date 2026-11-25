"""Moving shots generator and orchestrator for the heritage-short pipeline.

Features:
- Validates storyboard.json against schema/storyboard.schema.json.
- Supports --export-manifest: writes outputs/shots/wan22_manifest.json with all
  shots configured for Wan 2.2 TI2V-5B generation on Kaggle GPU.
- In normal mode (python pipeline/shots.py):
  1. Merges unique entries from outputs/shots/kaggle_runs.jsonl into runs.jsonl.
  2. For each shot:
     - If video_mode == 'wan22' and outputs/shots/{shot_id}.mp4 exists with
       matching input_hash, hits cache ([CACHE HIT]).
     - If video_mode == 'ken_burns' OR if the Wan 2.2 MP4 is missing, gracefully
       falls back to generating a smooth 24fps FFmpeg zoompan Ken Burns clip
       from outputs/keyframes/{shot_id}.png matching target_duration_s and
       ken_burns_motion (zoom_in, zoom_out, pan_left, pan_right).
     - Caches Ken Burns clips with sidecar metadata and logs to runs.jsonl
       with model="ffmpeg_ken_burns".
- Supports --shot <shot_id> and --force flags.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline.utils import (
    check_cache,
    compute_input_hash,
    log_run,
    probe_audio_duration,
    validate_storyboard,
    write_cache_meta,
)

# Discover ffmpeg binary path (system PATH or imageio_ffmpeg)
def get_ffmpeg_binary() -> str:
    """Finds an executable ffmpeg binary on PATH or via imageio_ffmpeg."""
    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        return ffmpeg_on_path

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return "ffmpeg"


def get_target_duration(shot: Dict[str, Any], default_duration: float = 5.0) -> float:
    """Calculates target shot duration in seconds.

    Derived from outputs/narration/{shot_id}.wav duration + pad_after_s.
    Falls back to default_duration if narration audio is missing.
    """
    shot_id = shot["id"]
    pad_after = float(shot.get("pad_after_s", 0.4))
    narration_path = os.path.join("outputs", "narration", f"{shot_id}.wav")

    if os.path.exists(narration_path):
        try:
            audio_duration = probe_audio_duration(narration_path)
            return round(audio_duration + pad_after, 2)
        except Exception as e:
            print(f"[WARNING] Could not probe audio duration for {shot_id}: {e}", file=sys.stderr)

    return round(float(shot.get("duration_s", default_duration)) + pad_after, 2)


def compute_shot_hash(
    shot_id: str,
    keyframe_path: str,
    motion_prompt: str,
    motion_type: str,
    target_duration_s: float,
    seed: int,
    mode: str,
) -> str:
    """Computes deterministic 12-char SHA-256 hash for a shot clip."""
    payload = {
        "shot_id": shot_id,
        "keyframe_path": os.path.normpath(keyframe_path).replace("\\", "/"),
        "motion_prompt": motion_prompt,
        "motion_type": motion_type,
        "target_duration_s": round(float(target_duration_s), 2),
        "seed": seed,
        "mode": mode,
    }
    return compute_input_hash(payload)


def merge_kaggle_runs(
    kaggle_runs_path: str = "outputs/shots/kaggle_runs.jsonl",
    main_runs_path: str = "runs.jsonl",
) -> int:
    """Merges unique run records from Kaggle GPU runs into root runs.jsonl."""
    if not os.path.exists(kaggle_runs_path):
        return 0

    existing_lines: Set[str] = set()
    if os.path.exists(main_runs_path):
        with open(main_runs_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    existing_lines.add(stripped)

    new_records: List[str] = []
    with open(kaggle_runs_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and stripped not in existing_lines:
                new_records.append(stripped)
                existing_lines.add(stripped)

    if new_records:
        with open(main_runs_path, "a", encoding="utf-8") as f:
            for record_str in new_records:
                f.write(record_str + "\n")
        print(f"[MERGE] Merged {len(new_records)} run record(s) from '{kaggle_runs_path}' into '{main_runs_path}'.")

    return len(new_records)


def export_wan22_manifest(
    storyboard: Dict[str, Any],
    manifest_path: str = "outputs/shots/wan22_manifest.json",
) -> Dict[str, Any]:
    """Generates and writes wan22_manifest.json for remote Kaggle GPU processing."""
    project = storyboard.get("project", {})
    shots = storyboard.get("shots", [])

    os.makedirs(os.path.dirname(os.path.abspath(manifest_path)), exist_ok=True)

    manifest_shots = []
    for shot in shots:
        shot_id = shot["id"]
        video_mode = shot.get("video_mode", "wan22")
        if video_mode != "wan22":
            continue

        keyframe_path = os.path.join("outputs", "keyframes", f"{shot_id}.png")
        target_duration = get_target_duration(shot)
        motion_prompt = shot.get("motion_prompt", "")
        motion_type = shot.get("ken_burns_motion", "zoom_in")
        seed = int(shot.get("seed", 42000))

        input_hash = compute_shot_hash(
            shot_id=shot_id,
            keyframe_path=keyframe_path,
            motion_prompt=motion_prompt,
            motion_type=motion_type,
            target_duration_s=target_duration,
            seed=seed,
            mode="wan22",
        )

        manifest_shots.append({
            "shot_id": shot_id,
            "keyframe_path": os.path.normpath(keyframe_path).replace("\\", "/"),
            "motion_prompt": motion_prompt,
            "seed": seed,
            "target_duration_s": target_duration,
            "ken_burns_motion": motion_type,
            "input_hash": input_hash,
        })

    manifest_payload = {
        "project_title": project.get("title", "Heritage Short"),
        "fps": project.get("fps", 24),
        "negative_prompt_video": project.get("negative_prompt_video", ""),
        "total_shots": len(manifest_shots),
        "shots": manifest_shots,
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_payload, f, indent=2, ensure_ascii=False)

    print(f"[EXPORT] Successfully generated Wan 2.2 manifest with {len(manifest_shots)} shot(s):")
    print(f"         --> {manifest_path}")
    return manifest_payload


def generate_ken_burns_clip(
    image_path: str,
    output_path: str,
    motion_type: str,
    duration_s: float,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
) -> float:
    """Generates a smooth Ken Burns pan/zoom video clip from a still keyframe image using FFmpeg.

    Supported motions:
    - zoom_in: zooms smoothly from 1.0x to 1.15x centered
    - zoom_out: zooms smoothly from 1.15x down to 1.0x centered
    - pan_left: drifts framing smoothly from right to left
    - pan_right: drifts framing smoothly from left to right

    Returns:
        Execution latency in seconds.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    total_frames = max(1, int(round(duration_s * fps)))

    if motion_type == "zoom_in":
        # Zoom from 1.0 to 1.15 smoothly
        vf_zoompan = (
            f"zoompan=z='min(zoom+0.0015,1.15)':d={total_frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
        )
    elif motion_type == "zoom_out":
        # Zoom from 1.15 down to 1.0 smoothly
        vf_zoompan = (
            f"zoompan=z='if(lte(zoom,1.0),1.15,max(1.001,zoom-0.0015))':d={total_frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
        )
    elif motion_type == "pan_left":
        # Zoom 1.12x and pan right-to-left
        vf_zoompan = (
            f"zoompan=z=1.12:d={total_frames}:"
            f"x='(1-on/{total_frames})*(iw-iw/zoom)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
        )
    elif motion_type == "pan_right":
        # Zoom 1.12x and pan left-to-right
        vf_zoompan = (
            f"zoompan=z=1.12:d={total_frames}:"
            f"x='(on/{total_frames})*(iw-iw/zoom)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
        )
    else:
        # Default gentle centered zoom in
        vf_zoompan = (
            f"zoompan=z='min(zoom+0.0015,1.12)':d={total_frames}:"
            f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={fps}"
        )

    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loop", "1",
        "-i", image_path,
        "-vf", vf_zoompan,
        "-c:v", "libx264",
        "-t", f"{duration_s:.3f}",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        output_path,
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    latency_s = time.time() - start_time

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg Ken Burns generation failed for {output_path}:\n{result.stderr}")

    return latency_s


def process_shots(
    storyboard: Dict[str, Any],
    specific_shot: Optional[str] = None,
    force: bool = False,
) -> None:
    """Processes shots: checks Wan 2.2 results first, falls back gracefully to Ken Burns."""
    project = storyboard.get("project", {})
    fps = project.get("fps", 24)
    shots = storyboard.get("shots", [])

    if specific_shot:
        filtered = [s for s in shots if s.get("id") == specific_shot]
        if not filtered:
            print(f"[ERROR] Shot '{specific_shot}' not found in storyboard.", file=sys.stderr)
            sys.exit(1)
        shots = filtered

    out_dir = os.path.join("outputs", "shots")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Merge any existing Kaggle GPU runs
    merge_kaggle_runs()

    print(f"\nProcessing moving shots for {len(shots)} shot(s) (Target FPS: {fps})...")

    wan_cache_hits = 0
    kb_cache_hits = 0
    kb_generated = 0

    for shot in shots:
        shot_id = shot["id"]
        video_mode = shot.get("video_mode", "wan22")
        motion_type = shot.get("ken_burns_motion", "zoom_in")
        motion_prompt = shot.get("motion_prompt", "")
        seed = int(shot.get("seed", 42000))
        target_duration = get_target_duration(shot)

        output_path = os.path.join(out_dir, f"{shot_id}.mp4")
        keyframe_path = os.path.join("outputs", "keyframes", f"{shot_id}.png")

        # 1. Check if Wan 2.2 clip is already present and valid
        wan_hash = compute_shot_hash(
            shot_id=shot_id,
            keyframe_path=keyframe_path,
            motion_prompt=motion_prompt,
            motion_type=motion_type,
            target_duration_s=target_duration,
            seed=seed,
            mode="wan22",
        )

        if not force and check_cache(output_path, wan_hash):
            print(f"[CACHE HIT] {shot_id:<8} | Wan 2.2 TI2V-5B MP4 verified: {output_path}")
            wan_cache_hits += 1
            continue

        # 2. Check if a previously generated Ken Burns fallback clip is cached
        kb_hash = compute_shot_hash(
            shot_id=shot_id,
            keyframe_path=keyframe_path,
            motion_prompt=motion_prompt,
            motion_type=motion_type,
            target_duration_s=target_duration,
            seed=seed,
            mode="ken_burns",
        )

        if not force and check_cache(output_path, kb_hash):
            print(f"[CACHE HIT] {shot_id:<8} | FFmpeg Ken Burns ({motion_type}) cached: {output_path}")
            kb_cache_hits += 1
            continue

        # 3. Generate Ken Burns fallback clip from keyframe image
        if not os.path.exists(keyframe_path):
            print(f"[WARNING] Keyframe missing at '{keyframe_path}'. Skipping {shot_id}.", file=sys.stderr)
            continue

        reason = "Wan 2.2 pending/missing" if video_mode == "wan22" else "video_mode=ken_burns"
        print(f"[FALLBACK]  {shot_id:<8} | Generating Ken Burns ({motion_type}, {target_duration}s, {reason})...")

        try:
            latency_s = generate_ken_burns_clip(
                image_path=keyframe_path,
                output_path=output_path,
                motion_type=motion_type,
                duration_s=target_duration,
                fps=fps,
            )

            write_cache_meta(
                output_path=output_path,
                input_hash=kb_hash,
                extra={
                    "shot_id": shot_id,
                    "model": "ffmpeg_ken_burns",
                    "motion_type": motion_type,
                    "target_duration_s": target_duration,
                    "fps": fps,
                    "latency_s": round(latency_s, 3),
                    "fallback_from": video_mode,
                },
            )

            log_run(
                stage="shots",
                shot_id=shot_id,
                model="ffmpeg_ken_burns",
                params={
                    "motion_type": motion_type,
                    "target_duration_s": target_duration,
                    "fps": fps,
                    "fallback_from": video_mode,
                },
                seed=seed,
                latency_s=latency_s,
                status="ok",
                est_cost_usd=0.0,
                cache_hit=False,
            )

            print(f"  [+] Saved Ken Burns MP4: {output_path} ({latency_s:.2f}s)")
            kb_generated += 1

        except Exception as e:
            print(f"[ERROR] Failed generating Ken Burns clip for {shot_id}: {e}", file=sys.stderr)
            log_run(
                stage="shots",
                shot_id=shot_id,
                model="ffmpeg_ken_burns",
                params={"motion_type": motion_type, "target_duration_s": target_duration},
                seed=seed,
                latency_s=0.0,
                status="error",
                error=str(e),
                est_cost_usd=0.0,
                cache_hit=False,
            )

    print("\n================================================================================")
    print("                             SHOTS STAGE SUMMARY                                ")
    print("================================================================================")
    print(f"Total Shots Processed   : {len(shots)}")
    print(f"Wan 2.2 Cache Hits      : {wan_cache_hits}")
    print(f"Ken Burns Cache Hits    : {kb_cache_hits}")
    print(f"Ken Burns Generated     : {kb_generated}")
    print("================================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Moving shots generator and orchestrator.")
    parser.add_argument(
        "--storyboard",
        default="storyboard.json",
        help="Path to storyboard JSON file (default: storyboard.json)",
    )
    parser.add_argument(
        "--export-manifest",
        action="store_true",
        help="Export outputs/shots/wan22_manifest.json for Kaggle GPU execution",
    )
    parser.add_argument(
        "--manifest-path",
        default="outputs/shots/wan22_manifest.json",
        help="Path for exported manifest (default: outputs/shots/wan22_manifest.json)",
    )
    parser.add_argument(
        "--shot",
        help="Process only a specific shot ID (e.g. shot_01)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration ignoring cache",
    )

    args = parser.parse_args()

    # 1. Validate storyboard first
    print("Validating storyboard before shots processing...")
    storyboard = validate_storyboard(args.storyboard)
    print(f"Storyboard validation passed: '{storyboard['project']['title']}'")

    # 2. Manifest export mode
    if args.export_manifest:
        export_wan22_manifest(storyboard, manifest_path=args.manifest_path)
        return

    # 3. Normal local orchestrator mode
    process_shots(storyboard, specific_shot=args.shot, force=args.force)


if __name__ == "__main__":
    main()
