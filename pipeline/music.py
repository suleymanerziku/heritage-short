"""Music stage for the heritage-short pipeline.

Features:
- Validates storyboard.json against schema/storyboard.schema.json.
- Checks that project.music.selected_file exists in outputs/music/.
- If missing, provides clear instructions and generates a graceful, high-quality
  subtle acoustic meditation pad placeholder so 'make film' never breaks.
- Normalizes the selected score to intermediate loudness (-18 LUFS) via FFmpeg
  loudnorm filter so ducking/mixing in assemble.py is predictable.
- Merges outputs/music/music_runs.jsonl into runs.jsonl if present.
- Records run information and writes cache sidecar metadata.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import wave
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


def get_ffmpeg_binary() -> str:
    """Finds executable ffmpeg binary on system PATH or via imageio_ffmpeg."""
    ffmpeg_on_path = shutil.which("ffmpeg")
    if ffmpeg_on_path:
        return ffmpeg_on_path

    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return "ffmpeg"


def merge_music_runs(
    music_runs_path: str = "outputs/music/music_runs.jsonl",
    main_runs_path: str = "runs.jsonl",
) -> int:
    """Merges unique run records from Kaggle/Colab music generation into runs.jsonl."""
    if not os.path.exists(music_runs_path):
        return 0

    existing_lines: Set[str] = set()
    if os.path.exists(main_runs_path):
        with open(main_runs_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    existing_lines.add(stripped)

    new_records: List[str] = []
    with open(music_runs_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped and stripped not in existing_lines:
                new_records.append(stripped)
                existing_lines.add(stripped)

    if new_records:
        with open(main_runs_path, "a", encoding="utf-8") as f:
            for record_str in new_records:
                f.write(record_str + "\n")
        print(f"[MERGE] Merged {len(new_records)} music run record(s) into '{main_runs_path}'.")

    return len(new_records)


def generate_subtle_ambient_pad(
    output_path: str,
    duration_s: float = 82.0,
    sample_rate: int = 44100,
    key_root_hz: float = 146.83, # D3 (D minor: D3, F3, A3, D4)
) -> None:
    """Generates a soft, pure acoustic meditation ambient pad WAV as a safe fallback.

    Uses standard library wave module so no external heavy audio dependencies are required.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    num_samples = int(duration_s * sample_rate)

    # Chord frequencies for D minor meditative mood
    freqs = [146.83, 174.61, 220.00, 293.66, 440.00]
    fade_in_samples = int(sample_rate * 3.5)
    fade_out_samples = int(sample_rate * 4.5)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(2)  # Stereo
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)

        chunk_size = 4096
        sample_idx = 0

        while sample_idx < num_samples:
            current_chunk = min(chunk_size, num_samples - sample_idx)
            frames = bytearray()

            for i in range(current_chunk):
                n = sample_idx + i
                t = n / sample_rate

                # Envelope calculation
                if n < fade_in_samples:
                    env = n / fade_in_samples
                elif n > (num_samples - fade_out_samples):
                    env = (num_samples - n) / fade_out_samples
                else:
                    env = 1.0

                # Subtle breathing vibrato
                vibrato = 0.8 * math.sin(2.0 * math.pi * 0.25 * t)

                # Left and right stereo synthesis
                left_val = 0.0
                right_val = 0.0

                for idx, f in enumerate(freqs):
                    amp = 0.18 / (idx + 1)
                    phase_l = (2.0 * math.pi * f * t) + vibrato
                    phase_r = (2.0 * math.pi * (f * 1.002) * t) + (vibrato * 0.9)
                    left_val += amp * math.sin(phase_l)
                    right_val += amp * math.sin(phase_r)

                left_val = max(-1.0, min(1.0, left_val * env * 0.7))
                right_val = max(-1.0, min(1.0, right_val * env * 0.7))

                left_int = int(left_val * 32767.0)
                right_int = int(right_val * 32767.0)

                frames.extend(struct.pack("<hh", left_int, right_int))

            wf.writeframes(frames)
            sample_idx += current_chunk


def normalize_to_target_lufs(
    input_audio_path: str,
    output_audio_path: str,
    target_lufs: float = -18.0,
    target_tp: float = -1.5,
    target_lra: float = 11.0,
) -> float:
    """Normalizes an audio track to standard target integrated loudness using FFmpeg loudnorm.

    Returns:
        Latency in seconds.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    os.makedirs(os.path.dirname(os.path.abspath(output_audio_path)), exist_ok=True)

    loudnorm_filter = (
        f"loudnorm=I={target_lufs}:TP={target_tp}:LRA={target_lra}"
    )

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", input_audio_path,
        "-af", loudnorm_filter,
        "-ar", "44100",
        "-c:a", "libmp3lame" if output_audio_path.endswith(".mp3") else "pcm_s16le",
        output_audio_path,
    ]

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    latency_s = time.time() - start_time

    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg loudnorm failed on {input_audio_path}:\n{result.stderr}")

    return latency_s


def run_music_stage(
    storyboard: Dict[str, Any],
    force: bool = False,
) -> str:
    """Orchestrates validation, selection verification, normalization, and logging."""
    project = storyboard.get("project", {})
    music_cfg = project.get("music", {})

    selected_rel_path = music_cfg.get("selected_file", "outputs/music/heritage_theme.mp3")
    caption = music_cfg.get("caption", "Acoustic Ethiopian meditative melody")
    bpm = music_cfg.get("bpm", 68)
    key = music_cfg.get("key", "D minor")
    duration_s = float(music_cfg.get("duration_s", 82.0))
    seed = int(music_cfg.get("seed", 1974))

    # Output normalized file path
    normalized_path = os.path.join("outputs", "music", "score_normalized.wav")

    # 1. Merge any external runs from notebooks
    merge_music_runs()

    # 2. Check if selected_file exists
    if not os.path.exists(selected_rel_path):
        print(f"\n[NOTICE] Configured music file '{selected_rel_path}' does not exist.")
        print("         To generate model candidates, run 'notebooks/acestep_score.ipynb' on Kaggle/Colab.")
        print(f"         Generating subtle graceful ambient pad placeholder ({duration_s}s, key of {key})...")

        placeholder_raw = os.path.join("outputs", "music", "score_placeholder.wav")
        generate_subtle_ambient_pad(
            output_path=placeholder_raw,
            duration_s=duration_s,
        )
        source_audio = placeholder_raw
        source_model = "procedural_meditation_pad"
    else:
        print(f"[FOUND] Found selected music file: '{selected_rel_path}'")
        source_audio = selected_rel_path
        source_model = "ACE-Step-1.5"

    # 3. Deterministic caching check
    input_hash = compute_input_hash({
        "source_audio": os.path.normpath(source_audio).replace("\\", "/"),
        "target_lufs": -18.0,
        "caption": caption,
        "bpm": bpm,
        "key": key,
        "seed": seed,
    })

    if not force and check_cache(normalized_path, input_hash):
        print(f"[CACHE HIT] Normalized music score cached at: {normalized_path}")
        return normalized_path

    # 4. Perform loudness normalization (-18 LUFS)
    print(f"[*] Normalizing score to -18 LUFS via FFmpeg loudnorm...")
    latency_s = normalize_to_target_lufs(
        input_audio_path=source_audio,
        output_audio_path=normalized_path,
        target_lufs=-18.0,
    )

    actual_duration = probe_audio_duration(normalized_path)

    # 5. Write cache metadata and log
    write_cache_meta(
        output_path=normalized_path,
        input_hash=input_hash,
        extra={
            "source_audio": source_audio,
            "source_model": source_model,
            "target_lufs": -18.0,
            "actual_duration_s": round(actual_duration, 2),
            "bpm": bpm,
            "key": key,
            "seed": seed,
            "latency_s": round(latency_s, 3),
        },
    )

    log_run(
        stage="music",
        shot_id=None,
        model="ffmpeg_loudnorm_normalization",
        params={
            "source_model": source_model,
            "source_file": source_audio,
            "target_lufs": -18.0,
            "duration_s": actual_duration,
            "bpm": bpm,
            "key": key,
        },
        seed=seed,
        latency_s=latency_s,
        status="ok",
        est_cost_usd=0.0,
        cache_hit=False,
    )

    print(f"  [+] Saved normalized score: {normalized_path} ({actual_duration:.1f}s, {latency_s:.2f}s)")
    return normalized_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Process and normalize music stage.")
    parser.add_argument(
        "--storyboard",
        default="storyboard.json",
        help="Path to storyboard JSON file (default: storyboard.json)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration ignoring cache",
    )

    args = parser.parse_args()

    # 1. Validate storyboard first
    print("Validating storyboard before music processing...")
    storyboard = validate_storyboard(args.storyboard)
    print(f"Storyboard validation passed: '{storyboard['project']['title']}'")

    # 2. Run music stage
    normalized_file = run_music_stage(storyboard, force=args.force)

    print("\n================================================================================")
    print("                             MUSIC STAGE SUMMARY                                ")
    print("================================================================================")
    print(f"Normalized Master Score: {normalized_file}")
    print(f"Target Loudness        : -18.0 LUFS")
    print("Status                 : Ready for mix & ducking in assemble.py")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
