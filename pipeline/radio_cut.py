"""Assembles an audio-first Radio Cut prototype for User Test #1.

Generates 1280x720 24fps video with black background and centered shot metadata
(shot_id, shot_type, subtitle_en), concatenating all narration WAVs with pad_after_s
silence, and optionally mixing outputs/music/temp.wav at -22 dB.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
from typing import Any, Dict, List, Optional

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image, ImageDraw, ImageFont

from pipeline.utils import log_run, probe_audio_duration, validate_storyboard


def get_ffmpeg_binary() -> str:
    """Locates the FFmpeg binary in system PATH or via imageio_ffmpeg."""
    path = shutil.which("ffmpeg")
    if path:
        return path

    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass

    return "ffmpeg"


def create_shot_slide(
    shot_id: str,
    shot_type: str,
    subtitle_en: str,
    output_path: str,
    width: int = 1280,
    height: int = 720,
) -> str:
    """Renders a high-contrast 1280x720 slide with centered shot metadata."""
    img = Image.new("RGB", (width, height), color=(12, 12, 12))
    draw = ImageDraw.Draw(img)

    # Attempt to load clean fonts with fallback to default
    font_header = None
    font_subtitle = None
    font_footer = None

    for font_name in ("arial.ttf", "segoeui.ttf", "calibri.ttf", "DejaVuSans.ttf"):
        try:
            font_header = ImageFont.truetype(font_name, 36)
            font_subtitle = ImageFont.truetype(font_name, 30)
            font_footer = ImageFont.truetype(font_name, 18)
            break
        except Exception:
            continue

    if font_header is None:
        font_header = ImageFont.load_default()
        font_subtitle = ImageFont.load_default()
        font_footer = ImageFont.load_default()

    # Center-horizontal reference
    center_x = width // 2

    # 1. Header (e.g. "[SHOT 01]  •  ESTABLISHING")
    header_str = f"[{shot_id.upper()}]   •   {shot_type.upper().replace('_', ' ')}"
    draw.text((center_x, 260), header_str, font=font_header, fill=(245, 195, 95), anchor="mm")

    # 2. Subtitle with line wrapping
    wrapped_lines = textwrap.wrap(f'"{subtitle_en}"', width=52)
    start_y = 350
    line_spacing = 42

    for i, line in enumerate(wrapped_lines):
        line_y = start_y + (i * line_spacing)
        draw.text((center_x, line_y), line, font=font_subtitle, fill=(255, 255, 255), anchor="mm")

    # 3. Footer / Watermark
    footer_text = "HERITAGE-SHORT  |  RADIO CUT PROTOTYPE  •  AUDIO PACING REVIEW"
    draw.text((center_x, height - 50), footer_text, font=font_footer, fill=(120, 120, 120), anchor="mm")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    img.save(output_path, "PNG")
    return output_path


def render_shot_clip(
    ffmpeg_bin: str,
    slide_image: str,
    narration_wav: str,
    pad_after_s: float,
    output_clip_path: str,
) -> float:
    """Renders a single video clip combining the slide image, narration audio, and trailing silence."""
    audio_dur = probe_audio_duration(narration_wav)
    total_dur = round(audio_dur + pad_after_s, 3)

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loop", "1",
        "-framerate", "24",
        "-t", str(total_dur),
        "-i", slide_image,
        "-i", narration_wav,
        "-filter_complex", f"[1:a]apad=pad_dur={pad_after_s}[a]",
        "-map", "0:v",
        "-map", "[a]",
        "-c:v", "libx264",
        "-tune", "stillimage",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-r", "24",
        "-shortest",
        output_clip_path,
    ]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"FFmpeg failed rendering clip {output_clip_path}: {res.stderr}")

    return total_dur


def assemble_radio_cut(
    storyboard_path: str = "storyboard.json",
    schema_path: str = "schema/storyboard.schema.json",
    narration_dir: str = "outputs/narration",
    music_temp_path: str = "outputs/music/temp.wav",
    output_mp4: str = "outputs/final/radio_cut.mp4",
) -> float:
    """Orchestrates end-to-end radio cut creation."""
    print("=" * 80)
    print("Building Audio-First Radio Cut Prototype (outputs/final/radio_cut.mp4)")
    print("=" * 80)

    # 1. Validate storyboard
    storyboard = validate_storyboard(storyboard_path, schema_path)
    shots = storyboard.get("shots", [])
    if not shots:
        raise ValueError("Storyboard contains zero shots.")

    ffmpeg_bin = get_ffmpeg_binary()
    print(f"Using FFmpeg binary: {ffmpeg_bin}\n")

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    start_time = time.time()

    # Check that all narration WAVs exist before proceeding
    missing_wavs = []
    for s in shots:
        wav_file = os.path.join(narration_dir, f"{s['id']}.wav")
        if not os.path.exists(wav_file):
            missing_wavs.append(wav_file)

    if missing_wavs:
        raise FileNotFoundError(
            f"Missing {len(missing_wavs)} narration audio file(s). Please run 'make narration' first:\n"
            + "\n".join(f"  - {f}" for f in missing_wavs[:5])
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        clip_paths: List[str] = []
        durations: List[float] = []

        for i, s in enumerate(shots, 1):
            shot_id = s["id"]
            shot_type = s["shot_type"]
            sub_en = s["subtitle_en"]
            pad_s = float(s.get("pad_after_s", 0.4))
            wav_path = os.path.join(narration_dir, f"{shot_id}.wav")

            slide_path = os.path.join(tmp_dir, f"slide_{shot_id}.png")
            create_shot_slide(shot_id, shot_type, sub_en, slide_path)

            clip_path = os.path.join(tmp_dir, f"clip_{shot_id}.mp4")
            clip_dur = render_shot_clip(
                ffmpeg_bin=ffmpeg_bin,
                slide_image=slide_path,
                narration_wav=wav_path,
                pad_after_s=pad_s,
                output_clip_path=clip_path,
            )

            clip_paths.append(clip_path)
            durations.append(clip_dur)
            print(f"  [{i}/{len(shots)}] Rendered {shot_id} ({clip_dur:.2f}s, pad {pad_s}s)")

        # Concatenate all clips
        concat_list_file = os.path.join(tmp_dir, "concat_list.txt")
        with open(concat_list_file, "w", encoding="utf-8") as f:
            for cp in clip_paths:
                # Format paths safely for FFmpeg concat demuxer
                safe_path = os.path.abspath(cp).replace("\\", "/")
                f.write(f"file '{safe_path}'\n")

        raw_concat_video = os.path.join(tmp_dir, "raw_radio_cut.mp4")
        concat_cmd = [
            ffmpeg_bin,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list_file,
            "-c", "copy",
            raw_concat_video,
        ]

        print("\nConcatenating shot clips into master timeline...")
        res = subprocess.run(concat_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg concatenation failed: {res.stderr}")

        # Check for optional background music/ambient track
        has_ambient = os.path.exists(music_temp_path)
        if has_ambient:
            print(f"Mixing ambient background audio from '{music_temp_path}' at -22 dB...")
            mix_cmd = [
                ffmpeg_bin,
                "-y",
                "-i", raw_concat_video,
                "-stream_loop", "-1",
                "-i", music_temp_path,
                "-filter_complex", (
                    "[0:a]volume=1.0[vocal];"
                    "[1:a]volume=-22dB[amb];"
                    "[vocal][amb]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                ),
                "-map", "0:v",
                "-map", "[aout]",
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                output_mp4,
            ]
            res_mix = subprocess.run(mix_cmd, capture_output=True, text=True)
            if res_mix.returncode != 0:
                print(f"[WARNING] Ambient mixing failed ({res_mix.stderr}). Exporting clean audio cut.", file=sys.stderr)
                shutil.copyfile(raw_concat_video, output_mp4)
        else:
            print("No 'outputs/music/temp.wav' found; exporting narration-only radio cut.")
            shutil.copyfile(raw_concat_video, output_mp4)

    total_duration = probe_audio_duration(output_mp4)
    latency_s = time.time() - start_time

    # Log run
    log_run(
        stage="radio_cut",
        shot_id="all",
        model="ffmpeg",
        params={
            "resolution": "1280x720",
            "fps": 24,
            "has_ambient": has_ambient,
            "num_shots": len(shots),
        },
        seed=None,
        latency_s=latency_s,
        status="ok",
        est_cost_usd=0.0,
        cache_hit=False,
    )

    print("\n" + "=" * 80)
    print("RADIO CUT EXPORT COMPLETE")
    print("=" * 80)
    print(f"Output File   : {output_mp4}")
    print(f"Total Duration: {total_duration:.2f} seconds")
    print(f"Assembly Time : {latency_s:.2f} seconds")
    print("=" * 80 + "\n")

    return total_duration


def main() -> int:
    parser = argparse.ArgumentParser(description="Assemble fast audio-first Radio Cut prototype.")
    parser.add_argument("--storyboard", type=str, default="storyboard.json", help="Path to storyboard.json")
    parser.add_argument("--schema", type=str, default="schema/storyboard.schema.json", help="Path to schema")
    parser.add_argument("--narration-dir", type=str, default="outputs/narration", help="Narration WAVs directory")
    parser.add_argument("--ambient-track", type=str, default="outputs/music/temp.wav", help="Optional ambient WAV")
    parser.add_argument("--output", type=str, default="outputs/final/radio_cut.mp4", help="Output MP4 path")

    args = parser.parse_args()

    try:
        assemble_radio_cut(
            storyboard_path=args.storyboard,
            schema_path=args.schema,
            narration_dir=args.narration_dir,
            music_temp_path=args.ambient_track,
            output_mp4=args.output,
        )
        return 0
    except Exception as exc:
        print(f"[FATAL] Radio cut assembly failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
