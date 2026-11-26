"""Assembles the final 16:9 and 9:16 master films purely from storyboard.json and cached stage outputs.

Requirements & Architecture:
1. Validates storyboard.json and verifies all narration WAVs, shot MP4s, and selected music WAV exist.
2. Shot Video Normalization & Trimming:
   - Trims 0.25s from start/end of Wan 2.2 clips (if long enough) to remove diffusion startup wobble,
     then adjusts/pads each shot cleanly to match narration_duration + pad_after_s.
   - Renders BOTH aspect ratios:
     - 16:9 horizontal (1280x720, 24fps, H.264 yuv420p)
     - 9:16 vertical (720x1280, 24fps, H.264 yuv420p, center-cropped preserving framed subject)
3. Programmatic AI End Card (2.5 seconds):
   - Generates a clean dark-background 2.5s end card in both 16:9 and 9:16 displaying project.end_card_text
     using Pillow + FFmpeg with smooth fade-out to black.
4. Audio Spine, Ducking & Two-Pass Loudness Normalization:
   - Concatenates per-shot narration WAVs with pad_after_s silence gaps.
   - Trims/loops music score to match total runtime with 1.5s fade-in and 2.5s fade-out.
   - Ducks music under narration via FFmpeg sidechaincompress (threshold, ratio, attack, release exposed as constants/CLI flags).
   - Runs two-pass FFmpeg loudnorm targeting I=-14, TP=-1.0, LRA=11 (AAC 192k, 48kHz).
5. Burned-In English Subtitles:
   - Auto-generates outputs/final/subtitles_{aspect}.ass from shot['subtitle_en'] using exact cumulative timecodes.
   - Text wrapped to max 2 lines, <= 40 chars/line.
   - Styled with readable white font and dark outline, safe area margin for 9:16 (MarginV >= 220px).
6. Outputs final master films to outputs/final/{title}_16x9.mp4 and outputs/final/{title}_9x16.mp4,
   measuring and reporting duration, resolution, and integrated LUFS.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from typing import Any, Dict, List, Optional, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from PIL import Image, ImageDraw, ImageFont

from pipeline.utils import (
    log_run,
    probe_audio_duration,
    validate_storyboard,
)

# Default Sidechain Compression Parameters for Music Ducking
DEFAULT_DUCK_THRESHOLD = 0.02
DEFAULT_DUCK_RATIO = 6.0
DEFAULT_DUCK_ATTACK = 300.0   # ms
DEFAULT_DUCK_RELEASE = 1500.0 # ms

# Target Loudness Normalization Parameters (Broadcast / Mobile standard)
TARGET_LUFS = -14.0
TARGET_TP = -1.0
TARGET_LRA = 11.0


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


def get_ffprobe_binary() -> str:
    """Finds executable ffprobe binary."""
    ffprobe_on_path = shutil.which("ffprobe")
    if ffprobe_on_path:
        return ffprobe_on_path
    return "ffprobe"


def probe_media_duration(filepath: str) -> float:
    """Returns duration in seconds using ffprobe."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Media file missing: {filepath}")

    ffprobe_bin = get_ffprobe_binary()
    cmd = [
        ffprobe_bin,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath,
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:
        # Fallback via probe_audio_duration if audio
        return probe_audio_duration(filepath)


def measure_integrated_lufs(audio_or_video_path: str) -> float:
    """Measures integrated loudness (LUFS) of a media file via FFmpeg ebur128 filter."""
    ffmpeg_bin = get_ffmpeg_binary()
    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-i", audio_or_video_path,
        "-af", "ebur128=framelog=verbose",
        "-f", "null",
        "-",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    stderr = res.stderr

    # Parse 'Integrated loudness:    I: -14.2 LUFS'
    match = re.search(r"Integrated loudness:\s+I:\s+([-\d.]+)\s+LUFS", stderr)
    if match:
        return float(match.group(1))

    # Alternate pattern
    match2 = re.search(r"I:\s+([-\d.]+)\s+LUFS", stderr)
    if match2:
        return float(match2.group(1))

    return -99.0


def get_safe_font(font_size: int = 32) -> ImageFont.ImageFont:
    """Detects and returns a clean TrueType sans-serif font across platforms."""
    font_candidates = [
        "arial.ttf",
        "calibri.ttf",
        "DejaVuSans.ttf",
        "LiberationSans-Regular.ttf",
        "Helvetica.ttc",
        "C:\\Windows\\Fonts\\arial.ttf",
        "C:\\Windows\\Fonts\\calibri.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for cand in font_candidates:
        try:
            return ImageFont.truetype(cand, font_size)
        except Exception:
            continue
    return ImageFont.load_default()


def generate_end_card_video(
    text: str,
    output_mp4: str,
    width: int,
    height: int,
    duration_s: float = 2.5,
    fps: int = 24,
) -> None:
    """Generates a clean 2.5s dark-background end card MP4 with smooth fade-out to black."""
    os.makedirs(os.path.dirname(os.path.abspath(output_mp4)), exist_ok=True)
    temp_img_path = output_mp4 + ".png"

    # 1. Render end card image with Pillow
    img = Image.new("RGB", (width, height), color=(14, 14, 16))
    draw = ImageDraw.Draw(img)

    font_title = get_safe_font(int(height * 0.038))
    font_body = get_safe_font(int(height * 0.024))
    font_foot = get_safe_font(int(height * 0.018))

    # Center-wrap text
    wrap_width = 38 if width < height else 62
    wrapped_lines = textwrap.wrap(text, width=wrap_width)

    # Calculate vertical spacing
    line_height = int(height * 0.036)
    total_block_h = len(wrapped_lines) * line_height
    start_y = int((height - total_block_h) / 2)

    for i, line in enumerate(wrapped_lines):
        line_y = start_y + (i * line_height)
        draw.text((width / 2, line_y), line, font=font_body, fill=(210, 210, 215), anchor="mm")

    # Subtle footer attribution
    draw.text((width / 2, height - int(height * 0.06)), "Heritage Short Film Series", font=font_foot, fill=(110, 110, 120), anchor="mm")
    img.save(temp_img_path, "PNG")

    # 2. Render to MP4 with fade-out in final 1.0s
    ffmpeg_bin = get_ffmpeg_binary()
    fade_start = max(0.5, duration_s - 1.0)
    vf = f"fade=t=out:st={fade_start:.2f}:d=1.0:color=black"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-loop", "1",
        "-i", temp_img_path,
        "-vf", vf,
        "-c:v", "libx264",
        "-t", f"{duration_s:.3f}",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        output_mp4,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if os.path.exists(temp_img_path):
        os.remove(temp_img_path)

    if res.returncode != 0:
        raise RuntimeError(f"Failed to generate end card video: {res.stderr}")


def format_ass_time(seconds: float) -> str:
    """Formats float seconds into ASS timestamp format: H:MM:SS.cs"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def generate_subtitles_ass(
    shot_timings: List[Dict[str, Any]],
    output_ass_path: str,
    aspect_ratio: str = "16:9",
    width: int = 1280,
    height: int = 720,
) -> None:
    """Generates an Advanced SubStation Alpha (.ass) subtitle file.

    Ensures:
    - Text wrapped to max 2 lines, <= 40 chars/line.
    - Clean white font with dark outline and shadow.
    - Safe-area margin: for 9:16, MarginV >= 220px to stay clear of TikTok/Reels UI buttons.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_ass_path)), exist_ok=True)

    if aspect_ratio == "9:16":
        font_size = 38
        margin_v = 240 # Safe area above bottom 20% UI danger zone
    else:
        font_size = 28
        margin_v = 60  # Cinematic lower-third margin

    ass_content = f"""[Script Info]
Title: Heritage Short Subtitles ({aspect_ratio})
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
PlayResX: {width}
PlayResY: {height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,{font_size},&H00FFFFFF,&H000000FF,&H00101010,&H80000000,1,0,0,0,100,100,0,0,1,2.5,1.5,2,40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    for item in shot_timings:
        start_ts = format_ass_time(item["start_s"])
        end_ts = format_ass_time(item["end_s"])
        sub_raw = item["subtitle"]
        if not sub_raw:
            continue

        # Wrap text to max 2 lines, <= 40 chars per line
        wrapped = textwrap.wrap(sub_raw, width=38)
        ass_text = "\\N".join(wrapped[:2])

        ass_content += f"Dialogue: 0,{start_ts},{end_ts},Default,,0,0,0,,{ass_text}\n"

    with open(output_ass_path, "w", encoding="utf-8") as f:
        f.write(ass_content)


def build_audio_spine(
    shot_timings: List[Dict[str, Any]],
    music_score_path: str,
    output_mixed_audio_path: str,
    total_duration_s: float,
    duck_threshold: float = DEFAULT_DUCK_THRESHOLD,
    duck_ratio: float = DEFAULT_DUCK_RATIO,
    duck_attack: float = DEFAULT_DUCK_ATTACK,
    duck_release: float = DEFAULT_DUCK_RELEASE,
) -> float:
    """Concatenates narration audio, mixes & ducks music, and applies two-pass loudnorm normalization.

    Returns:
        Measured final integrated LUFS.
    """
    ffmpeg_bin = get_ffmpeg_binary()
    temp_dir = os.path.join("outputs", "final", "temp_audio")
    os.makedirs(temp_dir, exist_ok=True)

    # 1. Build concatenated narration spine with exact silence gaps
    # For each shot, narration starts at item['start_s'] and ends at item['audio_end_s']
    # Then pad_after_s silence follows until next shot.
    filter_complex_narration = []
    input_args_narration = []

    for i, item in enumerate(shot_timings):
        input_args_narration.extend(["-i", item["narration_wav"]])
        # Pad shot narration with silence equal to its pad_after_s
        pad_s = item["pad_after_s"]
        filter_complex_narration.append(f"[{i}:a]apad=pad_dur={pad_s:.3f}[pad_{i}];")

    concat_inputs = "".join(f"[pad_{i}]" for i in range(len(shot_timings)))
    filter_complex_narration.append(f"{concat_inputs}concat=n={len(shot_timings)}:v=0:a=1[narration_full]")

    raw_narration_path = os.path.join(temp_dir, "narration_spine.wav")
    cmd_narration = [
        ffmpeg_bin,
        "-y",
        *input_args_narration,
        "-filter_complex", "".join(filter_complex_narration),
        "-map", "[narration_full]",
        "-ar", "48000",
        raw_narration_path,
    ]
    res_narration = subprocess.run(cmd_narration, capture_output=True, text=True)
    if res_narration.returncode != 0:
        raise RuntimeError(f"Narration spine concatenation failed: {res_narration.stderr}")

    # 2. Prepare trimmed / looped music track matching total film duration with fade in/out
    faded_music_path = os.path.join(temp_dir, "music_faded.wav")
    music_fade_out_start = max(1.0, total_duration_s - 2.5)
    music_filter = (
        f"aloop=loop=-1:size=2e+09,"
        f"atrim=0:{total_duration_s:.3f},"
        f"afade=t=in:st=0:d=1.5,"
        f"afade=t=out:st={music_fade_out_start:.3f}:d=2.5"
    )
    cmd_music = [
        ffmpeg_bin,
        "-y",
        "-i", music_score_path,
        "-af", music_filter,
        "-ar", "48000",
        faded_music_path,
    ]
    res_music = subprocess.run(cmd_music, capture_output=True, text=True)
    if res_music.returncode != 0:
        raise RuntimeError(f"Music score trimming & fading failed: {res_music.stderr}")

    # 3. Duck music under narration using FFmpeg sidechaincompress
    # Input 0: Music (to be compressed/ducked), Input 1: Narration (sidechain detector)
    ducked_mix_path = os.path.join(temp_dir, "ducked_mix_pre.wav")
    duck_filter = (
        f"[0:a][1:a]sidechaincompress="
        f"threshold={duck_threshold}:"
        f"ratio={duck_ratio}:"
        f"attack={duck_attack}:"
        f"release={duck_release}[ducked_music];"
        f"[ducked_music][1:a]amix=inputs=2:duration=first:dropout_transition=2[mixed_pre]"
    )
    cmd_duck = [
        ffmpeg_bin,
        "-y",
        "-i", faded_music_path,
        "-i", raw_narration_path,
        "-filter_complex", duck_filter,
        "-map", "[mixed_pre]",
        "-ar", "48000",
        ducked_mix_path,
    ]
    res_duck = subprocess.run(cmd_duck, capture_output=True, text=True)
    if res_duck.returncode != 0:
        raise RuntimeError(f"Music ducking failed: {res_duck.stderr}")

    # 4. Two-Pass FFmpeg Loudnorm Normalization (I=-14, TP=-1.0, LRA=11)
    # Pass 1: Measure audio statistics
    cmd_pass1 = [
        ffmpeg_bin,
        "-hide_banner",
        "-i", ducked_mix_path,
        "-af", f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}:print_format=json",
        "-f", "null",
        "-",
    ]
    res_p1 = subprocess.run(cmd_pass1, capture_output=True, text=True)
    stderr_p1 = res_p1.stderr

    # Extract JSON blob from loudnorm pass 1
    json_match = re.search(r"\{\s*\"input_i\".*?\}", stderr_p1, re.DOTALL)
    if not json_match:
        # Fallback to single-pass if JSON parse fails
        af_loudnorm = f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}"
    else:
        stats = json.loads(json_match.group(0))
        measured_i = stats.get("input_i", "-24.0")
        measured_tp = stats.get("input_tp", "-2.0")
        measured_lra = stats.get("input_lra", "7.0")
        measured_thresh = stats.get("input_thresh", "-34.0")
        target_offset = stats.get("target_offset", "0.0")

        af_loudnorm = (
            f"loudnorm=I={TARGET_LUFS}:TP={TARGET_TP}:LRA={TARGET_LRA}:"
            f"measured_I={measured_i}:measured_TP={measured_tp}:measured_LRA={measured_lra}:"
            f"measured_thresh={measured_thresh}:offset={target_offset}:linear=true"
        )

    # Pass 2: Apply linear loudness normalization
    os.makedirs(os.path.dirname(os.path.abspath(output_mixed_audio_path)), exist_ok=True)
    cmd_pass2 = [
        ffmpeg_bin,
        "-y",
        "-i", ducked_mix_path,
        "-af", af_loudnorm,
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "48000",
        output_mixed_audio_path,
    ]
    res_p2 = subprocess.run(cmd_pass2, capture_output=True, text=True)
    if res_p2.returncode != 0:
        raise RuntimeError(f"Two-pass loudness normalization failed: {res_p2.stderr}")

    # Clean up intermediate wav files
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    # Measure verified loudness
    final_lufs = measure_integrated_lufs(output_mixed_audio_path)
    return final_lufs


def normalize_and_crop_shot(
    shot_mp4: str,
    output_normalized_mp4: str,
    target_duration_s: float,
    aspect_ratio: str,
    width: int,
    height: int,
    is_wan22: bool = False,
    fps: int = 24,
) -> None:
    """Trims, crops, loops, and sets resolution for a single video shot clip."""
    ffmpeg_bin = get_ffmpeg_binary()
    parent_dir = os.path.dirname(os.path.abspath(output_normalized_mp4))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    input_duration = probe_media_duration(shot_mp4)

    # If clip is Wan 2.2 and long enough (> 1.5s), trim 0.25s from start/end to remove diffusion wobble
    trim_start = 0.25 if (is_wan22 and input_duration > 1.5) else 0.0
    effective_input_dur = max(0.5, input_duration - (0.5 if (is_wan22 and input_duration > 1.5) else 0.0))

    # Video filters for 16:9 vs 9:16
    if aspect_ratio == "9:16":
        # Center-crop 9:16 keeping subject framed: scale height to 1280, crop center 720
        # scale=-2:1280,crop=720:1280:(iw-720)/2:(ih-1280)/2
        vf_crop = f"scale=-2:{height},crop={width}:{height}:(iw-{width})/2:(ih-{height})/2"
    else:
        # 16:9 standard scale & pad if needed
        vf_crop = f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"

    vf_chain = f"trim=start={trim_start:.3f},setpts=PTS-STARTPTS,{vf_crop},fps={fps}"

    # Handle loop / extension if target_duration_s > effective_input_dur
    if target_duration_s > effective_input_dur:
        # Loop video to cover duration
        loop_count = int(math.ceil(target_duration_s / effective_input_dur)) + 1
        vf_chain = f"trim=start={trim_start:.3f},setpts=PTS-STARTPTS,loop=loop={loop_count}:size=32767:start=0,{vf_crop},fps={fps}"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-i", shot_mp4,
        "-vf", vf_chain,
        "-t", f"{target_duration_s:.3f}",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-r", str(fps),
        "-an",
        output_normalized_mp4,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Shot video normalization failed for {shot_mp4}:\n{res.stderr}")


def assemble_master_film(
    storyboard: Dict[str, Any],
    aspect_ratio: str,
    duck_threshold: float,
    duck_ratio: float,
    duck_attack: float,
    duck_release: float,
    skip_subtitles: bool = False,
) -> Tuple[str, float, str, float]:
    """Assembles a complete master film for a given aspect ratio (16:9 or 9:16).

    Returns:
        (output_film_path, total_duration_s, resolution_str, measured_lufs)
    """
    project = storyboard.get("project", {})
    raw_title = project.get("title", "Heritage Short")
    clean_title = re.sub(r"[^\w\-_]", "_", raw_title).lower().strip("_")
    fps = project.get("fps", 24)
    end_card_text = project.get("end_card_text", "Created with AI-assisted generative tools.")
    shots: List[Dict[str, Any]] = storyboard.get("shots", [])

    if aspect_ratio == "9:16":
        width, height = 720, 1280
        aspect_tag = "9x16"
    else:
        width, height = 1280, 720
        aspect_tag = "16x9"

    final_dir = os.path.join("outputs", "final")
    temp_dir = os.path.join(final_dir, f"temp_{aspect_tag}")
    os.makedirs(temp_dir, exist_ok=True)

    print(f"\n================================================================================")
    print(f"       ASSEMBLING MASTER FILM: {raw_title} [{aspect_tag} - {width}x{height}]    ")
    print(f"================================================================================")

    # 1. Compute exact shot durations and cumulative timeline
    shot_timings: List[Dict[str, Any]] = []
    current_time_s = 0.0

    for shot in shots:
        shot_id = shot["id"]
        narration_wav = os.path.join("outputs", "narration", f"{shot_id}.wav")
        shot_mp4 = os.path.join("outputs", "shots", f"{shot_id}.mp4")

        if not os.path.exists(narration_wav):
            raise FileNotFoundError(f"Missing narration audio for {shot_id}: {narration_wav}")
        if not os.path.exists(shot_mp4):
            raise FileNotFoundError(f"Missing shot video clip for {shot_id}: {shot_mp4}")

        narration_dur = probe_media_duration(narration_wav)
        pad_after = float(shot.get("pad_after_s", 0.4))
        target_duration = round(narration_dur + pad_after, 3)

        shot_timings.append({
            "shot_id": shot_id,
            "narration_wav": narration_wav,
            "shot_mp4": shot_mp4,
            "narration_duration": narration_dur,
            "pad_after_s": pad_after,
            "target_duration_s": target_duration,
            "start_s": current_time_s,
            "audio_end_s": current_time_s + narration_dur,
            "end_s": current_time_s + target_duration,
            "subtitle": shot.get("subtitle_en", ""),
            "video_mode": shot.get("video_mode", "wan22"),
        })
        current_time_s += target_duration

    shots_total_duration = current_time_s
    end_card_duration = 2.5
    total_film_duration = shots_total_duration + end_card_duration

    print(f"[*] Timeline breakdown:")
    print(f"    - Spoken Shots Runtime: {shots_total_duration:.2f}s ({len(shots)} shots)")
    print(f"    - Programmatic End Card : {end_card_duration:.1f}s")
    print(f"    - Total Master Duration: {total_film_duration:.2f}s")

    # 2. Normalize and format each shot video clip
    print(f"[*] Normalizing and conform-cropping {len(shots)} video shot(s)...")
    normalized_shot_paths = []
    for item in shot_timings:
        shot_id = item["shot_id"]
        norm_mp4 = os.path.join(temp_dir, f"norm_{shot_id}.mp4")
        normalize_and_crop_shot(
            shot_mp4=item["shot_mp4"],
            output_normalized_mp4=norm_mp4,
            target_duration_s=item["target_duration_s"],
            aspect_ratio=aspect_ratio,
            width=width,
            height=height,
            is_wan22=(item["video_mode"] == "wan22"),
            fps=fps,
        )
        normalized_shot_paths.append(norm_mp4)

    # 3. Generate Programmatic AI End Card (2.5s)
    end_card_mp4 = os.path.join(temp_dir, f"end_card_{aspect_tag}.mp4")
    print(f"[*] Generating programmatic {end_card_duration}s end card...")
    generate_end_card_video(
        text=end_card_text,
        output_mp4=end_card_mp4,
        width=width,
        height=height,
        duration_s=end_card_duration,
        fps=fps,
    )
    normalized_shot_paths.append(end_card_mp4)

    # 4. Concatenate all normalized video segments into single visual track
    concat_list_file = os.path.join(temp_dir, "concat_video_list.txt")
    with open(concat_list_file, "w", encoding="utf-8") as f:
        for p in normalized_shot_paths:
            # Use forward slashes for ffmpeg concat
            f.write(f"file '{os.path.abspath(p).replace(chr(92), '/')}'\n")

    raw_video_track = os.path.join(temp_dir, f"video_track_raw_{aspect_tag}.mp4")
    ffmpeg_bin = get_ffmpeg_binary()
    cmd_concat_v = [
        ffmpeg_bin,
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_list_file,
        "-c", "copy",
        raw_video_track,
    ]
    res_concat_v = subprocess.run(cmd_concat_v, capture_output=True, text=True)
    if res_concat_v.returncode != 0:
        raise RuntimeError(f"Video concat failed: {res_concat_v.stderr}")

    # 5. Generate and Burn English Subtitles (.ass)
    ass_subtitle_path = os.path.join(final_dir, f"subtitles_{aspect_tag}.ass")
    generate_subtitles_ass(
        shot_timings=shot_timings,
        output_ass_path=ass_subtitle_path,
        aspect_ratio=aspect_ratio,
        width=width,
        height=height,
    )
    print(f"[+] Subtitles generated at: {ass_subtitle_path}")

    # 6. Build Audio Spine & Normalization (-14 LUFS)
    music_score_path = os.path.join("outputs", "music", "score_normalized.wav")
    if not os.path.exists(music_score_path):
        raise FileNotFoundError(f"Normalized music score missing: {music_score_path}. Run 'python pipeline/music.py' first.")

    mixed_audio_path = os.path.join(temp_dir, f"master_audio_{aspect_tag}.m4a")
    print(f"[*] Mixing audio spine, ducking music (threshold={duck_threshold}, ratio={duck_ratio}), and normalizing to -14 LUFS...")
    measured_lufs = build_audio_spine(
        shot_timings=shot_timings,
        music_score_path=music_score_path,
        output_mixed_audio_path=mixed_audio_path,
        total_duration_s=total_film_duration,
        duck_threshold=duck_threshold,
        duck_ratio=duck_ratio,
        duck_attack=duck_attack,
        duck_release=duck_release,
    )

    # 7. Final Mux: Video + Subtitles + Master Audio -> Final MP4
    output_master_film = os.path.join(final_dir, f"{clean_title}_{aspect_tag}.mp4")
    print(f"[*] Burning subtitles and muxing master video: {output_master_film}...")

    # Escape subtitle path for FFmpeg filter argument
    escaped_ass_path = os.path.abspath(ass_subtitle_path).replace("\\", "/").replace(":", "\\:")
    vf_final = f"ass='{escaped_ass_path}'" if not skip_subtitles else "null"

    cmd_mux = [
        ffmpeg_bin,
        "-y",
        "-i", raw_video_track,
        "-i", mixed_audio_path,
        "-vf", vf_final,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "18",
        "-c:a", "copy",
        "-shortest",
        output_master_film,
    ]
    res_mux = subprocess.run(cmd_mux, capture_output=True, text=True)
    if res_mux.returncode != 0:
        raise RuntimeError(f"Final muxing failed for {output_master_film}:\n{res_mux.stderr}")

    # Clean up temp working files
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass

    final_duration = probe_media_duration(output_master_film)
    res_str = f"{width}x{height}"

    log_run(
        stage="assemble",
        shot_id=f"master_{aspect_tag}",
        model="ffmpeg_master_assemble",
        params={
            "aspect_ratio": aspect_ratio,
            "resolution": res_str,
            "duration_s": final_duration,
            "duck_threshold": duck_threshold,
            "duck_ratio": duck_ratio,
            "measured_lufs": measured_lufs,
            "burned_subtitles": not skip_subtitles,
        },
        seed=None,
        latency_s=0.0,
        status="ok",
        est_cost_usd=0.0,
        cache_hit=False,
    )

    print(f"\n[+] SUCCESS! Master film assembled:")
    print(f"    File      : {output_master_film}")
    print(f"    Resolution: {res_str} ({aspect_ratio})")
    print(f"    Duration  : {final_duration:.2f}s")
    print(f"    Loudness  : {measured_lufs:.1f} LUFS (Broadcast standard: -14.0 LUFS)")

    return output_master_film, final_duration, res_str, measured_lufs


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble final 16:9 and 9:16 master films.")
    parser.add_argument(
        "--storyboard",
        default="storyboard.json",
        help="Path to storyboard JSON file (default: storyboard.json)",
    )
    parser.add_argument(
        "--aspect",
        choices=["16:9", "9:16", "both"],
        default="both",
        help="Aspect ratio to assemble (default: both)",
    )
    parser.add_argument(
        "--duck-threshold",
        type=float,
        default=DEFAULT_DUCK_THRESHOLD,
        help=f"Sidechain ducking threshold (default: {DEFAULT_DUCK_THRESHOLD})",
    )
    parser.add_argument(
        "--duck-ratio",
        type=float,
        default=DEFAULT_DUCK_RATIO,
        help=f"Sidechain ducking compression ratio (default: {DEFAULT_DUCK_RATIO})",
    )
    parser.add_argument(
        "--duck-attack",
        type=float,
        default=DEFAULT_DUCK_ATTACK,
        help=f"Sidechain attack in ms (default: {DEFAULT_DUCK_ATTACK})",
    )
    parser.add_argument(
        "--duck-release",
        type=float,
        default=DEFAULT_DUCK_RELEASE,
        help=f"Sidechain release in ms (default: {DEFAULT_DUCK_RELEASE})",
    )
    parser.add_argument(
        "--skip-subtitles",
        action="store_true",
        help="Skip burning in subtitles",
    )

    args = parser.parse_args()

    # 1. Validate storyboard first
    print("Validating storyboard before film assembly...")
    storyboard = validate_storyboard(args.storyboard)
    print(f"Storyboard validation passed: '{storyboard['project']['title']}'")

    targets = ["16:9", "9:16"] if args.aspect == "both" else [args.aspect]
    results = []

    for aspect in targets:
        res = assemble_master_film(
            storyboard=storyboard,
            aspect_ratio=aspect,
            duck_threshold=args.duck_threshold,
            duck_ratio=args.duck_ratio,
            duck_attack=args.duck_attack,
            duck_release=args.duck_release,
            skip_subtitles=args.skip_subtitles,
        )
        results.append(res)

    print("\n================================================================================")
    print("                         FINAL ASSEMBLY SUMMARY                                 ")
    print("================================================================================")
    for out_path, dur, res_dim, lufs in results:
        print(f"Target : {os.path.basename(out_path)}")
        print(f"  * Resolution : {res_dim}")
        print(f"  * Duration   : {dur:.2f}s")
        print(f"  * Loudness   : {lufs:.1f} LUFS\n")
    print("================================================================================\n")


if __name__ == "__main__":
    main()
