"""Shared core utilities for the heritage-short AI short film pipeline.

Provides validation, content hashing, cache sidecars, exponential retry logic,
execution logging, and media probing.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union


def validate_storyboard(
    storyboard_path: str = "storyboard.json",
    schema_path: str = "schema/storyboard.schema.json",
) -> Dict[str, Any]:
    """Validates a storyboard JSON file against a jsonschema specification.

    Args:
        storyboard_path: File path to the storyboard JSON.
        schema_path: File path to the storyboard schema JSON.

    Returns:
        The loaded storyboard dict.

    Raises:
        FileNotFoundError: If either the storyboard or schema file cannot be found.
        ValueError: If validation fails, detailing each failing field path and error message.
    """
    import jsonschema

    if not os.path.exists(storyboard_path):
        raise FileNotFoundError(f"Storyboard file not found at: {storyboard_path}")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema file not found at: {schema_path}")

    with open(storyboard_path, "r", encoding="utf-8") as f:
        storyboard = json.load(f)

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    validator_cls = jsonschema.validators.validator_for(schema)
    validator = validator_cls(schema)

    errors = sorted(validator.iter_errors(storyboard), key=lambda e: list(e.path))
    if errors:
        error_lines = []
        for err in errors:
            field_loc = " -> ".join(str(p) for p in err.path) if err.path else "root"
            error_lines.append(f"Field '{field_loc}': {err.message}")
        raise ValueError(
            f"Storyboard validation failed for '{storyboard_path}' against '{schema_path}':\n"
            + "\n".join(f"  * {line}" for line in error_lines)
        )

    return storyboard


def compute_input_hash(payload: Dict[str, Any]) -> str:
    """Computes a deterministic 12-hex-character SHA-256 hash of a payload dictionary.

    Args:
        payload: Input dictionary (parameters, prompts, seeds, etc.).

    Returns:
        First 12 hex characters of the SHA-256 digest.
    """
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]


def check_cache(output_path: str, input_hash: str) -> bool:
    """Checks whether a valid cached artifact exists for the given input hash.

    Returns True ONLY if both output_path and its sidecar meta (<output_path>.meta.json)
    exist and the stored input_hash matches.

    Args:
        output_path: Target artifact file path.
        input_hash: 12-char SHA-256 input hash to compare against.

    Returns:
        True if the cache is valid and matching, False otherwise.
    """
    meta_path = f"{output_path}.meta.json"
    if not os.path.exists(output_path) or not os.path.exists(meta_path):
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return meta.get("input_hash") == input_hash
    except Exception:
        return False


def write_cache_meta(
    output_path: str,
    input_hash: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Writes a cache metadata sidecar file for an output artifact.

    Args:
        output_path: Path of the generated output file.
        input_hash: 12-char SHA-256 input hash.
        extra: Optional additional metadata dict to store alongside input_hash.

    Returns:
        Path to the written sidecar file (<output_path>.meta.json).
    """
    meta_path = f"{output_path}.meta.json"
    parent_dir = os.path.dirname(os.path.abspath(meta_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    meta_payload: Dict[str, Any] = {"input_hash": input_hash}
    if extra:
        meta_payload.update(extra)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_payload, f, indent=2, ensure_ascii=False)

    return meta_path


def with_retries(
    fn: Callable[..., Any],
    max_retries: int = 3,
    base_delay: float = 2.0,
    retryable_exceptions: Union[Type[Exception], Tuple[Type[Exception], ...]] = (Exception,),
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Executes a callable with exponential backoff on retryable exceptions.

    Retries with delays: base_delay * (2 ** attempt), e.g. 2s, 4s, 8s by default.

    Args:
        fn: Function or callable to execute.
        max_retries: Maximum number of retry attempts.
        base_delay: Initial retry delay in seconds.
        retryable_exceptions: Tuple or single exception class to trigger retries.
        *args: Positional arguments forwarded to fn.
        **kwargs: Keyword arguments forwarded to fn.

    Returns:
        Result of successful fn execution.

    Raises:
        The last encountered exception if all retries are exhausted.
    """
    if isinstance(retryable_exceptions, type) and issubclass(retryable_exceptions, Exception):
        retryable_exceptions = (retryable_exceptions,)
    elif not isinstance(retryable_exceptions, tuple):
        retryable_exceptions = tuple(retryable_exceptions)

    last_exception: Optional[Exception] = None
    fn_name = getattr(fn, "__name__", repr(fn))

    for attempt in range(max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except retryable_exceptions as e:
            last_exception = e
            if attempt >= max_retries:
                print(
                    f"[ERROR] with_retries: All {max_retries} retries exhausted for '{fn_name}'. "
                    f"Final error: {e}",
                    file=sys.stderr,
                )
                raise
            delay = base_delay * (2 ** attempt)
            print(
                f"[WARNING] with_retries: Attempt {attempt + 1}/{max_retries} for '{fn_name}' "
                f"failed with {type(e).__name__}: {e}. Retrying in {delay:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(delay)

    if last_exception is not None:
        raise last_exception


def log_run(
    stage: str,
    shot_id: Optional[str],
    model: str,
    params: Any,
    seed: Optional[int],
    latency_s: float,
    status: str = "ok",
    gpu_min: float = 0.0,
    est_cost_usd: float = 0.0,
    cache_hit: bool = False,
    error: Optional[str] = None,
    log_path: str = "runs.jsonl",
) -> Dict[str, Any]:
    """Appends a single compact JSON record with an ISO-8601 UTC timestamp to runs.jsonl.

    Should be called when an actual model call is attempted (or when recording cached status).

    Args:
        stage: Pipeline stage name (e.g. 'keyframes', 'narration', 'shots').
        shot_id: Shot identifier (e.g. 'shot_01') or None.
        model: Model identifier string.
        params: Key parameters used in the model invocation.
        seed: Random seed or None.
        latency_s: Execution duration in seconds.
        status: Run status ('ok', 'error', 'cached').
        gpu_min: Estimated GPU duration in minutes.
        est_cost_usd: Estimated cost in USD.
        cache_hit: Whether this run was served from cache.
        error: Error message string if failed, else None.
        log_path: Destination JSONL file path (default 'runs.jsonl').

    Returns:
        The recorded dictionary.
    """
    record: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "shot_id": shot_id,
        "model": model,
        "params": params,
        "seed": seed,
        "latency_s": round(float(latency_s), 4),
        "status": status,
        "gpu_min": round(float(gpu_min), 4),
        "est_cost_usd": round(float(est_cost_usd), 6),
        "cache_hit": bool(cache_hit),
        "error": str(error) if error is not None else None,
    }

    log_dir = os.path.dirname(os.path.abspath(log_path))
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    return record


def probe_audio_duration(filepath: str) -> float:
    """Uses ffprobe via subprocess to return the exact duration in seconds of an audio/video file.

    Args:
        filepath: Path to the media file.

    Returns:
        Duration in seconds as a float.

    Raises:
        FileNotFoundError: If filepath or the ffprobe binary is not found.
        RuntimeError: If ffprobe returns an error or invalid output.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Media file not found: {filepath}")

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        filepath,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        raw_output = result.stdout.strip()

        if not raw_output or raw_output == "N/A":
            # Fallback to audio stream duration
            cmd_stream = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a:0",
                "-show_entries",
                "stream=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                filepath,
            ]
            result_stream = subprocess.run(cmd_stream, capture_output=True, text=True, check=True)
            raw_output = result_stream.stdout.strip()

        return float(raw_output)

    except FileNotFoundError as e:
        # Fallback 1: Try ffmpeg -i parsing if ffmpeg is available
        try:
            import re
            import shutil
            ff_bin = shutil.which("ffmpeg")
            if not ff_bin:
                try:
                    import imageio_ffmpeg
                    ff_bin = imageio_ffmpeg.get_ffmpeg_exe()
                except Exception:
                    ff_bin = None
            if ff_bin:
                res_ff = subprocess.run([ff_bin, "-i", filepath], capture_output=True, text=True)
                match = re.search(r"Duration:\s*(\d+):(\d+):([\d\.]+)", res_ff.stderr)
                if match:
                    h, m, s = match.groups()
                    return float(h) * 3600.0 + float(m) * 60.0 + float(s)
        except Exception:
            pass

        # Fallback 2: For WAV files, use standard library wave module
        if filepath.lower().endswith(".wav"):
            try:
                import wave
                with wave.open(filepath, "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        return float(frames) / float(rate)
            except Exception:
                pass

        raise FileNotFoundError(
            "ffprobe binary not found in system PATH. Ensure ffmpeg/ffprobe is installed."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffprobe failed for '{filepath}' with exit code {e.returncode}: {e.stderr.strip()}"
        ) from e
    except ValueError as e:
        raise RuntimeError(
            f"Unable to parse audio duration from ffprobe output '{raw_output}' for '{filepath}'"
        ) from e


if __name__ == "__main__":
    import tempfile

    def run_self_tests() -> None:
        print("Running self-test for pipeline/utils.py...")

        # 1. Test compute_input_hash
        sample_payload = {"shot_id": "shot_01", "prompt": "Ancient temple at sunrise", "seed": 42}
        hash_val = compute_input_hash(sample_payload)
        assert len(hash_val) == 12, f"Expected 12-char hash, got: {hash_val}"
        assert hash_val == compute_input_hash(sample_payload), "Hash should be deterministic"
        print(f"  [OK] compute_input_hash: {hash_val}")

        # 2. Test check_cache and write_cache_meta
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_out = os.path.join(tmp_dir, "test_asset.png")
            assert not check_cache(test_out, hash_val), "Cache should be False before creation"

            # Create output artifact and meta sidecar
            with open(test_out, "w", encoding="utf-8") as f:
                f.write("fake image data")

            meta_path = write_cache_meta(test_out, hash_val, extra={"model": "test-model"})
            assert os.path.exists(meta_path), "Meta sidecar file must exist"
            assert check_cache(test_out, hash_val), "Cache check should return True with matching hash"
            assert not check_cache(test_out, "differenthash"), "Cache check should fail with mismatched hash"
            print("  [OK] check_cache & write_cache_meta verified.")

            # 3. Test log_run
            test_log = os.path.join(tmp_dir, "test_runs.jsonl")
            rec = log_run(
                stage="test_stage",
                shot_id="shot_01",
                model="gemini-2.5-flash",
                params={"temperature": 0.7},
                seed=42,
                latency_s=1.234,
                status="ok",
                gpu_min=0.0,
                est_cost_usd=0.0005,
                cache_hit=False,
                log_path=test_log,
            )
            assert os.path.exists(test_log), "Log file must exist"
            with open(test_log, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            assert len(lines) == 1, "Expected 1 log entry"
            assert lines[0]["shot_id"] == "shot_01"
            assert "timestamp" in lines[0]
            print("  [OK] log_run verified.")

        # 4. Test with_retries
        attempts = 0

        def flaky_func(val: int) -> int:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise ConnectionError(f"Temporary failure {attempts}")
            return val * 2

        res = with_retries(flaky_func, max_retries=3, base_delay=0.05, retryable_exceptions=(ConnectionError,), val=21)
        assert res == 42, f"Expected 42, got {res}"
        assert attempts == 3, f"Expected 3 attempts, got {attempts}"
        print("  [OK] with_retries verified.")

        # 5. Test validate_storyboard
        with tempfile.TemporaryDirectory() as tmp_dir:
            dummy_schema_path = os.path.join(tmp_dir, "schema.json")
            dummy_storyboard_path = os.path.join(tmp_dir, "storyboard.json")

            dummy_schema = {
                "type": "object",
                "required": ["title", "shots"],
                "properties": {
                    "title": {"type": "string"},
                    "shots": {"type": "array", "items": {"type": "object", "required": ["id"], "properties": {"id": {"type": "string"}}}}
                }
            }
            with open(dummy_schema_path, "w", encoding="utf-8") as f:
                json.dump(dummy_schema, f)

            # Valid storyboard
            with open(dummy_storyboard_path, "w", encoding="utf-8") as f:
                json.dump({"title": "Test Film", "shots": [{"id": "shot_01"}]}, f)

            validated = validate_storyboard(dummy_storyboard_path, dummy_schema_path)
            assert validated["title"] == "Test Film"

            # Invalid storyboard
            with open(dummy_storyboard_path, "w", encoding="utf-8") as f:
                json.dump({"title": 123}, f)

            try:
                validate_storyboard(dummy_storyboard_path, dummy_schema_path)
                assert False, "Should have raised ValueError on invalid schema"
            except ValueError as ve:
                assert "validation failed" in str(ve).lower()

            print("  [OK] validate_storyboard verified.")

        print("All self-tests passed successfully!")

    run_self_tests()
