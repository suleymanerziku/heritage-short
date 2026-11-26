"""ACE-Step 1.5 score-candidate generator for heritage-short.

Runs on a CUDA GPU (Kaggle T4, Colab, or local) INSIDE the ACE-Step checkout's own
environment, because ACE-Step pins its own torch / transformers versions:

    cd ACE-Step-1.5 && uv run python /path/to/acestep_score.py /path/to/config.json

It is driven by a JSON config (written by notebooks/acestep_score.ipynb) and, for every
(caption variant x seed), generates one instrumental candidate, writes
`<out_dir>/score_cand_<variant>_<seed>.wav`, and appends one record per candidate to
`<runs_log>` following the runs.jsonl contract (see ARCHITECTURE.md section 5), labelled with
the model that REALLY ran. If generation fails, the candidate is logged as an error. There is
no silent synthetic fallback, so a bad run cannot be mistaken for a model result.

API used (verified against ACE-Step-1.5 @ ca1e85f, MIT licence):
    AceStepHandler().initialize_service(project_root, config_path, device, offload_to_cpu, quantization)
    generate_music(dit_handler, llm_handler=None, GenerationParams, GenerationConfig, save_dir)
The 5Hz language model is deliberately skipped (thinking=False, llm_handler=None): captions here
are fully specified, and DiT-only mode fits a 16 GB T4 comfortably.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

MIN_PEAK = 0.02  # below this the output is treated as near-silent (a known failure mode)


def detect_platform() -> str:
    """Where this GPU time was spent (stored in params.platform so the runs.jsonl schema is unchanged)."""
    env = os.environ
    if env.get("COLAB_RELEASE_TAG") or env.get("COLAB_GPU"):
        return "colab"
    if env.get("KAGGLE_KERNEL_RUN_TYPE") or env.get("KAGGLE_URL_BASE"):
        return "kaggle"
    return "local"


def log_record(path: str, **fields: Any) -> None:
    rec: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": "music",
        "shot_id": None,
        "model": fields.pop("model"),
        "params": dict(fields.pop("params"), platform=detect_platform()),
        "seed": fields.pop("seed"),
        "latency_s": round(float(fields.pop("latency_s")), 4),
        "status": fields.pop("status"),
        "gpu_min": round(float(fields.pop("gpu_min")), 4),
        "est_cost_usd": 0.0,
        "cache_hit": False,
        "error": fields.pop("error", None),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")


def gpu_vram_gb() -> Optional[float]:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / 1024**3
    except Exception:
        pass
    return None


def init_dit(dit: Any, cfg: Dict[str, Any], vram_gb: float) -> Tuple[Optional[str], bool]:
    """Initialise the DiT, stepping down to safer memory settings if a try fails."""
    model = cfg["model"]
    root = cfg["project_root"]
    small = vram_gb < 20.0  # ACE-Step docs: INT8 + CPU offload are the defaults below ~20 GB

    quant_cfg = cfg.get("quantization", "auto")
    quant = ("int8_weight_only" if small else None) if quant_cfg == "auto" else (quant_cfg or None)
    off_cfg = cfg.get("offload_to_cpu", "auto")
    offload = small if off_cfg == "auto" else bool(off_cfg)

    # Pre-Ampere GPUs (T4/P100, compute capability < 8) make ACE-Step fall back to float16, which
    # overflows to NaN latents ("Generation produced NaN or Inf latents"). bfloat16 has float32's
    # range, so force it (emulated on Turing: slower but numerically safe). The ACESTEP_DTYPE env var
    # suggested by the error message is NOT read by this ACE-Step version. Opt out: "dtype_mode": "fp16".
    try:
        import torch
        from acestep import gpu_config
        major = torch.cuda.get_device_capability()[0]
        if major < 8 and cfg.get("dtype_mode", "bf16") == "bf16":
            gpu_config.cuda_supports_bfloat16 = lambda *a, **k: True
            print(f"[dtype] compute capability {major}.x < 8.0: forcing bfloat16 instead of float16 (avoids NaN latents)")
    except Exception as exc:  # noqa: BLE001
        print(f"[dtype] could not apply bfloat16 override: {exc}")

    attempts: List[Tuple[Optional[str], bool]] = []
    for a in ((quant, offload), (None, offload), (None, True)):
        if a not in attempts:
            attempts.append(a)

    for q, off in attempts:
        try:
            res = dit.initialize_service(
                project_root=root, config_path=model, device="cuda", offload_to_cpu=off, quantization=q
            )
            ok = res[1] if isinstance(res, tuple) else bool(res)
            msg = res[0] if isinstance(res, tuple) else ""
        except Exception as exc:  # noqa: BLE001
            ok, msg = False, f"{type(exc).__name__}: {exc}"
        print(f"[init] {model} quantization={q} offload_to_cpu={off} -> {'OK' if ok else 'FAILED'} {str(msg)[:200]}")
        if ok:
            return q, off
    raise RuntimeError(f"Could not initialise {model} with any memory configuration")


def peak_and_seconds(audio: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    tensor = audio.get("tensor")
    sr = audio.get("sample_rate") or 48000
    try:
        peak = float(abs(tensor).max())
        seconds = float(tensor.shape[-1]) / float(sr)
        return peak, seconds
    except Exception:
        return None, None


def main(cfg_path: str) -> int:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    vram = gpu_vram_gb()
    if vram is None:
        print("[FATAL] No CUDA GPU detected. In Kaggle choose Accelerator = GPU T4; in Colab Runtime > GPU.", file=sys.stderr)
        return 2
    print(f"[env] GPU VRAM {vram:.1f} GB | model {cfg['model']} | steps {cfg['steps']} | mode {cfg.get('mode')}")

    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    dit = AceStepHandler()
    quant, offload = init_dit(dit, cfg, vram)

    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)
    model_label = f"ACE-Step-1.5/{cfg['model']}"
    ok_count = fail_count = 0

    for variant in cfg["variants"]:
        for seed in cfg["seeds"]:
            name = f"score_cand_{variant['id']}_{seed}.wav"
            dest = os.path.join(out_dir, name)
            params_log = {
                "variant": variant["id"],
                "caption": variant["caption"],
                "bpm": cfg["bpm"],
                "keyscale": cfg["keyscale"],
                "duration_s": cfg["duration_s"],
                "steps": cfg["steps"],
                "quantization": quant,
                "offload_to_cpu": offload,
                "instrumental": True,
                "output_file": name,
            }
            print(f"[gen] variant {variant['id']} seed {seed} ...")
            t0 = time.time()
            tmp = tempfile.mkdtemp(prefix="acestep_")
            try:
                params = GenerationParams(
                    task_type="text2music",
                    caption=variant["caption"],
                    lyrics="[Instrumental]",
                    instrumental=True,
                    bpm=int(cfg["bpm"]),
                    keyscale=cfg["keyscale"],
                    vocal_language="unknown",
                    duration=float(cfg["duration_s"]),
                    inference_steps=int(cfg["steps"]),
                    seed=int(seed),
                    thinking=False,
                    use_cot_metas=False,
                    use_cot_caption=False,
                    use_cot_language=False,
                )
                gen_cfg = GenerationConfig(
                    batch_size=1, use_random_seed=False, seeds=[int(seed)], audio_format="wav"
                )
                result = generate_music(dit, None, params, gen_cfg, save_dir=tmp)
                if not result.success or not result.audios:
                    raise RuntimeError(f"generate_music failed: {result.error or result.status_message}")
                audio = result.audios[0]
                src = audio.get("path")
                if not src or not os.path.exists(src):
                    raise RuntimeError("generate_music reported success but wrote no audio file")
                peak, seconds = peak_and_seconds(audio)
                if peak is not None and peak < MIN_PEAK:
                    raise RuntimeError(f"near-silent output (peak {peak:.4f} < {MIN_PEAK})")
                shutil.copyfile(src, dest)
                latency = time.time() - t0
                params_log.update(peak=None if peak is None else round(peak, 4),
                                  actual_duration_s=None if seconds is None else round(seconds, 2),
                                  sample_rate=audio.get("sample_rate"))
                log_record(cfg["runs_log"], model=model_label, params=params_log, seed=int(seed),
                           latency_s=latency, status="ok", gpu_min=latency / 60.0)
                ok_count += 1
                print(f"  [+] {dest} ({seconds and round(seconds, 1)}s audio, {latency:.0f}s, peak {peak})")
            except Exception as exc:  # noqa: BLE001
                latency = time.time() - t0
                log_record(cfg["runs_log"], model=model_label, params=params_log, seed=int(seed),
                           latency_s=latency, status="error", gpu_min=latency / 60.0, error=f"{type(exc).__name__}: {exc}")
                fail_count += 1
                print(f"  [-] {name}: {exc}", file=sys.stderr)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:
                    pass

    print(f"\n[done] {ok_count} candidate(s) ok, {fail_count} failed -> {out_dir}")
    return 0 if ok_count else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: acestep_score.py <config.json>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
