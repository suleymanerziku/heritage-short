# Heritage Short: Multi-Stage AI Short Film Pipeline

## Overview
A modular, multi-stage automated AI short film production pipeline designed to generate consistent visual narratives, authentic narration, cinematic music, and final video assembly.

## Repository Structure
- `schema/`: JSON schemas defining data contracts (storyboards, stage outputs).
- `pipeline/`: Core execution scripts for each stage of production.
- `notebooks/`: Prototyping, prompt exploration, and evaluation notebooks.
- `outputs/`: Staged artifacts across keyframes, narration, shots, music, and final assembly.

## Prerequisites & Environment Setup
- Python 3.10+
- Virtual environment (`python -m venv .venv`)
- Configured `.env` file based on `.env.example`

## Quickstart & Makefile Commands
- `make validate`: Validate `storyboard.json` against `schema/storyboard.schema.json`.
- `make smoke`: Run end-to-end smoke test.
- `make radio`: Generate narration and assemble radio cut.
- `make film`: Run the full pipeline sequentially (validate -> narration -> keyframes -> shots -> music -> assemble).
- `make costs`: Compute and report generation costs to `COSTS.md`.

## Pipeline Workflow
1. **Storyboard Validation**: Ensure consistency and type correctness of scene specs.
2. **Narration**: Voice generation and timed subtitle generation.
3. **Radio Cut**: Rough audio cut assembly for timing checks.
4. **Keyframe Generation**: High-resolution anchor frames for shot generation.
5. **Video Shot Generation**: Motion synthesis driven by keyframes.
6. **Music Generation & Scoring**: Dynamic musical cues and soundscape.
7. **Final Assembly**: Multitrack muxing and export.

## Contributing & License
