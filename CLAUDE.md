# CLAUDE.md

## Project Overview

AR Spellcaster — cast magic spells with hand gestures. Real-time particle effects, screen shake, chromatic aberration, and synthesized sound rendered on a live webcam feed. Uses MediaPipe hand tracking for gesture recognition and a custom particle engine for visuals.

## Environment Setup

- Python 3.11+ virtualenv at `venv/`
- Activate: `source venv/bin/activate`
- Install deps: `pip install -r requirements.txt`

## Commands

```bash
# Activate environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the spellcaster
python scripts/run.py

# Run with video file
python scripts/run.py --source video.mp4

# Run without audio
python scripts/run.py --no-audio

# Record output to video
python scripts/record.py

# Run all tests
python -m pytest tests/ -v

# Lint check
ruff check src/ tests/

# Format
ruff format src/ tests/
```

## Architecture

```
src/
├── core/       — SpellEngine orchestrator, rendering pipeline
├── vision/     — Camera capture (OpenCV), MediaPipe hand tracking
├── gestures/   — Rule-based gesture classifier, state machine (tap/hold/swipe)
├── particles/  — Particle engine + emitters (burst/stream/ring/trail)
├── spells/     — 6 spells + registry with mana/cooldowns
├── effects/    — Glow, screen shake, flash, chromatic aberration
├── audio/      — Procedural sound synthesis via numpy + sounddevice
└── config.py   — Pydantic settings from YAML
```

## Key Dependencies

- opencv-python, mediapipe, numpy
- sounddevice (audio playback)
- pydantic, pyyaml, rich
- pytest, ruff
