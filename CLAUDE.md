# CLAUDE.md

## Project Overview

Gesture HUD is a real-time Iron Man-style AR command interface. It uses a webcam + MediaPipe hand tracking + YOLOv8 object detection to overlay a holographic HUD controlled entirely by hand gestures.

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

# Run the HUD
python scripts/run.py

# Run with video file
python scripts/run.py --source video.mp4

# Run without object detection (faster)
python scripts/run.py --no-detection

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
├── core/       — Main HUD engine, OpenCV rendering pipeline
├── vision/     — Camera capture, MediaPipe hands, YOLOv8 detection
├── gestures/   — Gesture recognition, state machine (tap/hold/swipe)
├── hud/        — Widget system: targeting, stats, radar, object tags, menu
├── effects/    — Post-processing: glow, scanlines, holographic flicker
└── config.py   — Pydantic settings from YAML
```

## Key Dependencies

- opencv-python, mediapipe, ultralytics (YOLOv8)
- numpy, psutil, pydantic, pyyaml, rich
- pytest, ruff
