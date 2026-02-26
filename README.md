# Gesture HUD

Real-time Iron Man-style AR command interface powered by computer vision. Hand gestures control a holographic HUD overlay on your webcam feed, with object detection, targeting reticle, radar minimap, and sci-fi visual effects.

## Quick Start (3 commands)

```bash
git clone https://github.com/rugwxd/gesture-hud.git && cd gesture-hud
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python3 scripts/run.py
```

Your webcam opens with the full HUD overlay. Use hand gestures to control it.

## What It Does

Point your webcam at yourself. A holographic HUD appears in real-time:

```
┌──────────────────────────────────────────────────────┐
│   ◂ ◆◇◇ COMBAT ▸                                    │
│                                                      │
│                    ╋ (0.52, 0.41)     ┌─ SYSTEM ──┐  │
│                  ╱   ╲                │ FPS: 28.3  │  │
│                 ╱  ●  ╲               │ CPU: 12.4% │  │
│                 ╲     ╱               │ RAM: 45.2% │  │
│                  ╲   ╱                │ 14:23:07   │  │
│                    ╋                  └────────────┘  │
│                                                      │
│  ┌PERSON 92%──┐                                      │
│  ╔            ╗     ┌PHONE 78%──┐                    │
│  ║            ║     ╔           ╗                    │
│  ║            ║     ║           ║                    │
│  ╚            ╝     ╚           ╝                    │
│  └────────────┘     └───────────┘                    │
│                                                      │
│  ┌──RADAR──┐                                         │
│  │  N      │                                         │
│  │W ─┼─ E  │                                         │
│  │  S  ●●  │                                         │
│  └─────────┘                                         │
└──────────────────────────────────────────────────────┘
```

## Gesture Controls

| Gesture | Action |
|---------|--------|
| **Point finger** | Targeting reticle follows your index finger |
| **Fist (hold)** | Lock on to nearest detected object |
| **Open palm** | Trigger scan mode — detection wave sweeps the frame |
| **Swipe left/right** | Switch HUD modes (Combat → Scan → Navigation) |
| **Thumbs up (hold)** | Take screenshot |
| **Peace sign** | Recognized but reserved for future use |
| `q` / `ESC` | Quit |

## Three HUD Modes

### Combat Mode
Targeting reticle, object detection with holographic tags, radar minimap, system stats.

### Scan Mode
Full object detection with detailed labels, scanning wave animation on open palm, targeting reticle.

### Navigation Mode
Minimal overlay — system stats panel only. Clean view for when you just want FPS/CPU monitoring.

## Architecture

```
Webcam ──▶ Camera Module ──▶ Frame
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              Hand Tracker  Detector    Frame Buffer
              (MediaPipe)   (YOLOv8)
                    │           │
                    ▼           │
              Gesture          │
              Recognizer       │
                    │           │
                    ▼           ▼
              ┌─────────────────────┐
              │     HUD Engine      │
              │  ┌───────────────┐  │
              │  │ Targeting     │  │
              │  │ Stats Panel   │  │
              │  │ Radar Minimap │  │
              │  │ Object Tags   │  │
              │  │ Mode Menu     │  │
              │  └───────────────┘  │
              └─────────┬───────────┘
                        ▼
                   Effects Pipeline
                  (glow, scanlines,
                   holographic)
                        ▼
                    cv2.imshow()
```

### Pipeline per frame (~30 FPS)

1. **Camera** captures BGR frame, flips for mirror effect
2. **MediaPipe** extracts 21 hand landmarks per hand
3. **Gesture Recognizer** classifies pose (fist, palm, point, pinch, thumbs up)
4. **Gesture Tracker** debounces changes, detects taps/holds/swipes
5. **YOLOv8** runs detection every 3rd frame (cached between)
6. **Widget System** updates and renders all active HUD elements
7. **Effects Pipeline** applies glow, scanlines, holographic flicker
8. **Renderer** composites overlay onto camera frame with transparency

## Key Technical Details

### Gesture Recognition

Rule-based classifier using finger extension states:

```python
# Finger state detection:
# - Thumb: tip-to-wrist distance > IP-to-wrist distance
# - Other fingers: tip.y < PIP.y (tip above PIP joint)

# Gesture mapping:
# Fist     → 0 fingers extended
# Palm     → 5 fingers extended
# Point    → only index extended
# Pinch    → thumb + index tips distance < 5% of frame
# Thumbs up → only thumb extended, hand vertical
```

### Gesture State Machine

Tracks transitions over time with debouncing:
- **Tap**: gesture held < 0.3 seconds
- **Hold**: gesture held > 0.5 seconds
- **Swipe**: hand center moves > 15% of frame in < 0.3 seconds
- **Debounce**: requires 5 consecutive frames of same gesture to confirm

### Object Detection Strategy

YOLOv8 nano runs on every 3rd frame for performance. Results are cached and reused between detection frames. This keeps FPS above 25 on most hardware.

### Visual Effects

Post-processing chain applied after all widgets render:
1. **Glow**: Extract bright pixels → Gaussian blur → additive blend
2. **Scanlines**: Semi-transparent horizontal lines every N pixels
3. **Holographic**: Random brightness flicker + chromatic aberration

## Configuration

All settings in `configs/default.yaml`:

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| `camera.device_id` | `0` | Webcam device number |
| `camera.width` | `1280` | Capture resolution width |
| `camera.fps` | `30` | Target frame rate |
| `detection.model` | `yolov8n.pt` | YOLO model (n/s/m/l/x) |
| `detection.confidence_threshold` | `0.5` | Min confidence for detections |
| `detection.run_every_n_frames` | `3` | Skip frames for performance |
| `gestures.debounce_frames` | `5` | Frames to confirm gesture change |
| `hud.color_primary` | `[0, 255, 200]` | Primary HUD color (cyan-green) |
| `hud.opacity` | `0.8` | Overlay transparency |
| `effects.glow_enabled` | `true` | Enable bloom effect |
| `effects.scanlines_enabled` | `true` | Enable CRT scanlines |

## CLI Options

```bash
python scripts/run.py                      # Default: webcam + full HUD
python scripts/run.py --source video.mp4   # Use video file instead of webcam
python scripts/run.py --no-detection       # Skip YOLO (faster startup)
python scripts/run.py --config custom.yaml # Custom config file
python scripts/run.py -v                   # Verbose logging

python scripts/record.py                   # Record HUD output to video
python scripts/record.py -o demo.mp4       # Specify output file
```

## Running Tests

```bash
python -m pytest tests/ -v          # All 109 tests
python -m pytest tests/ --cov=src   # With coverage
```

Tests use synthetic frames and mock landmarks — no camera or model downloads required.

## Project Structure

```
src/
├── core/          # HUD engine orchestrator, OpenCV rendering primitives
├── vision/        # Camera capture, MediaPipe hands, YOLOv8 detection
├── gestures/      # Rule-based gesture classifier, state machine
├── hud/           # Widget system: targeting, stats, radar, tags, menu
├── effects/       # Glow, scanlines, holographic post-processing
└── config.py      # Pydantic settings loaded from YAML
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Hand Tracking | MediaPipe | 21-landmark hand detection at 30+ FPS |
| Object Detection | YOLOv8 (ultralytics) | Real-time object recognition |
| Rendering | OpenCV | Frame capture, drawing, compositing |
| Effects | NumPy + OpenCV | Glow, scanlines, holographic flicker |
| Config | Pydantic + YAML | Type-safe configuration management |
| CLI | Rich | Formatted terminal output |
| Testing | pytest | 109 tests, all synthetic data |

## Docker

```bash
docker build -t gesture-hud .
docker compose up   # Requires /dev/video0 passthrough on Linux
```

Note: Camera access inside Docker requires device passthrough. For macOS, run natively.

## License

MIT
