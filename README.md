# AR Spellcaster

Cast magic spells with hand gestures. Real-time augmented reality spell effects rendered on your webcam feed — particle systems, screen shake, chromatic aberration, procedurally generated sound, and a mana system.

## Quick Start (3 commands)

```bash
git clone https://github.com/rugwed9/gesture-hud.git && cd gesture-hud
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python3 scripts/run.py
```

Your webcam opens. Cast spells with hand gestures.

## Spells

| Spell | Gesture | Visual Effect |
|-------|---------|---------------|
| **Fireball** | Swipe up | Flaming orb projectile with ember particle trail |
| **Lightning** | Point up + hold | Electric bolt strikes from top of screen to fingertip |
| **Shield** | Fist hold | Glowing hexagonal force field around hand |
| **Force Push** | Open palm | Expanding concentric shockwave rings |
| **Teleport** | Pinch + release | Screen glitch — RGB split, scan-line displacement, static |
| **Wind** | Swipe left/right | Particle debris sweeps horizontally across screen |

Each spell triggers particles + screen effects (shake/flash/aberration) + synthesized sound.

## Architecture

```
Webcam ──▶ Camera ──▶ Frame
                         │
                         ▼
                    Hand Tracker
                    (MediaPipe)
                         │
                         ▼
                    Gesture Recognizer
                    + State Machine
                         │
                    ┌────┴────┐
                    ▼         ▼
              Spell Registry  Particle Engine
              (cast/cooldown) (emit/update/render)
                    │              │
                    ▼              ▼
              Active Spells ──▶ Particles on Frame
                    │
                    ├──▶ Screen Effects (shake, flash, aberration)
                    └──▶ Audio Player (synthesized sounds)
                              │
                              ▼
                         cv2.imshow()
```

### Pipeline per frame (~30 FPS)

1. **Camera** captures BGR frame, flips for mirror
2. **MediaPipe** extracts 21 hand landmarks (HandLandmarker Tasks API)
3. **Gesture Recognizer** classifies pose (fist, palm, point, pinch)
4. **Gesture Tracker** debounces changes, detects taps/holds/swipes
5. **Spell Registry** maps gesture events to spells, checks mana/cooldowns
6. **Active Spells** update physics, emit particles, trigger screen effects
7. **Particle Engine** simulates 2000+ particles with alpha blending
8. **Screen Effects** apply shake, flash, chromatic aberration
9. **Audio Player** synthesizes and plays spell sounds (non-blocking)

## Spell System

### Particle Engine

Each particle has position, velocity, acceleration, gravity, drag, color, size, lifetime, alpha, and shape (circle/line/spark). Four emitter types:

- **BurstEmitter** — radial explosion for impacts
- **StreamEmitter** — directional stream for fire/beams
- **RingEmitter** — expanding ring for shockwaves
- **TrailEmitter** — trailing particles for motion effects

### Screen Effects

Post-processing applied to the full frame after compositing:

- **Screen Shake** — random translation offset with linear decay
- **Flash** — additive color overlay with quadratic fade
- **Chromatic Aberration** — RGB channel split for glitch/impact

### Audio

All sounds are procedurally generated with numpy — no .wav files:

- Fireball: rising noise sweep + bass rumble
- Lightning: white noise crackle + electric buzz
- Shield: resonant harmonic hum with vibrato
- Force Push: deep bass thud + air whoosh
- Teleport: descending pitch sweep into digital glitch
- Wind: filtered noise with slow amplitude modulation

### Mana System

- Spells cost mana (10-25 per spell)
- Mana regenerates at 8/second
- Visual mana bar at bottom of screen
- Per-spell cooldown timers prevent spam

## Configuration

All settings in `configs/default.yaml`:

| Setting | Default | What It Controls |
|---------|---------|-----------------|
| `camera.device_id` | `0` | Webcam device number |
| `camera.width` | `1280` | Capture resolution |
| `camera.fps` | `30` | Target frame rate |
| `particles.max_particles` | `2000` | Max simultaneous particles |
| `particles.glow_enabled` | `true` | Bloom effect on particles |
| `spells.max_mana` | `100` | Mana pool size |
| `spells.mana_regen` | `8.0` | Mana per second |
| `audio.enabled` | `true` | Enable spell sounds |
| `audio.volume` | `0.5` | Sound volume (0-1) |
| `gestures.debounce_frames` | `5` | Frames to confirm gesture |

## CLI Options

```bash
python scripts/run.py                      # Default: webcam + spells
python scripts/run.py --source video.mp4   # Use video file
python scripts/run.py --no-audio           # Disable sound
python scripts/run.py --config custom.yaml # Custom config
python scripts/run.py -v                   # Verbose logging

python scripts/record.py                   # Record output to video
python scripts/record.py -o demo.mp4       # Specify output file
```

## Running Tests

```bash
python -m pytest tests/ -v          # All 155 tests
python -m pytest tests/ --cov=src   # With coverage
```

Tests use synthetic frames and mock data — no camera or model downloads required.

## Project Structure

```
src/
├── core/       — SpellEngine orchestrator, rendering pipeline
├── vision/     — Camera capture, MediaPipe hand tracking
├── gestures/   — Gesture recognition, state machine (tap/hold/swipe)
├── particles/  — Particle engine, emitters (burst/stream/ring/trail)
├── spells/     — Spell implementations + registry with mana/cooldowns
├── effects/    — Glow, screen shake, flash, chromatic aberration
├── audio/      — Procedural sound synthesis + playback
└── config.py   — Pydantic settings from YAML
```

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Hand Tracking | MediaPipe | 21-landmark hand detection |
| Rendering | OpenCV + NumPy | Frame capture, particle blending |
| Particles | Custom engine | 2000+ particles at 30fps |
| Audio | NumPy + sounddevice | Procedural spell sounds |
| Config | Pydantic + YAML | Type-safe settings |
| CLI | Rich | Terminal formatting |
| Testing | pytest | 155 tests |

## Docker

```bash
docker build -t gesture-hud .
docker compose up   # Requires /dev/video0 passthrough on Linux
```

Note: Camera and audio access inside Docker requires device passthrough. For macOS, run natively.

## License

MIT
