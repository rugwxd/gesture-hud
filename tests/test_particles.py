"""Tests for the particle system."""

from __future__ import annotations

import numpy as np

from src.particles.emitters import BurstEmitter, RingEmitter, StreamEmitter, TrailEmitter
from src.particles.engine import Particle, ParticleEngine


class TestParticle:
    def test_initial_state(self):
        p = Particle(x=0.5, y=0.5)
        assert p.alive
        assert p.age == 0.0
        assert p.alpha == 1.0
        assert p.life_ratio == 0.0

    def test_update_position(self):
        p = Particle(x=0.5, y=0.5, vx=0.1, vy=-0.1, lifetime=2.0)
        p.update(1.0)
        assert abs(p.x - 0.6) < 0.001
        assert abs(p.y - 0.4) < 0.001

    def test_lifetime_expiry(self):
        p = Particle(x=0.5, y=0.5, lifetime=0.5)
        p.update(0.6)
        assert not p.alive

    def test_gravity(self):
        p = Particle(x=0.5, y=0.5, gravity=1.0, lifetime=2.0)
        p.update(1.0)
        assert p.vy > 0  # Gravity increased vertical velocity
        assert p.y > 0.5  # Moved down

    def test_drag(self):
        p = Particle(x=0.5, y=0.5, vx=1.0, drag=2.0, lifetime=2.0)
        p.update(0.5)
        assert p.vx < 1.0  # Drag slowed it down

    def test_alpha_decay(self):
        p = Particle(x=0.5, y=0.5, lifetime=1.0, decay_rate=1.0)
        p.update(0.5)
        assert p.alpha < 1.0
        assert p.alpha > 0.0

    def test_size_decay(self):
        p = Particle(x=0.5, y=0.5, size=10.0, size_decay=5.0, lifetime=2.0)
        p.update(1.0)
        assert p.size < 10.0

    def test_life_ratio(self):
        p = Particle(x=0.5, y=0.5, lifetime=1.0)
        p.update(0.5)
        assert abs(p.life_ratio - 0.5) < 0.01

    def test_acceleration(self):
        p = Particle(x=0.5, y=0.5, ax=1.0, ay=1.0, lifetime=2.0)
        p.update(1.0)
        assert p.vx > 0
        assert p.vy > 0


class TestParticleEngine:
    def test_empty_engine(self):
        engine = ParticleEngine(max_particles=100)
        assert engine.count == 0

    def test_emit_particles(self, particle_engine):
        particles = [Particle(x=0.5, y=0.5) for _ in range(10)]
        particle_engine.emit(particles)
        assert particle_engine.count == 10

    def test_max_particles_limit(self):
        engine = ParticleEngine(max_particles=5)
        particles = [Particle(x=0.5, y=0.5) for _ in range(10)]
        engine.emit(particles)
        assert engine.count == 5

    def test_update_removes_dead(self, particle_engine):
        particles = [
            Particle(x=0.5, y=0.5, lifetime=0.1),
            Particle(x=0.5, y=0.5, lifetime=10.0),
        ]
        particle_engine.emit(particles)
        particle_engine.update(0.5)
        assert particle_engine.count == 1

    def test_render_on_frame(self, particle_engine, sample_frame):
        particles = [
            Particle(x=0.5, y=0.5, color=(0, 255, 0), size=5, lifetime=1.0)
        ]
        particle_engine.emit(particles)
        result = particle_engine.render(sample_frame)
        assert result is not None
        assert result.shape == sample_frame.shape

    def test_render_circle_particle(self, sample_frame):
        engine = ParticleEngine()
        engine.emit([Particle(x=0.5, y=0.5, size=5, lifetime=1.0, shape="circle")])
        result = engine.render(sample_frame)
        assert np.any(result > 0)

    def test_render_spark_particle(self, sample_frame):
        engine = ParticleEngine()
        engine.emit([Particle(x=0.5, y=0.5, size=5, lifetime=1.0, shape="spark")])
        result = engine.render(sample_frame)
        assert np.any(result > 0)

    def test_render_line_particle(self, sample_frame):
        engine = ParticleEngine()
        engine.emit([
            Particle(x=0.5, y=0.5, vx=0.1, vy=0.1, size=5, lifetime=1.0, shape="line")
        ])
        result = engine.render(sample_frame)
        assert np.any(result > 0)

    def test_clear(self, particle_engine):
        particle_engine.emit([Particle(x=0.5, y=0.5) for _ in range(10)])
        particle_engine.clear()
        assert particle_engine.count == 0

    def test_offscreen_particles_skipped(self, sample_frame):
        engine = ParticleEngine()
        engine.emit([Particle(x=-1.0, y=-1.0, size=5, lifetime=1.0)])
        result = engine.render(sample_frame)
        assert np.array_equal(result, sample_frame)


class TestBurstEmitter:
    def test_emit_count(self):
        emitter = BurstEmitter(count=20)
        particles = emitter.emit(0.5, 0.5)
        assert len(particles) == 20

    def test_particles_centered(self):
        emitter = BurstEmitter(count=100)
        particles = emitter.emit(0.5, 0.5)
        avg_x = sum(p.x for p in particles) / len(particles)
        avg_y = sum(p.y for p in particles) / len(particles)
        assert abs(avg_x - 0.5) < 0.05
        assert abs(avg_y - 0.5) < 0.05

    def test_particles_have_velocity(self):
        emitter = BurstEmitter(count=10, speed_min=0.1, speed_max=0.3)
        particles = emitter.emit(0.5, 0.5)
        for p in particles:
            speed = (p.vx**2 + p.vy**2) ** 0.5
            assert speed > 0


class TestStreamEmitter:
    def test_emit_count(self):
        emitter = StreamEmitter(count_per_emit=8)
        particles = emitter.emit(0.5, 0.5)
        assert len(particles) == 8

    def test_directional(self):
        emitter = StreamEmitter(count_per_emit=50, direction=0, spread=0.01)
        particles = emitter.emit(0.5, 0.5)
        avg_vx = sum(p.vx for p in particles) / len(particles)
        assert avg_vx > 0  # Positive x direction


class TestRingEmitter:
    def test_emit_count(self):
        emitter = RingEmitter(count=30)
        particles = emitter.emit(0.5, 0.5)
        assert len(particles) == 30

    def test_ring_shape(self):
        emitter = RingEmitter(count=100, radius=0.1)
        particles = emitter.emit(0.5, 0.5)
        for p in particles:
            dist = ((p.x - 0.5) ** 2 + (p.y - 0.5) ** 2) ** 0.5
            assert abs(dist - 0.1) < 0.01


class TestTrailEmitter:
    def test_emit_count(self):
        emitter = TrailEmitter(count_per_emit=5)
        particles = emitter.emit(0.5, 0.5)
        assert len(particles) == 5

    def test_low_velocity(self):
        emitter = TrailEmitter(count_per_emit=20)
        particles = emitter.emit(0.5, 0.5)
        for p in particles:
            speed = (p.vx**2 + p.vy**2) ** 0.5
            assert speed < 0.1
