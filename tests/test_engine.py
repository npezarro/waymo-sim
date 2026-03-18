"""Tests for the world engine: kinematics, entities, collisions, determinism."""

import math

import numpy as np
import pytest

from engine.entity import Entity, EntityType
from engine.kinematics import BicycleModel, HolonomicWalker
from engine.world import World


class TestBicycleModel:
    def test_straight_line(self):
        model = BicycleModel()
        x, y, h, s = model.step(0, 0, 0, 10.0, np.array([0.0, 0.0]), 0.1)
        assert x == pytest.approx(1.0, abs=0.01)
        assert y == pytest.approx(0.0, abs=0.01)
        assert h == pytest.approx(0.0, abs=0.01)
        assert s == pytest.approx(10.0, abs=0.01)

    def test_acceleration(self):
        model = BicycleModel()
        _, _, _, s = model.step(0, 0, 0, 0.0, np.array([0.0, 4.0]), 0.1)
        assert s == pytest.approx(0.4, abs=0.01)

    def test_braking_clamps_at_zero(self):
        model = BicycleModel()
        _, _, _, s = model.step(0, 0, 0, 1.0, np.array([0.0, -8.0]), 0.5)
        assert s >= 0.0

    def test_steering_changes_heading(self):
        model = BicycleModel()
        _, _, h, _ = model.step(0, 0, 0, 10.0, np.array([0.3, 0.0]), 0.1)
        assert h != 0.0

    def test_max_speed_clamp(self):
        model = BicycleModel(max_speed=20.0)
        _, _, _, s = model.step(0, 0, 0, 19.5, np.array([0.0, 4.0]), 1.0)
        assert s <= 20.0


class TestHolonomicWalker:
    def test_move_north(self):
        model = HolonomicWalker()
        x, y, h, s = model.step(0, 0, 0, 0, np.array([0.0, 1.5]), 1.0)
        assert x == pytest.approx(0.0, abs=0.01)
        assert y == pytest.approx(1.5, abs=0.01)
        assert h == pytest.approx(math.pi / 2, abs=0.1)

    def test_speed_clamped(self):
        model = HolonomicWalker(max_speed=2.0)
        _, _, _, s = model.step(0, 0, 0, 0, np.array([5.0, 5.0]), 1.0)
        assert s <= 2.0 + 0.01

    def test_stationary_preserves_heading(self):
        model = HolonomicWalker()
        _, _, h, _ = model.step(0, 0, 1.0, 0, np.array([0.0, 0.0]), 1.0)
        assert h == pytest.approx(1.0, abs=0.01)


class TestEntity:
    def test_corners_axis_aligned(self):
        e = Entity(id="car", entity_type=EntityType.VEHICLE, x=0, y=0, heading=0, length=4.0, width=2.0)
        corners = e.get_corners()
        assert corners.shape == (4, 2)
        assert corners[:, 0].max() == pytest.approx(2.0, abs=0.01)
        assert corners[:, 0].min() == pytest.approx(-2.0, abs=0.01)

    def test_step_moves_entity(self):
        e = Entity(id="car", entity_type=EntityType.VEHICLE, x=0, y=0, heading=0, speed=10)
        e.step(np.array([0.0, 0.0]), 0.1)
        assert e.x > 0


class TestWorld:
    def test_step_advances_time(self):
        w = World(dt=0.1)
        ego = Entity(id="ego", entity_type=EntityType.VEHICLE, is_ego=True)
        w.add_entity(ego)
        state = w.step({"ego": np.array([0.0, 1.0])})
        assert state.timestep == 1
        assert state.time == pytest.approx(0.1)

    def test_collision_detection(self):
        w = World(dt=0.1)
        a = Entity(id="a", entity_type=EntityType.VEHICLE, x=0, y=0, heading=0, is_ego=True)
        b = Entity(id="b", entity_type=EntityType.VEHICLE, x=1.0, y=0, heading=0)
        w.add_entity(a)
        w.add_entity(b)
        state = w.step({"a": np.array([0.0, 0.0])})
        assert len(state.collisions) > 0

    def test_no_collision_when_far_apart(self):
        w = World(dt=0.1)
        a = Entity(id="a", entity_type=EntityType.VEHICLE, x=0, y=0, heading=0, is_ego=True)
        b = Entity(id="b", entity_type=EntityType.VEHICLE, x=100, y=100, heading=0)
        w.add_entity(a)
        w.add_entity(b)
        state = w.step({"a": np.array([0.0, 0.0])})
        assert len(state.collisions) == 0

    def test_determinism(self):
        """Same seed + same actions = identical trajectories."""
        trajectories = []
        for _ in range(2):
            w = World(dt=0.1, seed=123)
            ego = Entity(id="ego", entity_type=EntityType.VEHICLE, x=0, y=0, heading=0, speed=0, is_ego=True)
            w.add_entity(ego)
            traj = []
            for _ in range(50):
                state = w.step({"ego": np.array([0.1, 2.0])})
                e = state.entities["ego"]
                traj.append((e.x, e.y, e.heading, e.speed))
            trajectories.append(traj)

        for step_a, step_b in zip(trajectories[0], trajectories[1]):
            for va, vb in zip(step_a, step_b):
                assert va == pytest.approx(vb, abs=1e-10)

    def test_observation(self):
        w = World(dt=0.1)
        ego = Entity(id="ego", entity_type=EntityType.VEHICLE, x=0, y=0, is_ego=True)
        other = Entity(id="other", entity_type=EntityType.PEDESTRIAN, x=10, y=0)
        w.add_entity(ego)
        w.add_entity(other)
        obs = w.get_observation("ego")
        assert "position" in obs
        assert "nearby_entities" in obs
        assert len(obs["nearby_entities"]) == 1
        assert obs["nearby_entities"][0]["id"] == "other"

    def test_history_recorded(self):
        w = World(dt=0.1)
        ego = Entity(id="ego", entity_type=EntityType.VEHICLE, is_ego=True)
        w.add_entity(ego)
        for _ in range(5):
            w.step({"ego": np.array([0.0, 1.0])})
        assert len(w._history) == 5

    def test_reset_restores_entity_positions(self):
        """reset() must restore entities to their initial positions."""
        w = World(dt=0.1)
        ego = Entity(
            id="ego", entity_type=EntityType.VEHICLE,
            x=5.0, y=10.0, heading=1.0, speed=3.0, is_ego=True,
        )
        w.add_entity(ego)
        # Move the entity
        for _ in range(20):
            w.step({"ego": np.array([0.1, 2.0])})
        assert ego.x != pytest.approx(5.0, abs=0.1)
        # Reset and verify restoration
        w.reset()
        assert ego.x == pytest.approx(5.0)
        assert ego.y == pytest.approx(10.0)
        assert ego.heading == pytest.approx(1.0)
        assert ego.speed == pytest.approx(3.0)
        assert ego.collided is False
        assert w.timestep == 0
        assert w.time == 0.0
        assert len(w._history) == 0

    def test_interpolate_waypoint_uses_dt(self):
        """P-controller in interpolate_waypoint must use actual dt, not hardcoded 0.1."""
        entity = Entity(
            id="npc", entity_type=EntityType.VEHICLE,
            x=0.0, y=0.0, heading=0.0, speed=0.0,
        )
        entity.waypoints = [
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (2.0, 10.0, 0.0, 0.0, 0.0),
        ]
        # At time=1.0, target_x=5.0, entity at x=0 → dist=5.0
        # With dt=0.1: accel = 5.0/0.1 - 0.0 = 50.0
        # With dt=0.5: accel = 5.0/0.5 - 0.0 = 10.0
        action_small_dt = entity.interpolate_waypoint(1.0, dt=0.1)
        action_large_dt = entity.interpolate_waypoint(1.0, dt=0.5)
        # The acceleration should differ when dt differs
        assert action_small_dt is not None
        assert action_large_dt is not None
        assert action_small_dt[1] != pytest.approx(action_large_dt[1], abs=0.01)
