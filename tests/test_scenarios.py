"""Tests for scenario loading and scenario-based regression tests."""

import math

import numpy as np
import pytest

from engine.entity import EntityType
from scenarios.loader import ScenarioLoader


class TestScenarioLoader:
    def test_list_scenarios(self):
        names = ScenarioLoader.list_scenarios()
        assert "straight_road" in names
        assert "intersection" in names
        assert "jaywalker" in names
        assert "lane_change" in names
        assert "pedestrian_crossing" in names

    def test_load_straight_road(self):
        s = ScenarioLoader.load_by_name("straight_road")
        assert s.name == "straight_road"
        assert s.goal is not None
        assert len(s.entities) == 1
        assert s.entities[0].is_ego

    def test_load_intersection(self):
        s = ScenarioLoader.load_by_name("intersection")
        assert len(s.entities) == 2
        assert len(s.roads) == 4

    def test_build_world(self):
        s = ScenarioLoader.load_by_name("straight_road")
        world = ScenarioLoader.build_world(s)
        assert world.get_ego() is not None
        assert len(world.roads) > 0

    def test_unknown_scenario_raises(self):
        with pytest.raises(FileNotFoundError):
            ScenarioLoader.load_by_name("nonexistent_scenario_xyz")


class TestScenarioRegressions:
    """Run each scenario with a simple policy and assert on outcomes."""

    def test_straight_road_drive_forward(self):
        """Driving straight at constant throttle should reach the goal."""
        s = ScenarioLoader.load_by_name("straight_road")
        world = ScenarioLoader.build_world(s)
        ego = world.get_ego()

        for _ in range(s.max_steps):
            state = world.step({"ego": np.array([0.0, 3.0])})
            if ego.collided:
                break

        assert not ego.collided, "Ego collided on an empty road"
        assert ego.x > 100, f"Ego only reached x={ego.x}, expected > 100"

    def test_intersection_no_immediate_collision(self):
        """Ego should be able to stop before the intersection."""
        s = ScenarioLoader.load_by_name("intersection")
        world = ScenarioLoader.build_world(s)
        ego = world.get_ego()

        # Brake hard for 5 seconds
        for _ in range(50):
            state = world.step({"ego": np.array([0.0, -8.0])})

        assert not ego.collided, "Ego collided while braking"
        assert ego.speed < 1.0, f"Ego still moving at {ego.speed} m/s after braking"

    def test_jaywalker_exists(self):
        """Jaywalker scenario has a pedestrian with waypoints."""
        s = ScenarioLoader.load_by_name("jaywalker")
        world = ScenarioLoader.build_world(s)
        jay = world.entities.get("jaywalker")
        assert jay is not None
        assert jay.entity_type == EntityType.PEDESTRIAN
        assert len(jay.waypoints) > 0

    def test_lane_change_slow_vehicle(self):
        """Slow vehicle should move forward with its waypoints."""
        s = ScenarioLoader.load_by_name("lane_change")
        world = ScenarioLoader.build_world(s)
        slow = world.entities["slow_vehicle"]
        initial_x = slow.x

        for _ in range(100):
            world.step({"ego": np.array([0.0, 0.0])})

        assert slow.x > initial_x, "Slow vehicle didn't move with waypoints"

    def test_pedestrian_crossing_ped_moves(self):
        """Pedestrians should move across the road via waypoints."""
        s = ScenarioLoader.load_by_name("pedestrian_crossing")
        world = ScenarioLoader.build_world(s)
        ped = world.entities["ped_north"]
        initial_y = ped.y

        for _ in range(80):
            world.step({"ego": np.array([0.0, -4.0])})

        assert ped.y < initial_y, "Pedestrian didn't move south"
