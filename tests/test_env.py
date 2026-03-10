"""Tests for the gymnasium environment wrapper."""

import numpy as np
import pytest

from agent.env import DrivingEnv, OBS_DIM
from engine.entity import Entity, EntityType
from engine.world import World
from scenarios.loader import ScenarioLoader


class TestDrivingEnv:
    def _make_env(self) -> DrivingEnv:
        s = ScenarioLoader.load_by_name("straight_road")
        world = ScenarioLoader.build_world(s)
        env = DrivingEnv(world=world, goal=s.goal, max_steps=200)
        env._save_initial_states()
        return env

    def test_reset_returns_obs(self):
        env = self._make_env()
        obs, info = env.reset()
        assert obs.shape == (OBS_DIM,)
        assert "timestep" in info

    def test_step_returns_tuple(self):
        env = self._make_env()
        env.reset()
        action = np.array([0.0, 0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_truncation_at_max_steps(self):
        env = DrivingEnv(max_steps=5)
        ego = Entity(id="ego", entity_type=EntityType.VEHICLE, is_ego=True)
        env.world.add_entity(ego)
        env._save_initial_states()
        env.reset()

        for i in range(10):
            _, _, terminated, truncated, _ = env.step(np.array([0.0, 0.0]))
            if truncated or terminated:
                break

        assert truncated or terminated
        assert i <= 5

    def test_action_space_valid(self):
        env = self._make_env()
        assert env.action_space.shape == (2,)
        sample = env.action_space.sample()
        assert sample.shape == (2,)

    def test_observation_space_valid(self):
        env = self._make_env()
        assert env.observation_space.shape == (OBS_DIM,)

    def test_multiple_resets(self):
        env = self._make_env()
        for _ in range(3):
            obs, _ = env.reset()
            assert obs.shape == (OBS_DIM,)
            for _ in range(10):
                env.step(np.array([0.0, 1.0]))
