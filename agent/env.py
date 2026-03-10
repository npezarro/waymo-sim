"""Gymnasium environment wrapping the world engine.

Provides a standard RL interface:
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from engine.entity import EntityType
from engine.world import World


# Observation: [ego_x, ego_y, ego_heading, ego_speed, collided,
#               then for up to N nearby entities: rel_x, rel_y, rel_heading, rel_speed, type_onehot(3)]
MAX_NEARBY = 10
ENTITY_OBS_DIM = 7  # rel_x, rel_y, rel_heading, rel_speed, is_vehicle, is_ped, is_cyclist
EGO_OBS_DIM = 5  # x, y, heading, speed, collided
OBS_DIM = EGO_OBS_DIM + MAX_NEARBY * ENTITY_OBS_DIM

TYPE_INDEX = {
    EntityType.VEHICLE: 0,
    EntityType.PEDESTRIAN: 1,
    EntityType.CYCLIST: 2,
}


class DrivingEnv(gym.Env):
    """2D autonomous driving environment.

    Action space (continuous):
        For vehicles: [steering_angle, acceleration]
        For pedestrians: [dx, dy]

    Observation space (continuous):
        Flat vector of ego state + nearby entities.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        world: Optional[World] = None,
        max_steps: int = 500,
        goal: Optional[Tuple[float, float]] = None,
        goal_radius: float = 3.0,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.world = world or World()
        self.max_steps = max_steps
        self.goal = np.array(goal, dtype=np.float64) if goal else None
        self.goal_radius = goal_radius
        self.render_mode = render_mode
        self._renderer = None
        self._step_count = 0

        # Store initial entity states for reset
        self._initial_states: Dict[str, dict] = {}

        # Action space: [steering/dx, acceleration/dy]
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBS_DIM,),
            dtype=np.float64,
        )

    def _save_initial_states(self) -> None:
        self._initial_states = {}
        for eid, entity in self.world.entities.items():
            self._initial_states[eid] = {
                "x": entity.x,
                "y": entity.y,
                "heading": entity.heading,
                "speed": entity.speed,
            }

    def _restore_initial_states(self) -> None:
        for eid, state in self._initial_states.items():
            if eid in self.world.entities:
                entity = self.world.entities[eid]
                entity.x = state["x"]
                entity.y = state["y"]
                entity.heading = state["heading"]
                entity.speed = state["speed"]

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        if seed is not None:
            self.world.seed = seed

        self._restore_initial_states()
        self.world.reset()
        self._step_count = 0

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        ego = self.world.get_ego()
        if ego is None:
            raise RuntimeError("No ego entity in the world")

        # Scale action to actual ranges
        if ego.entity_type == EntityType.PEDESTRIAN:
            scaled = np.array([
                action[0] * ego.kinematics.max_speed,
                action[1] * ego.kinematics.max_speed,
            ])
        else:
            scaled = np.array([
                action[0] * ego.kinematics.max_steer,
                action[1] * ego.kinematics.max_accel,
            ])

        state = self.world.step({ego.id: scaled})
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(ego)
        terminated = self._is_terminated(ego)
        truncated = self._step_count >= self.max_steps
        info = self._get_info()

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        ego = self.world.get_ego()
        if ego is None:
            return np.zeros(OBS_DIM, dtype=np.float64)

        raw = self.world.get_observation(ego.id)
        obs = np.zeros(OBS_DIM, dtype=np.float64)

        # Ego state
        obs[0] = ego.x
        obs[1] = ego.y
        obs[2] = ego.heading
        obs[3] = ego.speed
        obs[4] = float(ego.collided)

        # Nearby entities (sorted by distance already)
        for i, nearby in enumerate(raw["nearby_entities"][:MAX_NEARBY]):
            offset = EGO_OBS_DIM + i * ENTITY_OBS_DIM
            obs[offset] = nearby["x"] - ego.x
            obs[offset + 1] = nearby["y"] - ego.y
            obs[offset + 2] = nearby["heading"] - ego.heading
            obs[offset + 3] = nearby["speed"]
            # One-hot entity type
            etype = EntityType(nearby["type"])
            type_idx = TYPE_INDEX.get(etype, 0)
            obs[offset + 4 + type_idx] = 1.0

        return obs

    def _compute_reward(self, ego: Entity) -> float:
        reward = 0.0

        # Collision penalty
        if ego.collided:
            reward -= 100.0

        # Progress toward goal
        if self.goal is not None:
            dist = np.linalg.norm(np.array([ego.x, ego.y]) - self.goal)
            reward -= dist * 0.01  # small penalty for distance to goal
            if dist <= self.goal_radius:
                reward += 50.0  # goal reached bonus

        # Small reward for forward motion (encourages movement)
        reward += ego.speed * 0.1

        return reward

    def _is_terminated(self, ego: Entity) -> bool:
        if ego.collided:
            return True
        if self.goal is not None:
            dist = np.linalg.norm(np.array([ego.x, ego.y]) - self.goal)
            if dist <= self.goal_radius:
                return True
        return False

    def _get_info(self) -> dict:
        ego = self.world.get_ego()
        info = {
            "timestep": self.world.timestep,
            "time": self.world.time,
            "collisions": self.world.collisions,
        }
        if ego and self.goal is not None:
            info["distance_to_goal"] = float(
                np.linalg.norm(np.array([ego.x, ego.y]) - self.goal)
            )
        return info

    def render(self):
        if self.render_mode is None:
            return None

        if self._renderer is None:
            from viewer.renderer import Renderer
            self._renderer = Renderer(self.world, mode=self.render_mode)

        return self._renderer.render(goal=self.goal)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
