"""Kinematic models for different entity types.

Two implementations:
- BicycleModel: for vehicles (steering + throttle)
- HolonomicWalker: for pedestrians (dx + dy, no heading constraint)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np


class KinematicModel(Protocol):
    """Protocol defining the interface for all kinematic models."""

    def step(
        self,
        x: float,
        y: float,
        heading: float,
        speed: float,
        action: np.ndarray,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        """Advance state by one timestep. Returns (x, y, heading, speed)."""
        ...


@dataclass
class BicycleModel:
    """Simplified bicycle model for vehicle dynamics.

    Action space: [steering_angle, acceleration]
        steering_angle: radians, clamped to [-max_steer, max_steer]
        acceleration: m/s^2, clamped to [-max_brake, max_accel]
    """

    wheelbase: float = 2.8  # meters, typical sedan
    max_steer: float = math.radians(35)
    max_speed: float = 30.0  # m/s (~108 km/h)
    max_accel: float = 4.0  # m/s^2
    max_brake: float = 8.0  # m/s^2

    def step(
        self,
        x: float,
        y: float,
        heading: float,
        speed: float,
        action: np.ndarray,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        steering = float(np.clip(action[0], -self.max_steer, self.max_steer))
        accel = float(np.clip(action[1], -self.max_brake, self.max_accel))

        new_speed = np.clip(speed + accel * dt, 0.0, self.max_speed)
        avg_speed = (speed + new_speed) / 2.0

        if abs(steering) < 1e-6:
            new_x = x + avg_speed * math.cos(heading) * dt
            new_y = y + avg_speed * math.sin(heading) * dt
            new_heading = heading
        else:
            turn_radius = self.wheelbase / math.tan(steering)
            angular_vel = avg_speed / turn_radius
            new_heading = heading + angular_vel * dt
            new_x = x + avg_speed * math.cos((heading + new_heading) / 2.0) * dt
            new_y = y + avg_speed * math.sin((heading + new_heading) / 2.0) * dt

        new_heading = (new_heading + math.pi) % (2 * math.pi) - math.pi
        return new_x, new_y, new_heading, float(new_speed)


@dataclass
class HolonomicWalker:
    """Holonomic (omnidirectional) model for pedestrians.

    Action space: [dx, dy]
        dx, dy: desired velocity in m/s, clamped to max_speed magnitude
    """

    max_speed: float = 2.0  # m/s (~7.2 km/h, brisk walk)

    def step(
        self,
        x: float,
        y: float,
        heading: float,
        speed: float,
        action: np.ndarray,
        dt: float,
    ) -> Tuple[float, float, float, float]:
        vx = float(action[0])
        vy = float(action[1])

        magnitude = math.sqrt(vx * vx + vy * vy)
        if magnitude > self.max_speed:
            scale = self.max_speed / magnitude
            vx *= scale
            vy *= scale
            magnitude = self.max_speed

        new_x = x + vx * dt
        new_y = y + vy * dt

        if magnitude > 1e-6:
            new_heading = math.atan2(vy, vx)
        else:
            new_heading = heading

        return new_x, new_y, new_heading, magnitude
