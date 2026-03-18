"""Entity definition — the fundamental object in the simulation world."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from engine.kinematics import BicycleModel, HolonomicWalker, KinematicModel


class EntityType(Enum):
    VEHICLE = "vehicle"
    PEDESTRIAN = "pedestrian"
    CYCLIST = "cyclist"


# Default dimensions in meters (length, width)
DEFAULT_DIMENSIONS = {
    EntityType.VEHICLE: (4.5, 2.0),
    EntityType.PEDESTRIAN: (0.5, 0.5),
    EntityType.CYCLIST: (1.8, 0.6),
}


@dataclass
class Entity:
    """A single entity in the simulation."""

    id: str
    entity_type: EntityType
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0  # radians
    speed: float = 0.0  # m/s
    length: float = 0.0
    width: float = 0.0
    is_ego: bool = False
    collided: bool = False
    kinematics: Optional[KinematicModel] = field(default=None, repr=False)

    # Scripted waypoints: list of (time, x, y, heading, speed) tuples
    waypoints: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        if self.length == 0.0 or self.width == 0.0:
            dims = DEFAULT_DIMENSIONS.get(self.entity_type, (1.0, 1.0))
            self.length = self.length or dims[0]
            self.width = self.width or dims[1]

        if self.kinematics is None:
            if self.entity_type == EntityType.PEDESTRIAN:
                self.kinematics = HolonomicWalker()
            else:
                self.kinematics = BicycleModel()

    def step(self, action: np.ndarray, dt: float) -> None:
        """Advance this entity by one timestep using its kinematic model."""
        self.x, self.y, self.heading, self.speed = self.kinematics.step(
            self.x, self.y, self.heading, self.speed, action, dt
        )

    def get_corners(self) -> np.ndarray:
        """Return 4 corner points of the entity's bounding box as (4, 2) array."""
        cos_h = math.cos(self.heading)
        sin_h = math.sin(self.heading)
        half_l = self.length / 2.0
        half_w = self.width / 2.0

        # Corners relative to center, then rotated
        corners = np.array([
            [half_l, half_w],
            [half_l, -half_w],
            [-half_l, -half_w],
            [-half_l, half_w],
        ])
        rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])
        rotated = corners @ rot.T
        rotated[:, 0] += self.x
        rotated[:, 1] += self.y
        return rotated

    def interpolate_waypoint(self, time: float, dt: float = 0.1) -> Optional[np.ndarray]:
        """Get scripted action at given time, or None if no waypoints cover it."""
        if not self.waypoints:
            return None

        # Find surrounding waypoints
        if time <= self.waypoints[0][0]:
            wp = self.waypoints[0]
            dx = wp[1] - self.x
            dy = wp[2] - self.y
            if self.entity_type == EntityType.PEDESTRIAN:
                return np.array([dx, dy])
            target_heading = wp[3] if len(wp) > 3 else self.heading
            target_speed = wp[4] if len(wp) > 4 else self.speed
            steer = target_heading - self.heading
            accel = target_speed - self.speed
            return np.array([steer, accel])

        if time >= self.waypoints[-1][0]:
            return None  # Past last waypoint — entity stops being scripted

        # Linear interpolation between surrounding waypoints
        for i in range(len(self.waypoints) - 1):
            t0, x0, y0 = self.waypoints[i][0], self.waypoints[i][1], self.waypoints[i][2]
            t1, x1, y1 = self.waypoints[i + 1][0], self.waypoints[i + 1][1], self.waypoints[i + 1][2]
            if t0 <= time < t1:
                alpha = (time - t0) / (t1 - t0)
                target_x = x0 + alpha * (x1 - x0)
                target_y = y0 + alpha * (y1 - y0)
                dx = target_x - self.x
                dy = target_y - self.y
                if self.entity_type == EntityType.PEDESTRIAN:
                    return np.array([dx, dy])
                dist = math.sqrt(dx * dx + dy * dy)
                target_heading = math.atan2(dy, dx)
                steer = target_heading - self.heading
                steer = (steer + math.pi) % (2 * math.pi) - math.pi
                accel = (dist / dt) - self.speed  # P-controller: target speed to cover dist in one dt
                return np.array([steer, accel])

        return None
