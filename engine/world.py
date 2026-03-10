"""World engine — deterministic state machine for the simulation.

The world is a pure step function: given current state + actions, produce next state.
No rendering, no IO. Seed-based RNG for any stochastic elements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from engine.entity import Entity, EntityType


@dataclass
class RoadSegment:
    """A road defined by a polyline (list of (x, y) points) and width."""

    points: List[Tuple[float, float]]
    width: float = 3.5  # meters per lane
    lanes: int = 2
    speed_limit: float = 13.9  # m/s (~50 km/h)


@dataclass
class WorldState:
    """Snapshot of the entire simulation state at one timestep."""

    timestep: int
    time: float
    entities: Dict[str, Entity]
    collisions: List[Tuple[str, str]]

    def to_dict(self) -> dict:
        """Serialize state for logging/replay."""
        return {
            "timestep": self.timestep,
            "time": round(self.time, 4),
            "entities": {
                eid: {
                    "x": round(e.x, 4),
                    "y": round(e.y, 4),
                    "heading": round(e.heading, 4),
                    "speed": round(e.speed, 4),
                    "collided": e.collided,
                    "type": e.entity_type.value,
                }
                for eid, e in self.entities.items()
            },
            "collisions": self.collisions,
        }


class World:
    """Deterministic 2D simulation world.

    Usage:
        world = World(dt=0.1, seed=42)
        world.add_entity(entity)
        for _ in range(100):
            state = world.step({"ego": np.array([0.0, 1.0])})
    """

    VERSION = 1

    def __init__(self, dt: float = 0.1, seed: int = 42):
        self.dt = dt
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.timestep = 0
        self.time = 0.0
        self.entities: Dict[str, Entity] = {}
        self.roads: List[RoadSegment] = []
        self.collisions: List[Tuple[str, str]] = []
        self._history: List[dict] = []

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_road(self, road: RoadSegment) -> None:
        self.roads.append(road)

    def get_ego(self) -> Optional[Entity]:
        for e in self.entities.values():
            if e.is_ego:
                return e
        return None

    def step(self, actions: Dict[str, np.ndarray]) -> WorldState:
        """Advance world by one timestep.

        Args:
            actions: mapping of entity_id -> action array.
                     Non-ego entities use scripted waypoints if available.
        """
        current_time = self.time

        for eid, entity in self.entities.items():
            if entity.is_ego:
                if eid in actions:
                    entity.step(actions[eid], self.dt)
            else:
                scripted = entity.interpolate_waypoint(current_time)
                if scripted is not None:
                    entity.step(scripted, self.dt)
                elif eid in actions:
                    entity.step(actions[eid], self.dt)
                # else: entity stays stationary

        self.collisions = self._detect_collisions()

        for id_a, id_b in self.collisions:
            self.entities[id_a].collided = True
            self.entities[id_b].collided = True

        self.timestep += 1
        self.time = round(self.timestep * self.dt, 6)

        state = WorldState(
            timestep=self.timestep,
            time=self.time,
            entities=self.entities,
            collisions=list(self.collisions),
        )

        self._history.append(state.to_dict())
        return state

    def get_observation(self, entity_id: str, radius: float = 50.0) -> dict:
        """Build an observation dict for a specific entity."""
        entity = self.entities[entity_id]
        nearby = []

        for eid, other in self.entities.items():
            if eid == entity_id:
                continue
            dist = math.sqrt(
                (other.x - entity.x) ** 2 + (other.y - entity.y) ** 2
            )
            if dist <= radius:
                nearby.append({
                    "id": eid,
                    "type": other.entity_type.value,
                    "x": other.x,
                    "y": other.y,
                    "heading": other.heading,
                    "speed": other.speed,
                    "distance": dist,
                })

        nearby.sort(key=lambda n: n["distance"])

        return {
            "position": np.array([entity.x, entity.y], dtype=np.float64),
            "velocity": np.array(
                [
                    entity.speed * math.cos(entity.heading),
                    entity.speed * math.sin(entity.heading),
                ],
                dtype=np.float64,
            ),
            "heading": entity.heading,
            "speed": entity.speed,
            "collided": entity.collided,
            "nearby_entities": nearby,
        }

    def reset(self) -> None:
        """Reset world to initial state with same seed."""
        self.rng = np.random.default_rng(self.seed)
        self.timestep = 0
        self.time = 0.0
        self.collisions = []
        self._history = []
        for entity in self.entities.values():
            entity.collided = False

    def _detect_collisions(self) -> List[Tuple[str, str]]:
        """Check all entity pairs for axis-aligned bounding box overlap,
        then do SAT (Separating Axis Theorem) for oriented boxes."""
        collisions = []
        entity_list = list(self.entities.values())

        for i in range(len(entity_list)):
            for j in range(i + 1, len(entity_list)):
                a = entity_list[i]
                b = entity_list[j]

                # Quick distance check first
                dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
                max_dim = max(a.length, a.width, b.length, b.width)
                if dist > max_dim * 2:
                    continue

                if self._sat_collision(a.get_corners(), b.get_corners()):
                    collisions.append((a.id, b.id))

        return collisions

    @staticmethod
    def _sat_collision(corners_a: np.ndarray, corners_b: np.ndarray) -> bool:
        """Separating Axis Theorem for two convex polygons."""
        for corners in [corners_a, corners_b]:
            for i in range(len(corners)):
                edge = corners[(i + 1) % len(corners)] - corners[i]
                axis = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(axis)
                if norm < 1e-10:
                    continue
                axis = axis / norm

                proj_a = corners_a @ axis
                proj_b = corners_b @ axis

                if proj_a.max() < proj_b.min() or proj_b.max() < proj_a.min():
                    return False

        return True
