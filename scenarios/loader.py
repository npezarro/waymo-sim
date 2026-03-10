"""YAML scenario loader.

Scenarios define:
- Map geometry (roads, intersections)
- Entity spawn positions and types
- Scripted waypoints with timestamps
- Ego entity designation
- Goal position
- Simulation parameters (duration, dt)
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from engine.entity import Entity, EntityType
from engine.kinematics import BicycleModel, HolonomicWalker
from engine.world import RoadSegment, World


@dataclass
class Scenario:
    """Parsed scenario ready to be loaded into a World."""

    name: str
    description: str = ""
    dt: float = 0.1
    duration: float = 30.0
    seed: int = 42
    entities: List[Entity] = field(default_factory=list)
    roads: List[RoadSegment] = field(default_factory=list)
    goal: Optional[Tuple[float, float]] = None
    goal_radius: float = 3.0

    @property
    def max_steps(self) -> int:
        return int(self.duration / self.dt)


class ScenarioLoader:
    """Loads scenarios from YAML files."""

    SCENARIOS_DIR = Path(__file__).parent / "maps"

    @classmethod
    def load(cls, path: str) -> Scenario:
        """Load a scenario from a YAML file path."""
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls._parse(data)

    @classmethod
    def load_by_name(cls, name: str) -> Scenario:
        """Load a scenario from the built-in scenarios directory."""
        path = cls.SCENARIOS_DIR / f"{name}.yaml"
        if not path.exists():
            available = [p.stem for p in cls.SCENARIOS_DIR.glob("*.yaml")]
            raise FileNotFoundError(
                f"Scenario '{name}' not found. Available: {available}"
            )
        return cls.load(str(path))

    @classmethod
    def list_scenarios(cls) -> List[str]:
        """List all available built-in scenario names."""
        return sorted(p.stem for p in cls.SCENARIOS_DIR.glob("*.yaml"))

    @classmethod
    def _parse(cls, data: dict) -> Scenario:
        scenario = Scenario(
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
            dt=data.get("dt", 0.1),
            duration=data.get("duration", 30.0),
            seed=data.get("seed", 42),
        )

        # Parse goal
        if "goal" in data:
            g = data["goal"]
            scenario.goal = (float(g["x"]), float(g["y"]))
            scenario.goal_radius = float(g.get("radius", 3.0))

        # Parse roads
        for road_data in data.get("roads", []):
            points = [(p["x"], p["y"]) for p in road_data["points"]]
            road = RoadSegment(
                points=points,
                width=road_data.get("width", 3.5),
                lanes=road_data.get("lanes", 2),
                speed_limit=road_data.get("speed_limit", 13.9),
            )
            scenario.roads.append(road)

        # Parse entities
        for ent_data in data.get("entities", []):
            etype = EntityType(ent_data["type"])

            if etype == EntityType.PEDESTRIAN:
                kinematics = HolonomicWalker(
                    max_speed=ent_data.get("max_speed", 2.0)
                )
            else:
                kinematics = BicycleModel(
                    wheelbase=ent_data.get("wheelbase", 2.8),
                    max_speed=ent_data.get("max_speed", 30.0),
                )

            waypoints = []
            for wp in ent_data.get("waypoints", []):
                waypoints.append((
                    float(wp["t"]),
                    float(wp["x"]),
                    float(wp["y"]),
                    float(wp.get("heading", 0.0)),
                    float(wp.get("speed", 0.0)),
                ))

            entity = Entity(
                id=ent_data["id"],
                entity_type=etype,
                x=float(ent_data.get("x", 0.0)),
                y=float(ent_data.get("y", 0.0)),
                heading=float(ent_data.get("heading", 0.0)),
                speed=float(ent_data.get("speed", 0.0)),
                length=float(ent_data.get("length", 0.0)),
                width=float(ent_data.get("width", 0.0)),
                is_ego=ent_data.get("ego", False),
                kinematics=kinematics,
                waypoints=waypoints,
            )
            scenario.entities.append(entity)

        return scenario

    @classmethod
    def build_world(cls, scenario: Scenario) -> World:
        """Create a World instance from a parsed Scenario."""
        world = World(dt=scenario.dt, seed=scenario.seed)

        for road in scenario.roads:
            world.add_road(road)

        for entity in scenario.entities:
            world.add_entity(entity)

        return world
