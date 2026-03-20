"""Pygame-based 2D renderer for the simulation.

Read-only — never modifies world state. Can run headless (rgb_array mode)
for CI/batch rendering.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from engine.entity import Entity, EntityType
from engine.world import World

# Colors (RGB)
COLOR_BG = (40, 40, 40)
COLOR_ROAD = (80, 80, 80)
COLOR_ROAD_LINE = (200, 200, 200)
COLOR_VEHICLE = (50, 150, 255)
COLOR_EGO = (0, 255, 100)
COLOR_PEDESTRIAN = (255, 200, 50)
COLOR_CYCLIST = (255, 100, 50)
COLOR_GOAL = (255, 50, 50)
COLOR_COLLISION = (255, 0, 0)
COLOR_TEXT = (220, 220, 220)

ENTITY_COLORS = {
    EntityType.VEHICLE: COLOR_VEHICLE,
    EntityType.PEDESTRIAN: COLOR_PEDESTRIAN,
    EntityType.CYCLIST: COLOR_CYCLIST,
}

# Display settings
WINDOW_WIDTH = 1024
WINDOW_HEIGHT = 768
PIXELS_PER_METER = 8.0


class Renderer:
    """Renders the simulation world using pygame."""

    def __init__(
        self,
        world: World,
        mode: str = "human",
        width: int = WINDOW_WIDTH,
        height: int = WINDOW_HEIGHT,
        ppm: float = PIXELS_PER_METER,
    ):
        self.world = world
        self.mode = mode
        self.width = width
        self.height = height
        self.ppm = ppm
        self._screen = None
        self._clock = None
        self._font = None
        self._initialized = False

    def _init_pygame(self):
        if self._initialized:
            return

        import pygame

        pygame.init()

        if self.mode == "human":
            self._screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Waymo-Sim")
        else:
            self._screen = pygame.Surface((self.width, self.height))

        self._clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("monospace", 14)
        self._initialized = True

    def _world_to_screen(
        self, wx: float, wy: float, camera_x: float, camera_y: float
    ) -> Tuple[int, int]:
        sx = int((wx - camera_x) * self.ppm + self.width / 2)
        sy = int(self.height / 2 - (wy - camera_y) * self.ppm)
        return sx, sy

    def render(
        self,
        goal: Optional[np.ndarray] = None,
        fps: int = 10,
    ) -> Optional[np.ndarray]:
        import pygame

        self._init_pygame()

        # Camera follows ego
        ego = self.world.get_ego()
        cam_x = ego.x if ego else 0.0
        cam_y = ego.y if ego else 0.0

        self._screen.fill(COLOR_BG)

        # Draw roads
        for road in self.world.roads:
            if len(road.points) < 2:
                continue
            screen_pts = [
                self._world_to_screen(p[0], p[1], cam_x, cam_y)
                for p in road.points
            ]
            half_w = road.width * self.ppm / 2

            for i in range(len(screen_pts) - 1):
                p1 = screen_pts[i]
                p2 = screen_pts[i + 1]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length < 1:
                    continue
                nx = -dy / length * half_w
                ny = dx / length * half_w

                poly = [
                    (p1[0] + nx, p1[1] + ny),
                    (p2[0] + nx, p2[1] + ny),
                    (p2[0] - nx, p2[1] - ny),
                    (p1[0] - nx, p1[1] - ny),
                ]
                pygame.draw.polygon(self._screen, COLOR_ROAD, poly)

            # Road center line (dashed)
            for i in range(len(screen_pts) - 1):
                pygame.draw.line(
                    self._screen, COLOR_ROAD_LINE,
                    screen_pts[i], screen_pts[i + 1], 1
                )

        # Draw goal
        if goal is not None:
            gx, gy = self._world_to_screen(goal[0], goal[1], cam_x, cam_y)
            pygame.draw.circle(self._screen, COLOR_GOAL, (gx, gy), 8)
            pygame.draw.circle(self._screen, (255, 255, 255), (gx, gy), 8, 2)

        # Draw entities
        for entity in self.world.entities.values():
            self._draw_entity(entity, cam_x, cam_y)

        # HUD
        self._draw_hud(ego)

        if self.mode == "human":
            pygame.display.flip()
            self._clock.tick(fps)

            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            return None
        else:
            return np.transpose(
                pygame.surfarray.array3d(self._screen), axes=(1, 0, 2)
            )

    def _draw_entity(
        self, entity: Entity, cam_x: float, cam_y: float
    ):
        import pygame

        corners = entity.get_corners()
        screen_corners = [
            self._world_to_screen(c[0], c[1], cam_x, cam_y) for c in corners
        ]

        if entity.collided:
            color = COLOR_COLLISION
        elif entity.is_ego:
            color = COLOR_EGO
        else:
            color = ENTITY_COLORS.get(entity.entity_type, COLOR_VEHICLE)

        pygame.draw.polygon(self._screen, color, screen_corners)
        pygame.draw.polygon(self._screen, (255, 255, 255), screen_corners, 1)

        # Heading indicator
        cx, cy = self._world_to_screen(entity.x, entity.y, cam_x, cam_y)
        arrow_len = max(entity.length, 2.0) * self.ppm * 0.6
        ax = int(cx + arrow_len * math.cos(-entity.heading))
        ay = int(cy + arrow_len * math.sin(-entity.heading))
        pygame.draw.line(self._screen, (255, 255, 255), (cx, cy), (ax, ay), 2)

        # Label
        label = self._font.render(entity.id, True, COLOR_TEXT)
        self._screen.blit(label, (cx - label.get_width() // 2, cy - 20))

    def _draw_hud(self, ego: Optional[Entity]):
        y = 10
        lines = [
            f"Time: {self.world.time:.1f}s  Step: {self.world.timestep}",
        ]
        if ego:
            lines.append(
                f"Ego: ({ego.x:.1f}, {ego.y:.1f})  "
                f"Speed: {ego.speed:.1f} m/s  "
                f"Heading: {math.degrees(ego.heading):.0f}°"
            )
        if self.world.collisions:
            lines.append(f"COLLISION: {self.world.collisions}")

        for line in lines:
            surf = self._font.render(line, True, COLOR_TEXT)
            self._screen.blit(surf, (10, y))
            y += 18

    def close(self):
        if self._initialized:
            import pygame
            pygame.quit()
            self._initialized = False
