#!/usr/bin/env python3
"""Demo script: runs a scenario with a simple policy and prints state."""

import sys
import numpy as np
from scenarios.loader import ScenarioLoader
from agent.env import DrivingEnv


def main():
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "straight_road"
    render = "--render" in sys.argv

    print(f"Loading scenario: {scenario_name}")
    scenario = ScenarioLoader.load_by_name(scenario_name)
    world = ScenarioLoader.build_world(scenario)

    env = DrivingEnv(
        world=world,
        goal=scenario.goal,
        max_steps=scenario.max_steps,
        render_mode="human" if render else None,
    )
    env._save_initial_states()

    obs, info = env.reset()
    total_reward = 0.0

    print(f"Goal: {scenario.goal}")
    print(f"Max steps: {scenario.max_steps}")
    print("-" * 60)

    for step in range(scenario.max_steps):
        # Simple policy: accelerate forward, no steering
        action = np.array([0.0, 0.5], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            ego = world.get_ego()
            dist = info.get("distance_to_goal", "N/A")
            print(
                f"Step {step:4d} | "
                f"pos=({ego.x:7.1f}, {ego.y:7.1f}) | "
                f"speed={ego.speed:5.1f} m/s | "
                f"dist_to_goal={dist} | "
                f"reward={reward:+.2f}"
            )

        if render:
            env.render()

        if terminated or truncated:
            reason = "COLLISION" if world.get_ego().collided else (
                "GOAL REACHED" if terminated else "TIME LIMIT"
            )
            print(f"\nEpisode ended: {reason}")
            break

    print(f"Total reward: {total_reward:.2f}")
    print(f"Final time: {world.time:.1f}s")
    env.close()


if __name__ == "__main__":
    main()
