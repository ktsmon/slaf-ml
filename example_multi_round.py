"""
Example demonstrating multi-round game play.

This shows how to use the SlaveEnv environment with multiple rounds,
where card exchanges between rounds affect player positions and strategies.
"""

import numpy as np
from environment.slave_env import SlaveEnv
from game.game_state import Position


def play_game_greedy(env, max_steps=1000):
    """
    Play a game using greedy action selection (prefer non-pass actions).

    Args:
        env: SlaveEnv environment
        max_steps: Maximum steps before timeout

    Returns:
        Number of steps taken
    """
    step_count = 0

    while not all(env.terminations.values()) and step_count < max_steps:
        agent = env.agent_selection
        obs = env.observe(agent)
        action_mask = env.infos[agent]["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            break

        # Prefer non-pass actions
        action = valid_actions[0]
        for act in valid_actions:
            if act != 0:  # Not pass
                action = act
                break

        env.step(action)
        step_count += 1

    return step_count


def print_game_result(env):
    """Print the final game result with positions."""
    print("\nFinal Positions:")
    for agent in env.agents:
        player_id = env.agent_name_mapping[agent]
        position = env.game_state.get_position(player_id)
        reward = env._cumulative_rewards[agent]
        print(f"  {agent}: {position.value:10} | Reward: {reward:7.2f}")


def main():
    print("=" * 70)
    print("Multi-Round Slave Card Game Example")
    print("=" * 70)

    # Example 1: Single round (default behavior)
    print("\n" + "-" * 70)
    print("Example 1: Single Round Game (Default)")
    print("-" * 70)

    env = SlaveEnv(num_rounds=1)
    env.reset(seed=42)

    print(f"Configuration: num_rounds={env.num_rounds}")
    steps = play_game_greedy(env)

    print(f"Game completed in {steps} steps")
    print(f"Rounds completed: {env.rounds_completed}")
    print_game_result(env)

    # Example 2: Two rounds with position changes
    print("\n" + "-" * 70)
    print("Example 2: Two-Round Game (Positions Change Between Rounds)")
    print("-" * 70)

    env = SlaveEnv(num_rounds=2)
    env.reset(seed=42)

    print(f"Configuration: num_rounds={env.num_rounds}")
    print(f"Starting Round 1...")

    # Play until first round ends
    step_count = 0
    max_steps = 500
    while step_count < max_steps:
        agent = env.agent_selection
        obs = env.observe(agent)
        action_mask = env.infos[agent]["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            break

        action = valid_actions[0]
        for act in valid_actions:
            if act != 0:
                action = act
                break

        env.step(action)
        step_count += 1

        # Check if round changed
        if env.game_state.round_number == 1 and env.rounds_completed == 1:
            print(f"\nRound 1 Complete! (after {step_count} steps)")
            print("Positions after Round 1:")
            for agent in env.agents:
                player_id = env.agent_name_mapping[agent]
                position = env.game_state.get_position(player_id)
                print(f"  {agent}: {position.value}")

            print(f"\nStarting Round 2 (agents will exchange cards based on positions)...")
            break

    # Continue playing until game ends
    while not all(env.terminations.values()) and step_count < max_steps + 500:
        agent = env.agent_selection
        obs = env.observe(agent)
        action_mask = env.infos[agent]["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            break

        action = valid_actions[0]
        for act in valid_actions:
            if act != 0:
                action = act
                break

        env.step(action)
        step_count += 1

    print(f"\nRound 2 Complete! Total steps: {step_count}")
    print(f"Rounds completed: {env.rounds_completed}/{env.num_rounds}")
    print_game_result(env)

    # Example 3: Three rounds
    print("\n" + "-" * 70)
    print("Example 3: Three-Round Game")
    print("-" * 70)

    env = SlaveEnv(num_rounds=3)
    env.reset(seed=99)

    print(f"Configuration: num_rounds={env.num_rounds}")
    steps = play_game_greedy(env, max_steps=2000)

    print(f"Game completed in {steps} steps")
    print(f"Rounds completed: {env.rounds_completed}/{env.num_rounds}")
    print_game_result(env)

    print("\n" + "=" * 70)
    print("Multi-Round Examples Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
