"""
Example demonstrating the Slave card game RL environment.

This shows how to use the PettingZoo environment wrapper for training RL agents.
"""

import numpy as np
from environment.slave_env import SlaveEnv


def play_random_game(env, render=True):
    """
    Play one complete game with random agents.

    Args:
        env: SlaveEnv instance
        render: Whether to render game state

    Returns:
        Dictionary with game statistics
    """
    env.reset(seed=None)

    step_count = 0
    max_steps = 500

    while not all(env.terminations.values()) and step_count < max_steps:
        agent = env.agent_selection

        # Get observation
        obs = env.observe(agent)

        # Get action mask (valid actions)
        action_mask = env.infos[agent]["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]

        # Random agent: choose random valid action
        # Prefer non-pass actions to make progress
        action = np.random.choice(valid_actions)

        # If multiple valid actions, prefer non-pass 80% of the time
        if len(valid_actions) > 1 and np.random.random() < 0.8:
            non_pass_actions = [a for a in valid_actions if a != 0]
            if len(non_pass_actions) > 0:
                action = np.random.choice(non_pass_actions)

        # Take action
        env.step(action)
        step_count += 1

        if render and step_count % 20 == 0:
            print(f"\nStep {step_count}:")
            env.render()

    if render:
        print("\n" + "=" * 60)
        print("GAME OVER")
        print("=" * 60)

        # Show final positions and rewards
        for agent in env.agents:
            player_id = env.agent_name_mapping[agent]
            position = env.game_state.get_position(player_id)
            reward = env._cumulative_rewards[agent]

            print(f"{agent}: {position.value} (Reward: {reward:.2f})")

    # Return statistics
    stats = {
        "steps": step_count,
        "rewards": {agent: env._cumulative_rewards[agent] for agent in env.agents},
        "positions": {
            agent: env.game_state.get_position(env.agent_name_mapping[agent]).value
            for agent in env.agents
        }
    }

    return stats


def run_multiple_games(num_games=10):
    """
    Run multiple games and collect statistics.

    Args:
        num_games: Number of games to run

    Returns:
        Dictionary with aggregate statistics
    """
    env = SlaveEnv()

    all_rewards = {f"player_{i}": [] for i in range(4)}
    all_positions = {pos: 0 for pos in ["King", "Queen", "Commoner", "Slave"]}
    all_steps = []

    print(f"\nRunning {num_games} games with random agents...")

    for game_num in range(num_games):
        stats = play_random_game(env, render=False)

        all_steps.append(stats["steps"])

        for agent, reward in stats["rewards"].items():
            all_rewards[agent].append(reward)

        for agent, position in stats["positions"].items():
            all_positions[position] += 1

        if (game_num + 1) % 10 == 0:
            print(f"  Completed {game_num + 1}/{num_games} games")

    # Calculate statistics
    print("\n" + "=" * 60)
    print("STATISTICS")
    print("=" * 60)

    print(f"\nAverage game length: {np.mean(all_steps):.1f} steps")
    print(f"Min/Max game length: {min(all_steps)}/{max(all_steps)} steps")

    print("\nAverage rewards per player:")
    for agent, rewards in all_rewards.items():
        print(f"  {agent}: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")

    print("\nPosition distribution (total across all players in all games):")
    total = sum(all_positions.values())
    for position, count in sorted(all_positions.items()):
        percentage = (count / total) * 100
        print(f"  {position}: {count} ({percentage:.1f}%)")

    print("=" * 60)

    return {
        "avg_steps": np.mean(all_steps),
        "avg_rewards": {agent: np.mean(rewards) for agent, rewards in all_rewards.items()},
        "position_distribution": all_positions
    }


def demonstrate_observation_space():
    """
    Demonstrate the observation space structure.
    """
    env = SlaveEnv()
    env.reset(seed=42)

    agent = env.agent_selection
    obs = env.observe(agent)

    print("\n" + "=" * 60)
    print("OBSERVATION SPACE STRUCTURE")
    print("=" * 60)

    print(f"\nObservation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")

    # Break down observation components
    idx = 0
    print("\nObservation components:")

    print(f"  [0:52]   Own hand (52 binary) - {np.sum(obs[0:52]):.0f} cards")
    idx += 52

    print(f"  [52:104] Played cards (52 binary) - {np.sum(obs[52:104]):.0f} cards played")
    idx += 52

    print(f"  [104:109] Last play type (5 one-hot) - type {np.argmax(obs[104:109])}")
    idx += 5

    print(f"  [109:122] Last play ranks (13 features) - {np.sum(obs[109:122]):.0f} ranks")
    idx += 13

    print(f"  [122:126] Last play suit (4 one-hot) - suit {np.argmax(obs[122:126])}")
    idx += 4

    print(f"  [126:142] Positions (4x4 one-hot)")
    idx += 16

    print(f"  [142:146] Cards remaining (4 normalized)")
    for i, cards in enumerate(obs[142:146]):
        print(f"    Player {i}: {cards * 13:.0f} cards")
    idx += 4

    print(f"  [146:147] Round number: {obs[146] * 10:.0f}")
    idx += 1

    print(f"  [147:151] Current player (4 binary) - player {np.argmax(obs[147:151])}")
    idx += 4

    print(f"  [151:155] Own position (4 one-hot) - position {np.argmax(obs[151:155])}")

    print("=" * 60)


def demonstrate_action_space():
    """
    Demonstrate the action space structure.
    """
    env = SlaveEnv()
    env.reset(seed=42)

    agent = env.agent_selection
    action_mask = env.infos[agent]["action_mask"]

    print("\n" + "=" * 60)
    print("ACTION SPACE STRUCTURE")
    print("=" * 60)

    print(f"\nAction space size: {len(action_mask)}")
    print(f"Valid actions: {np.sum(action_mask):.0f}")

    print("\nAction categories:")
    print(f"  [0]       Pass")
    print(f"  [1-52]    Singles (52 cards)")
    print(f"  [53-130]  Pairs (78 combinations)")
    print(f"  [131-180] Straights (~50 combinations)")
    print(f"  [181-193] Four-of-a-kinds (13 ranks)")

    # Show valid actions by category
    pass_valid = action_mask[0] == 1
    singles_valid = np.sum(action_mask[1:53])
    pairs_valid = np.sum(action_mask[53:131])
    straights_valid = np.sum(action_mask[131:181])
    fours_valid = np.sum(action_mask[181:194])

    print("\nValid actions in this state:")
    print(f"  Pass: {'Yes' if pass_valid else 'No'}")
    print(f"  Singles: {singles_valid:.0f}")
    print(f"  Pairs: {pairs_valid:.0f}")
    print(f"  Straights: {straights_valid:.0f}")
    print(f"  Four-of-a-kinds: {fours_valid:.0f}")

    print("=" * 60)


def main():
    """Main demonstration function."""
    print("=" * 60)
    print("Slave Card Game RL Environment Demonstration")
    print("=" * 60)

    # 1. Show observation space structure
    demonstrate_observation_space()

    # 2. Show action space structure
    demonstrate_action_space()

    # 3. Play one game with rendering
    print("\n" + "=" * 60)
    print("PLAYING ONE GAME (with rendering every 20 steps)")
    print("=" * 60)

    env = SlaveEnv(render_mode="human")
    play_random_game(env, render=True)

    # 4. Run multiple games and show statistics
    run_multiple_games(num_games=100)

    print("\n" + "=" * 60)
    print("Environment demonstration complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("  1. Implement baseline agents (random, greedy)")
    print("  2. Train RL agents using PPO")
    print("  3. Evaluate agent performance")
    print("=" * 60)


if __name__ == "__main__":
    main()
