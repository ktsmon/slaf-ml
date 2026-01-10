"""
Tournament script to test GreedyAgent vs RandomAgent.

Runs 100 games and collects statistics to verify GreedyAgent achieves
the expected 70%+ win rate against RandomAgent.
"""

import numpy as np
from environment.slave_env import SlaveEnv
from agents.random_agent import RandomAgent
from agents.greedy_agent import GreedyAgent
from game.game_state import Position


def run_tournament(num_games: int = 100):
    """
    Run tournament between GreedyAgent and RandomAgent.

    Args:
        num_games: Number of games to play
    """
    # Track statistics
    stats = {
        'greedy_positions': [],  # Final positions for greedy agents
        'random_positions': [],  # Final positions for random agents
        'greedy_wins': 0,  # Number of times greedy finishes 1st
        'random_wins': 0,  # Number of times random finishes 1st
        'game_lengths': [],  # Steps per game
    }

    position_values = {
        Position.KING: 1,
        Position.QUEEN: 2,
        Position.COMMONER: 3,
        Position.SLAVE: 4
    }

    print(f"Running tournament: {num_games} games")
    print(f"Players 0, 2: GreedyAgent")
    print(f"Players 1, 3: RandomAgent")
    print("=" * 60)

    for game_num in range(num_games):
        # Create environment
        env = SlaveEnv()
        env.reset(seed=game_num)

        # Create agents (2 greedy, 2 random)
        agents = [
            GreedyAgent(0, "Greedy_0"),
            RandomAgent(1, "Random_1", seed=game_num),
            GreedyAgent(2, "Greedy_2"),
            RandomAgent(3, "Random_3", seed=game_num + 1000)
        ]

        # Play game
        steps = 0
        max_steps = 500

        while not all(env.terminations.values()) and steps < max_steps:
            # Get current agent
            agent_name = env.agent_selection
            player_id = env.agent_name_mapping[agent_name]
            agent = agents[player_id]

            # Get observation and action mask
            observation = env.observe(agent_name)
            action_mask = env.infos[agent_name].get("action_mask")

            # Select action
            action = agent.select_action(observation, action_mask)

            # Execute action
            env.step(action)
            steps += 1

        # Record results
        stats['game_lengths'].append(steps)

        # Get final positions
        for player_id in range(4):
            position = env.game_state.get_position(player_id)
            position_val = position_values[position]

            if player_id in [0, 2]:  # Greedy agents
                stats['greedy_positions'].append(position_val)
                if position == Position.KING:
                    stats['greedy_wins'] += 1
            else:  # Random agents
                stats['random_positions'].append(position_val)
                if position == Position.KING:
                    stats['random_wins'] += 1

        # Print progress every 10 games
        if (game_num + 1) % 10 == 0:
            print(f"Completed {game_num + 1}/{num_games} games")

    # Calculate statistics
    print("\n" + "=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)

    greedy_avg_position = np.mean(stats['greedy_positions'])
    random_avg_position = np.mean(stats['random_positions'])
    avg_game_length = np.mean(stats['game_lengths'])

    # Win rates (percentage of games where agent finished 1st)
    greedy_win_rate = (stats['greedy_wins'] / (num_games * 2)) * 100  # 2 greedy agents per game
    random_win_rate = (stats['random_wins'] / (num_games * 2)) * 100  # 2 random agents per game

    # Position distribution
    greedy_position_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    random_position_counts = {1: 0, 2: 0, 3: 0, 4: 0}

    for pos in stats['greedy_positions']:
        greedy_position_counts[pos] += 1
    for pos in stats['random_positions']:
        random_position_counts[pos] += 1

    print(f"\nGames played: {num_games}")
    print(f"Average game length: {avg_game_length:.1f} steps")

    print(f"\n--- GreedyAgent Performance ---")
    print(f"Average position: {greedy_avg_position:.2f}")
    print(f"Win rate (1st place): {greedy_win_rate:.1f}%")
    print(f"Position distribution:")
    for pos in [1, 2, 3, 4]:
        count = greedy_position_counts[pos]
        pct = (count / len(stats['greedy_positions'])) * 100
        print(f"  Position {pos}: {count} ({pct:.1f}%)")

    print(f"\n--- RandomAgent Performance ---")
    print(f"Average position: {random_avg_position:.2f}")
    print(f"Win rate (1st place): {random_win_rate:.1f}%")
    print(f"Position distribution:")
    for pos in [1, 2, 3, 4]:
        count = random_position_counts[pos]
        pct = (count / len(stats['random_positions'])) * 100
        print(f"  Position {pos}: {count} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)

    # Calculate game-level win rate (which agent won each game)
    game_win_rate = (stats['greedy_wins'] / num_games) * 100

    # Check if GreedyAgent meets the 70% win rate target
    print(f"\nGames won by GreedyAgent: {stats['greedy_wins']}/{num_games} ({game_win_rate:.1f}%)")
    print(f"Games won by RandomAgent: {stats['random_wins']}/{num_games} ({100 - game_win_rate:.1f}%)")

    if game_win_rate >= 70.0:
        print(f"\n[SUCCESS] GreedyAgent achieves {game_win_rate:.1f}% game win rate (target: 70%+)")
    else:
        print(f"\n[BELOW TARGET] GreedyAgent achieves {game_win_rate:.1f}% game win rate (target: 70%+)")

    if greedy_avg_position < random_avg_position:
        print(f"[SUCCESS] GreedyAgent has better average position ({greedy_avg_position:.2f} vs {random_avg_position:.2f})")
    else:
        print(f"[WARNING] RandomAgent has better average position ({random_avg_position:.2f} vs {greedy_avg_position:.2f})")

    print("=" * 60)

    return stats


if __name__ == "__main__":
    run_tournament(num_games=100)
