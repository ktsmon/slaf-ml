# Slave Card Game RL

A reinforcement learning system where 4 AI agents learn to play the Thai "Slave" card game through self-play training using Deep Reinforcement Learning (PPO algorithm).

## Overview

This project implements a complete RL training pipeline for the Thai card game "Slave" (also known as "President" or "Scum" in other regions). Four AI agents compete against each other, learning optimal strategies through thousands of self-play games.

### Game Rules Summary

- **Players:** 4 players, each starting with 13 cards
- **Objective:** Get rid of all your cards first to become King
- **Card Ranking:** 2 (highest) > A > K > Q > ... > 3 (lowest)
- **Suit Ranking:** ♠ > ♥ > ♣ > ♦
- **Play Types:** Singles, Pairs, Straights, Four-of-a-kind
- **Positions:** Players finish as King, Queen, Commoner, or Slave
- **Card Exchange:** Between rounds, Slave gives best cards to King, Commoner exchanges with Queen
- **Special Rule:** If non-King finishes first, the previous King becomes Slave

## Features

- **Complete Game Engine:** Full implementation of Thai Slave card game rules
- **PettingZoo Environment:** Standard multi-agent RL interface
- **Multiple Agent Types:** Random, Greedy (rule-based), and PPO-based RL agents
- **Self-Play Training:** Agents improve by playing against themselves
- **Curriculum Learning:** Progressive difficulty from simplified to full game rules
- **Evaluation Suite:** Win rates, position tracking, head-to-head tournaments
- **Visualization:** Training curves, strategy heatmaps, game replays

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd slave-card-rl

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- Stable-Baselines3
- PettingZoo
- NumPy
- Matplotlib
- TensorBoard

## Project Structure

```
slave-card-rl/
├── README.md
├── requirements.txt
├── config.py                    # Game & training configurations
├── game/
│   ├── card.py                  # Card representation & comparisons
│   ├── deck.py                  # Deck shuffling & dealing
│   ├── rules.py                 # Game rules engine
│   └── game_state.py            # Game state management
├── environment/
│   ├── slave_env.py             # PettingZoo environment wrapper
│   └── observations.py          # State representation for agents
├── agents/
│   ├── base_agent.py            # Abstract agent interface
│   ├── random_agent.py          # Random baseline
│   ├── greedy_agent.py          # Rule-based baseline
│   └── rl_agent.py              # PPO agent wrapper
├── training/
│   ├── self_play.py             # Self-play training loop
│   ├── evaluator.py             # Agent evaluation & metrics
│   └── curriculum.py            # Progressive difficulty training
├── utils/
│   ├── logger.py                # Training logs & metrics
│   └── visualization.py         # Plot training progress
├── tests/
│   ├── test_card.py
│   ├── test_rules.py
│   └── test_environment.py
└── main.py                      # Entry point
```

## Quick Start

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rules.py -v
```

### Train an Agent

```bash
# Train from scratch (100k timesteps)
python main.py --mode train --timesteps 100000

# Continue training from checkpoint
python main.py --mode train --timesteps 100000 --load models/checkpoint_50000.zip

# Monitor training with TensorBoard
tensorboard --logdir logs/
```

### Evaluate Agent Performance

```bash
# Evaluate trained agent vs baselines
python main.py --mode eval --model models/best_agent.zip --games 100

# Run tournament between multiple agents
python main.py --mode tournament --agents 4

# Watch agent play (human-readable output)
python main.py --mode watch --model models/best_agent.zip
```

### Play Against AI

```bash
# Play interactive game against trained agents
python main.py --mode play --model models/best_agent.zip
```

## Training Pipeline

### Phase 1: Foundation (Weeks 1-2)
- Core game engine (cards, deck, rules, game state)
- PettingZoo environment wrapper
- State observation encoding

### Phase 2: Baseline Agents (Week 3)
- Random agent (baseline)
- Greedy agent (rule-based heuristics)
- Validation: Greedy beats Random 70%+ of games

### Phase 3: RL Training (Week 4)
- PPO agent implementation with Stable-Baselines3
- Initial training vs random opponents
- Action masking for valid moves only

### Phase 4: Self-Play (Week 5)
- Self-play training loop
- Evaluation metrics and checkpoints
- Expected: Beat greedy agent after 50k games

### Phase 5: Advanced Training (Week 6)
- Curriculum learning (progressive difficulty)
- Large-scale training (100k+ games)
- Strategy analysis and visualization

## Configuration

Key parameters in [config.py](config.py):

```python
# Game Configuration
GAME_CONFIG = {
    'num_players': 4,
    'cards_per_player': 13,
    'exchange_cards_king_slave': 2,
    'exchange_cards_queen_commoner': 1,
}

# Training Configuration
TRAINING_CONFIG = {
    'total_timesteps': 1_000_000,
    'learning_rate': 3e-4,
    'batch_size': 64,
    'eval_frequency': 10_000,
}

# Reward Structure
REWARD_CONFIG = {
    'king': 10.0,           # 1st place
    'queen': 5.0,           # 2nd place
    'commoner': -5.0,       # 3rd place
    'slave': -10.0,         # 4th place
}
```

## Agent Architecture

### PPO Neural Network

```
Input: Observation Vector (156 features)
  - Own hand encoding (52 binary)
  - Played cards (52 binary)
  - Last play encoding (22 features)
  - Game context (30 features: positions, cards remaining, round)

Hidden Layers:
  - Dense(256) + ReLU
  - Dense(256) + ReLU
  - Dense(128) + ReLU

Output Heads:
  - Policy Head: Action probabilities (194 actions)
  - Value Head: State value estimation
```

### Action Space

- All single cards: 52 actions
- All pairs: 78 actions
- Common straights: ~50 actions
- Four-of-a-kind: 13 actions
- Pass: 1 action
- **Total: 194 discrete actions**

## Expected Training Results

| Training Games | Performance Milestones |
|----------------|------------------------|
| 10,000 | Legal moves 99%+, beats random 80%+ |
| 50,000 | Beats greedy agent 60%+, position awareness |
| 100,000+ | Near-optimal play, advanced strategies |

### Learning Progression

1. **Early (0-10k games):** Learn valid moves, basic card value
2. **Mid (10k-50k games):** Position importance, card exchange strategy
3. **Advanced (50k+ games):** Timing, bluffing, exploiting opponents

## Evaluation Metrics

- **Win Rate:** Percentage finishing as King
- **Average Position:** 1.0 (King) to 4.0 (Slave)
- **Game Length:** Average turns to complete
- **Head-to-Head Matrix:** Performance vs different opponents
- **Strategy Diversity:** Action entropy (exploration vs exploitation)

## Testing

### Unit Tests
- Card comparison logic
- Play validation for all combinations
- Game state transitions
- Card exchange mechanics

### Integration Tests
- 1000-game simulation (stability test)
- PettingZoo API compliance
- Environment-agent interface
- Training loop execution

### Checkpoints

```bash
# Checkpoint 1: Game engine works
pytest tests/test_card.py tests/test_rules.py -v

# Checkpoint 2: Environment valid
python -c "from pettingzoo.test import api_test; from environment.slave_env import SlaveEnv; api_test(SlaveEnv())"

# Checkpoint 3: Agents functional
python main.py --mode test --agents random random random random --games 100

# Checkpoint 4: RL learning
python main.py --mode train --timesteps 10000
tensorboard --logdir logs/

# Checkpoint 5: Self-play working
python main.py --mode selfplay --iterations 100
```

## Development Guidelines

- **Testing First:** Implement unit tests alongside each component
- **Type Hints:** Use Python type hints throughout
- **Docstrings:** Detailed documentation for complex game logic
- **Logging:** Extensive logging for debugging training issues
- **Commits:** Commit after each working component

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-agent-type`)
3. Write tests for new functionality
4. Implement changes with type hints and docstrings
5. Run test suite (`pytest tests/ -v`)
6. Submit pull request

## Troubleshooting

### Training not improving
- Check action masking is working (no invalid actions)
- Verify reward signal is correct
- Try lower learning rate (1e-4)
- Increase entropy coefficient for more exploration

### Out of memory
- Reduce batch_size in config
- Decrease network size (fewer neurons)
- Use smaller n_steps (fewer steps before update)

### Game rules issues
- Check [plan.md](plan.md) for detailed rule specifications
- Run unit tests to verify specific rule logic
- Enable debug logging in game_state.py

## License

MIT License - See LICENSE file for details

## Acknowledgments

- **PettingZoo:** Multi-agent RL environment standard
- **Stable-Baselines3:** PPO implementation
- **Thai Card Game Community:** Game rules and strategy insights

## Roadmap

- [ ] Complete Phase 1-3 (Core engine + Baselines)
- [ ] Phase 4: Initial RL training
- [ ] Phase 5: Self-play training
- [ ] Advanced features:
  - [ ] Human vs AI play mode
  - [ ] Web interface for watching games
  - [ ] MCTS agent for comparison
  - [ ] Distributed training support
  - [ ] Mobile app integration

## Contact

For questions, issues, or contributions, please open an issue on GitHub.

---

**Status:** 🚧 In Development - See [plan.md](plan.md) for detailed implementation roadmap
