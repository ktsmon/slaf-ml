# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a reinforcement learning system for training AI agents to play the Thai "Slave" card game (also known as "President" or "Scum"). Four agents compete through self-play training using the PPO algorithm. The project follows a phased implementation approach.

**Current Status:** Phase 1 complete (core game engine). The game engine includes:
- Complete card system with rank and suit representation
- Deck management with shuffling and dealing
- Full game rules implementation (singles, pairs, straights, four-of-a-kind)
- Game state management with strategic card exchange mechanics
- PettingZoo environment wrapper
- State observation encoding for agents

**Next Phases:** Baseline agents (random, greedy), RL training with PPO, self-play training loop, curriculum learning.

## Common Commands

### Testing
```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_card.py -v
pytest tests/test_rules.py -v
pytest tests/test_game_state.py -v
pytest tests/test_environment.py -v

# Run tests with coverage
pytest tests/ --cov=game --cov=environment -v

# Run specific test by name
pytest tests/test_rules.py::test_pairs_beat_singles -v
```

### Running Examples
```bash
# Test strategic exchange mechanics
python example_strategic_exchange.py

# Test environment integration
python example_environment.py

# Test multi-round game play
python example_multi_round.py
```

### Running the Game
```bash
# Watch example games (once main.py exists)
python main.py --mode watch --games 5

# Run tests with agents
python main.py --mode test --agents random random random random --games 100
```

## Code Architecture

### Core Game Logic (`game/`)

**Card System (`game/card.py`)**
- `Card` class: Represents a single card with rank and suit
- `Rank` enum: Values 3-2 (where 2 is highest), stored as IntEnum values 3-15
- `Suit` enum: Spades > Hearts > Diamonds > Clubs (values 4, 3, 2, 1)
- Card comparison uses rank first, then suit as tiebreaker
- `to_int()` and `from_int()`: Bidirectional conversion to 0-51 encoding (rank_index * 4 + suit_index)
- `create_deck()`: Creates standard 52-card deck

**Deck Management (`game/deck.py`)**
- `Deck` class: Manages shuffling and dealing
- `shuffle(seed)`: Supports deterministic shuffling with seed
- `deal(num_players)`: Returns list of hands (13 cards each for 4 players)

**Game Rules (`game/rules.py`)**
- `PlayType` enum: PASS, SINGLE, PAIR, STRAIGHT, FOUR_OF_KIND
- `Play` class: Represents a played combination with type detection
- Key functions:
  - `validate_play(cards, last_play)`: Checks if play is legal
  - `compare_plays(play1, play2)`: Returns -1 (play1 loses), 0 (tie), 1 (play1 wins)
  - `get_valid_actions(hand, last_play)`: Returns all legal moves
  - `determine_winner(plays)`: Identifies who won the trick
- Handles complex rules: pairs beat singles, straights lose to pairs, four-of-a-kind beats pairs, suit tiebreakers for same rank

**Game State Management (`game/game_state.py`)**
- `GameState` class: Manages complete game flow
- `reset()`: Initializes new game, determines starting position
- `step(player_id, action)`: Processes a move, returns observation, reward, done, info
- `exchange_cards()`: Handles between-round exchanges:
  - Slave gives 2 best cards to King
  - Commoner gives 1 best card to Queen
  - King/Queen choose which cards to give (strategic choice)
- `determine_positions()`: Sets King, Queen, Commoner, Slave ranks
- `is_game_over()`: Game ends when all but one player have finished
- Implements special rule: If non-King finishes first, previous King becomes Slave
- Tracking: hands, positions, last play, current player, round number, trick history

### Environment for RL (`environment/`)

**Observation Encoding (`environment/observations.py`)**
- `ObservationEncoder` class: Converts game state to neural network input
- Observation vector (~156 features):
  - Own hand (52 binary): Which cards you hold
  - Played cards (52 binary): Cards out of play
  - Last play encoding (22 features): Play type (5 one-hot), card ranks (13), suit (4)
  - Game context (30 features): Positions, cards remaining, round, current player indicator
- `encode_observation(game_state, player_id)`: Returns numpy array
- `get_action_mask(hand, last_play)`: Returns binary mask of valid actions (194 total)
- Action space (194 actions):
  - 52 single cards
  - 78 possible pairs
  - ~50 common straights
  - 13 four-of-a-kind
  - 1 pass action

**PettingZoo Environment (`environment/slave_env.py`)**
- `SlaveEnv` class: AEC (Agent Environment Cycle) wrapper
- Multi-agent environment with 4 players
- Constructor parameter: `num_rounds` (default: 1) - determines how many rounds to play before episode ends
- Methods: `reset()`, `step()`, `observe()`, `reward()`, `termination()`, `truncation()`
- Action space: Discrete(194)
- Observation space: Box(0, 1, shape=(156,), dtype=float32)
- Reward structure:
  - King (1st): +10
  - Queen (2nd): +5
  - Commoner (3rd): -5
  - Slave (4th): -10
  - Per-card penalty: -0.01 for each card remaining at game end
  - Invalid move penalty: -1
- Multi-round play:
  - Rewards accumulate across rounds
  - Cards are exchanged between rounds based on positions (Slave→King: 2 cards, Commoner→Queen: 1 card)
  - Positions are reassigned after each round
  - Episode only terminates after all N rounds complete
  - Example: `env = SlaveEnv(num_rounds=3)` plays 3 full rounds before ending
- Integrates with `GameState` and `ObservationEncoder`

### Testing (`tests/`)

- `test_card.py`: Card comparison, encoding/decoding, ranking
- `test_deck.py`: Shuffling, dealing, deck integrity
- `test_rules.py`: Play validation, play comparison, valid action generation
- `test_game_state.py`: Game flow, exchanges, position determination, King demotion rule
- `test_environment.py`: Environment reset/step, observation encoding, action masking
- `test_strategic_exchange.py`: Strategic card exchange (King/Queen decision-making)

All tests pass. Use `pytest tests/ -v` to run the full suite.

## Key Implementation Details

### Multi-Round Game Play

The `SlaveEnv` supports multi-round play where agents compete across multiple complete games in a single episode:

```python
# Single round (default)
env = SlaveEnv(num_rounds=1)

# Multiple rounds
env = SlaveEnv(num_rounds=3)
env.reset(seed=42)

# Play the game - will run 3 complete rounds before terminating
# Rewards accumulate across all rounds
# Positions change after each round, affecting card exchanges
```

**How it works:**
1. First round plays to completion (all but one player finish)
2. Positions are determined: King, Queen, Commoner, Slave
3. Instead of terminating the episode, a new round starts
4. Cards are dealt, exchanges occur (Slave gives best 2 to King, etc.)
5. Round counter increments, but episode continues
6. After N rounds complete, all agents are marked terminated and episode ends

**Key Points:**
- Rewards are given at the end of each round and accumulate
- Position changes between rounds affect agent strategies
- Card exchanges persist (Slave must give good cards again if they finish first again)
- Use `env.rounds_completed` to track progress
- Use `env.game_state.round_number` to know the current round being played

### Card Ranking System
```
Rank: 2 (15) > A (14) > K (13) > Q (12) > J (11) > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3 (3)
Suit: Spades (4) > Hearts (3) > Diamonds (2) > Clubs (1)
```

### Card Exchange Strategy
During card exchange phase, King and Slave must submit 2 cards each:
- Slave **must** give 2 best cards to King
- King **chooses** which 2 cards to give Slave (strategic decision-making point)
- Similarly for Queen/Commoner with 1 card exchange

### Game Loop Flow
1. Start new round
2. Exchange cards (if not first round)
3. Determine starting player
4. Play tricks until only one player remains
5. Record finish position and calculate rewards
6. Repeat until game ends (or episode limit)

### Action Encoding
Actions are integer IDs mapping to card combinations:
- 0-51: Single cards (by card ID)
- 52-129: Pairs
- 130-179: Straights (common patterns)
- 180-192: Four-of-a-kind
- 193: Pass

The `ObservationEncoder.get_action_mask()` ensures only valid moves are selectable.

## Dependencies

Key packages (see requirements.txt):
- **pettingzoo**: Multi-agent environment standard
- **stable-baselines3**: PPO implementation
- **torch**: Neural network backend
- **numpy**: Numerical computing
- **pytest**: Testing framework
- **tensorboard**: Training visualization
- **matplotlib, seaborn**: Plotting

## Next Implementation Steps

### Phase 2: Baseline Agents (agents/)
- `base_agent.py`: Abstract interface
- `random_agent.py`: Random valid move selection
- `greedy_agent.py`: Rule-based heuristics (play low cards early, conserve high cards)
- Test: Greedy beats random 70%+ of games

### Phase 3: RL Training (training/)
- `self_play.py`: Training loop with PPO
- `evaluator.py`: Performance metrics (win rate, average position, head-to-head)
- `curriculum.py`: Progressive difficulty (simplified rules → full game)

### Phase 4: Entry Point & Scripts
- `main.py`: CLI with modes: train, eval, tournament, watch, play
- Training pipeline setup with checkpointing and TensorBoard logging

## Testing Checkpoints

| Checkpoint | Command | Expected Result |
|-----------|---------|-----------------|
| Game Engine | `pytest tests/test_card.py tests/test_rules.py -v` | All tests pass |
| Environment | `pytest tests/test_environment.py -v` | Environment valid, action masking works |
| Strategic Exchange | `python example_strategic_exchange.py` | Exchanges work correctly |
| Full Game | `python example_environment.py` | Complete games play without errors |

## Development Guidelines

- **Type hints**: Use throughout for clarity (see existing code for patterns)
- **Docstrings**: Include for complex game logic and public methods
- **Testing**: Implement tests alongside features
- **State management**: Keep `GameState` as single source of truth
- **Reward signals**: Ensure rewards are meaningful for learning
- **Determinism**: Use seeds for reproducible gameplay during debugging

## Important Files to Read First

When starting Phase 2+:
1. `plan.md`: Detailed implementation roadmap
2. `game/rules.py`: Play validation logic (needed to understand valid actions)
3. `game/game_state.py`: State transitions and reward calculation
4. `environment/observations.py`: State-to-vector encoding
5. `tests/test_strategic_exchange.py`: Example of agent decision-making

## Common Pitfalls

- **Action masking**: Always use `get_action_mask()` to prevent invalid moves in training
- **Card exchange timing**: Only happens between rounds, not during tricks
- **Reward structure**: Ensure agents are rewarded for finishing early
- **Round tracking**: First round starts with 3♣ holder; subsequent rounds vary based on finish order
- **Suit comparison**: Only used as tiebreaker for same rank (e.g., A♠ > A♥)
