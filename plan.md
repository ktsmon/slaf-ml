# Slave Card Game RL - Implementation Plan for Claude Code

## Project Overview
Build a reinforcement learning system where 4 AI agents learn to play the Thai "Slave" card game through self-play training. This plan is optimized for implementation with Claude Code.

---

## Project Structure

```
slave-card-rl/
├── README.md
├── requirements.txt
├── config.py                    # Game & training configurations
├── game/
│   ├── __init__.py
│   ├── card.py                  # Card representation & comparisons
│   ├── deck.py                  # Deck shuffling & dealing
│   ├── rules.py                 # Game rules engine
│   └── game_state.py            # Game state management
├── environment/
│   ├── __init__.py
│   ├── slave_env.py             # PettingZoo environment wrapper
│   └── observations.py          # State representation for agents
├── agents/
│   ├── __init__.py
│   ├── base_agent.py            # Abstract agent interface
│   ├── random_agent.py          # Random baseline
│   ├── greedy_agent.py          # Rule-based baseline
│   └── rl_agent.py              # PPO agent wrapper
├── training/
│   ├── __init__.py
│   ├── self_play.py             # Self-play training loop
│   ├── evaluator.py             # Agent evaluation & metrics
│   └── curriculum.py            # Progressive difficulty training
├── utils/
│   ├── __init__.py
│   ├── logger.py                # Training logs & metrics
│   └── visualization.py         # Plot training progress
├── tests/
│   ├── test_card.py
│   ├── test_rules.py
│   └── test_environment.py
└── main.py                      # Entry point
```

---

## Implementation Phases

### Phase 1: Core Game Engine (Priority 1)

#### 1.1 Card System (`game/card.py`)
```python
"""
Implement:
- Card class with rank (3-2) and suit (♣♦♥♠)
- Card comparison logic (rank first, then suit)
- Card encoding/decoding (0-51 mapping)

Key methods:
- __init__(rank, suit)
- __lt__, __gt__, __eq__ for comparisons
- to_int() and from_int(card_id)
- __repr__ for debugging
"""
```

**Test criteria:**
- ✓ 2♠ > A♠ > K♠ > ... > 3♠
- ✓ A♠ > A♥ > A♦ > A♣ (suit ordering)
- ✓ All 52 cards have unique integer IDs

#### 1.2 Deck Management (`game/deck.py`)
```python
"""
Implement:
- Deck class with 52 cards
- Shuffle and deal methods
- Deal to 4 players evenly (13 cards each)

Key methods:
- __init__() - create standard 52-card deck
- shuffle(seed=None)
- deal(num_players=4) -> List[List[Card]]
"""
```

**Test criteria:**
- ✓ Deck contains exactly 52 unique cards
- ✓ Dealing distributes all cards evenly
- ✓ Shuffling with same seed produces same order

#### 1.3 Play Validation (`game/rules.py`)
```python
"""
Implement complete game rules:

1. PlayType enum: SINGLE, PAIR, STRAIGHT, FOUR_OF_KIND, PASS
2. validate_play(cards, last_play) -> bool
3. compare_plays(play1, play2) -> int (-1, 0, 1)
4. get_valid_actions(hand, last_play) -> List[List[Card]]
5. determine_winner(plays) -> int (player_id)

Complex rules to handle:
- Pairs must beat pairs (same type, higher value)
- Straights beat singles but lose to pairs
- Four-of-a-kind beats pairs
- Suit tiebreakers for same rank
- Special case: 2 (highest rank)
"""
```

**Test criteria:**
- ✓ Single 5 cannot beat pair of 3s
- ✓ Straight 3-4-5 beats single K but loses to pair 3s
- ✓ Four 4s beats pair of As
- ✓ Pass is always valid
- ✓ Suit comparison works for same rank

#### 1.4 Game State (`game/game_state.py`)
```python
"""
Implement:
- GameState class tracking full game
- Player hands, positions, current trick
- Round management (first round vs subsequent)
- Card exchange logic for King/Slave, Queen/Commoner

Key methods:
- __init__(num_players=4)
- reset() - new game
- step(player_id, action) -> (observation, reward, done, info)
- exchange_cards() - handle between-round exchanges
- determine_positions() - King/Queen/Commoner/Slave
- is_game_over() -> bool

State tracking:
- hands: Dict[int, List[Card]]
- positions: Dict[int, str] ("King", "Queen", "Commoner", "Slave")
- last_play: Tuple[int, List[Card]]
- current_player: int
- round_number: int
- trick_history: List
"""
```

**Test criteria:**
- ✓ First round starts with 3♣ holder
- ✓ Subsequent rounds: Slave exchanges 2 best cards with King
- ✓ Queen/Commoner exchange 1 card
- ✓ Special rule: Non-King finishing first makes previous King become Slave
- ✓ Game ends when all but one player finish

---

### Phase 2: RL Environment (Priority 2)

#### 2.1 Observation Space (`environment/observations.py`)
```python
"""
Design state representation for neural network:

Observation vector components:
1. Own hand (52 binary features) - which cards you hold
2. Played cards (52 binary) - which cards are out of play
3. Last play encoding:
   - Play type (5 one-hot: single/pair/straight/four/none)
   - Card ranks in play (13 features)
   - Leading suit (4 one-hot)
4. Game context:
   - Current positions (4x4 one-hot: K/Q/C/S for each player)
   - Cards remaining per player (4 integers normalized)
   - Round number (1 integer)
   - Current player indicator (4 binary)
   - Player's own position (4 one-hot)

Total: ~150 features

Key methods:
- encode_observation(game_state, player_id) -> np.array
- decode_action(action_id, hand) -> List[Card]
- get_action_mask(hand, last_play) -> np.array (valid actions only)
"""
```

**Design decisions:**
- Use binary encodings for cards (sparse but clear)
- Normalize continuous values (0-1 range)
- Include action masking to prevent invalid moves

#### 2.2 PettingZoo Environment (`environment/slave_env.py`)
```python
"""
Implement AEC (Agent Environment Cycle) wrapper:

Key methods:
- reset(seed, options) -> observation
- step(action) -> None (updates internal state)
- observe(agent) -> observation
- reward(agent) -> float
- termination(agent) -> bool
- truncation(agent) -> bool

Action space:
- Discrete(N) where N = all possible card combinations
- Simplified: Use integer encoding for common plays
- Advanced: Use multi-discrete for card selection

Reward structure:
- Finishing 1st (King): +10
- Finishing 2nd (Queen): +5
- Finishing 3rd (Commoner): -5
- Finishing 4th (Slave): -10
- Intermediate: -0.01 per card in hand at game end
- Penalty for invalid moves: -1

PettingZoo requirements:
- agents: List[str] = ["player_0", "player_1", "player_2", "player_3"]
- action_spaces: Dict[str, Space]
- observation_spaces: Dict[str, Space]
- rewards: Dict[str, float]
"""
```

**Test criteria:**
- ✓ Environment compatible with PettingZoo API test
- ✓ Can run 1000 random games without errors
- ✓ Rewards sum correctly across all agents
- ✓ Action masking prevents invalid moves

---

### Phase 3: Baseline Agents (Priority 3)

#### 3.1 Agent Interface (`agents/base_agent.py`)
```python
"""
Abstract base class for all agents:

class BaseAgent(ABC):
    @abstractmethod
    def select_action(self, observation, valid_actions):
        pass
    
    @abstractmethod
    def update(self, experience):
        pass  # For learning agents
    
    def reset(self):
        pass  # Reset episode-specific state
"""
```

#### 3.2 Random Agent (`agents/random_agent.py`)
```python
"""
Plays random valid moves:
- Randomly selects from valid_actions
- No learning or strategy
- Useful for initial environment testing
"""
```

#### 3.3 Greedy Agent (`agents/greedy_agent.py`)
```python
"""
Simple heuristic strategy:
1. Always play lowest valid card/combination
2. Pass if no low cards available early in round
3. Play aggressively when few cards remain
4. Prioritize getting rid of single cards

Pseudo-logic:
- If hand size <= 3: play highest to win
- Else: play lowest valid card to conserve high cards
- If King/Queen: protect position
- If Slave: try to finish first to trigger King demotion
"""
```

**Test criteria:**
- ✓ Greedy agent beats random agent 70%+ of the time
- ✓ Agents complete games without errors
- ✓ Greedy agent shows consistent strategy

---

### Phase 4: RL Training (Priority 4)

#### 4.1 PPO Agent Wrapper (`agents/rl_agent.py`)
```python
"""
Wrapper around Stable-Baselines3 PPO:

Key features:
- Custom feature extractor for card game observation
- Action masking integration
- Model saving/loading

Network architecture:
Input (observation) 
  → Dense(256, relu)
  → Dense(256, relu)
  → Dense(128, relu)
  → Policy head (action probabilities)
  → Value head (state value)

Hyperparameters:
- learning_rate: 3e-4
- n_steps: 2048 (steps before update)
- batch_size: 64
- n_epochs: 10
- gamma: 0.99 (discount factor)
- gae_lambda: 0.95
- clip_range: 0.2
- ent_coef: 0.01 (exploration bonus)
"""
```

#### 4.2 Self-Play Training (`training/self_play.py`)
```python
"""
Training loop with self-play:

Algorithm:
1. Initialize 4 identical PPO agents
2. For each training iteration:
   a. Play N games with current agents
   b. Collect experience (states, actions, rewards)
   c. Update all agents with collected data
   d. Evaluate progress every K iterations
   e. Save checkpoints
3. Gradually increase opponent strength (curriculum)

Key methods:
- train(num_iterations, games_per_iteration)
- play_episode(agents) -> List[Experience]
- update_agents(experiences)
- save_checkpoint(iteration)

Training phases:
- Phase 1 (0-10k games): vs random agents
- Phase 2 (10k-50k): vs mix of random/greedy
- Phase 3 (50k+): pure self-play
"""
```

#### 4.3 Evaluation (`training/evaluator.py`)
```python
"""
Measure agent performance:

Metrics:
1. Win rate (% finishing as King)
2. Average position (1=King, 4=Slave)
3. Average game length (turns to finish)
4. Head-to-head matrix (agent vs agent)
5. Strategy diversity (action entropy)

Methods:
- evaluate(agent, opponent, num_games=100)
- tournament([agent1, agent2, ...])
- plot_learning_curve(checkpoint_dir)
- compare_strategies(agent1, agent2)

Evaluation schedule:
- Every 1000 training games
- Test against: random, greedy, previous checkpoints
"""
```

#### 4.4 Curriculum Learning (`training/curriculum.py`)
```python
"""
Progressive difficulty training:

Stages:
1. Simplified rules (no suit rankings)
2. Add suit rankings
3. Add card exchanges
4. Add King demotion rule
5. Full game complexity

Opponent mixing:
- Start: 75% random, 25% self
- Middle: 25% random, 50% greedy, 25% self
- End: 100% self-play

Methods:
- get_opponent_pool(stage) -> List[Agent]
- should_advance_stage(metrics) -> bool
- adapt_reward_weights(stage)
"""
```

---

### Phase 5: Training Infrastructure (Priority 5)

#### 5.1 Configuration (`config.py`)
```python
"""
Central configuration management:

GAME_CONFIG = {
    'num_players': 4,
    'deck_size': 52,
    'cards_per_player': 13,
    'exchange_cards_king_slave': 2,
    'exchange_cards_queen_commoner': 1,
}

TRAINING_CONFIG = {
    'total_timesteps': 1_000_000,
    'eval_frequency': 10_000,
    'checkpoint_frequency': 50_000,
    'num_eval_games': 100,
    'learning_rate': 3e-4,
    'batch_size': 64,
}

REWARD_CONFIG = {
    'king': 10.0,
    'queen': 5.0,
    'commoner': -5.0,
    'slave': -10.0,
    'cards_remaining_penalty': -0.01,
    'invalid_action_penalty': -1.0,
}

PATHS = {
    'models': './models',
    'logs': './logs',
    'checkpoints': './checkpoints',
}
"""
```

#### 5.2 Logging (`utils/logger.py`)
```python
"""
Comprehensive logging system:

Logs:
1. Training progress (TensorBoard)
   - Average reward per episode
   - Episode length
   - Win rate by position
   - Loss values

2. Game events (text logs)
   - Card plays each turn
   - Invalid action attempts
   - Final positions

3. Evaluation results (JSON)
   - Checkpoint performance
   - Head-to-head matrices

Methods:
- log_training_step(metrics)
- log_episode(game_state, actions)
- log_evaluation(results)
- export_metrics(format='csv'/'json')
"""
```

#### 5.3 Visualization (`utils/visualization.py`)
```python
"""
Training progress visualization:

Plots:
1. Learning curves (reward over time)
2. Win rate by position
3. Strategy heatmaps (which cards played when)
4. Action distribution
5. Elo rating progression

Methods:
- plot_training_progress(log_dir)
- plot_head_to_head(results)
- visualize_strategy(agent, num_games=10)
- export_replay(game_state, filename)
"""
```

---

### Phase 6: Testing & Validation (Priority 6)

#### 6.1 Unit Tests (`tests/`)
```python
"""
Test coverage:

test_card.py:
- Card comparison logic
- All 52 cards unique
- Encoding/decoding consistency

test_rules.py:
- Valid play detection for all combinations
- Play comparison correctness
- Edge cases (empty hand, single card, etc)

test_game_state.py:
- Correct card dealing
- Exchange mechanics
- Position determination
- King demotion rule

test_environment.py:
- PettingZoo API compliance
- Observation space correctness
- Action masking
- Reward calculation
"""
```

#### 6.2 Integration Tests
```python
"""
End-to-end scenarios:

1. Full game simulation (4 random agents)
2. Training for 1000 steps (doesn't crash)
3. Agent saves and loads correctly
4. Evaluation produces consistent results
"""
```

---

## Implementation Order (Step-by-Step)

### Week 1: Foundation
- [ ] Day 1-2: `card.py` + `deck.py` + tests
- [ ] Day 3-4: `rules.py` (play validation) + tests
- [ ] Day 5-7: `game_state.py` + integration tests

### Week 2: Environment
- [ ] Day 1-3: `observations.py` (state encoding)
- [ ] Day 4-7: `slave_env.py` (PettingZoo wrapper) + validation

### Week 3: Baseline Agents
- [ ] Day 1-2: `base_agent.py` + `random_agent.py`
- [ ] Day 3-5: `greedy_agent.py` + strategy tuning
- [ ] Day 6-7: Test agents play 1000+ games successfully

### Week 4: RL Integration
- [ ] Day 1-3: `rl_agent.py` (PPO wrapper)
- [ ] Day 4-5: Train single agent vs random opponents
- [ ] Day 6-7: Verify learning (reward increases)

### Week 5: Self-Play
- [ ] Day 1-3: `self_play.py` training loop
- [ ] Day 4-5: `evaluator.py` metrics
- [ ] Day 6-7: Run first self-play training (10k games)

### Week 6: Polish & Scale
- [ ] Day 1-2: `curriculum.py` staged training
- [ ] Day 3-4: Logging and visualization
- [ ] Day 5-7: Large-scale training run (100k+ games)

---

## Testing Checkpoints

### Checkpoint 1: Game Engine Works
```bash
python -m pytest tests/test_card.py tests/test_rules.py -v
python tests/test_game_state.py  # Manual play-through
```
**Expected:** All tests pass, can play complete game manually

### Checkpoint 2: Environment Valid
```bash
python -c "from pettingzoo.test import api_test; from environment.slave_env import SlaveEnv; api_test(SlaveEnv())"
```
**Expected:** PettingZoo API test passes

### Checkpoint 3: Agents Functional
```bash
python main.py --mode test --agents random random random random --games 100
```
**Expected:** 100 games complete, positions distributed

### Checkpoint 4: RL Learning
```bash
python main.py --mode train --timesteps 10000
tensorboard --logdir logs/
```
**Expected:** Reward curve trending upward

### Checkpoint 5: Self-Play Working
```bash
python main.py --mode selfplay --iterations 100
```
**Expected:** Agents improve beyond greedy baseline

---

## Key Configuration Values

```python
# config.py

# Action space size (simplified encoding)
# All single cards (52) + all pairs (78) + common straights (~50) + four-of-kinds (13) + pass (1)
ACTION_SPACE_SIZE = 194

# Observation space size
OBSERVATION_SIZE = 156  # As detailed in observations.py

# Training
TOTAL_TIMESTEPS = 1_000_000
EVAL_FREQUENCY = 10_000
N_EVAL_GAMES = 100

# PPO Hyperparameters
LEARNING_RATE = 3e-4
N_STEPS = 2048
BATCH_SIZE = 64
GAMMA = 0.99
```

---

## Expected Outcomes

### After 10k Training Games:
- Agents learn to play legal moves consistently
- Beat random agents 80%+ of the time
- Basic strategy emerges (playing low cards early)

### After 50k Training Games:
- Beat greedy agents 60%+ of the time
- Understand position importance (King protection)
- Card exchange strategies develop

### After 100k+ Training Games:
- Near-optimal play in most situations
- Complex strategies (setting up straights, bluffing with pass)
- Exploit opponent weaknesses

---

## Quick Start Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/ -v

# Train from scratch
python main.py --mode train --timesteps 100000

# Evaluate agent
python main.py --mode eval --model models/checkpoint_50000.zip

# Self-play tournament
python main.py --mode tournament --agents 4

# Watch game (human-readable output)
python main.py --mode watch --model models/best_agent.zip
```

---

## Success Criteria

✅ **Phase 1 Complete:** 4 random agents can play 1000 games without errors
✅ **Phase 2 Complete:** Environment passes PettingZoo API test
✅ **Phase 3 Complete:** Greedy agent beats random agent 70%+
✅ **Phase 4 Complete:** RL agent learns to beat random agents
✅ **Phase 5 Complete:** Self-play agents surpass greedy baseline
✅ **Phase 6 Complete:** Agent demonstrates strategic play (exchanges, position awareness)

---

## Notes for Claude Code

- **Start with Phase 1** completely before moving forward
- **Test each component** individually before integration
- Use **type hints** throughout for clarity
- Keep **detailed docstrings** for complex logic
- **Commit after each working component**
- Log extensively during development for debugging

This plan prioritizes getting a working prototype quickly, then iteratively improving strategy and performance.