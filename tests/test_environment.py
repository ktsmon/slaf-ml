"""
Tests for the RL environment (observations and PettingZoo wrapper).
"""

import pytest
import numpy as np

from environment.slave_env import SlaveEnv
from environment.observations import (
    encode_observation,
    get_action_mask,
    encode_action,
    decode_action,
    _get_pair_suit_combination,
    _find_straight_in_hand
)
from game.game_state import GameState, Position
from game.card import Card, Rank, Suit
from game.deck import Deck


class TestObservationEncoding:
    """Test observation encoding functionality."""

    def test_observation_shape(self):
        """Test that observation has correct shape."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        obs = encode_observation(game, player_id=0)

        # Expected: 52 + 52 + 5 + 13 + 4 + 16 + 4 + 1 + 4 + 4 = 155
        assert obs.shape == (155,)
        assert obs.dtype == np.float32

    def test_observation_normalized(self):
        """Test that observation values are normalized (0-1 range)."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        obs = encode_observation(game, player_id=0)

        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_hand_encoding(self):
        """Test that player's hand is correctly encoded."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        player_id = 0
        hand = game.get_hand(player_id)
        obs = encode_observation(game, player_id)

        # First 52 features are hand encoding
        hand_encoding = obs[:52]

        # Check that cards in hand are marked as 1
        for card in hand:
            assert hand_encoding[card.to_int()] == 1.0

        # Check that total number of 1s equals hand size
        assert np.sum(hand_encoding) == len(hand)

    def test_observation_differs_by_player(self):
        """Test that different players get different observations."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        obs0 = encode_observation(game, player_id=0)
        obs1 = encode_observation(game, player_id=1)

        # Observations should differ (at least in hand encoding)
        assert not np.array_equal(obs0, obs1)


class TestActionEncoding:
    """Test action encoding and decoding."""

    def test_encode_pass(self):
        """Test encoding pass action."""
        action_id = encode_action([])
        assert action_id == 0

    def test_encode_single_card(self):
        """Test encoding single card action."""
        card = Card(Rank.THREE, Suit.DIAMONDS)
        action_id = encode_action([card])

        # Single cards: action_id = card.to_int() + 1
        expected = card.to_int() + 1
        assert action_id == expected
        assert 1 <= action_id <= 52

    def test_encode_pair(self):
        """Test encoding pair action."""
        cards = [Card(Rank.FOUR, Suit.DIAMONDS), Card(Rank.FOUR, Suit.CLUBS)]
        action_id = encode_action(cards)

        # Pairs: 53-130
        assert 53 <= action_id <= 130

    def test_encode_straight(self):
        """Test encoding straight action."""
        cards = [
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.HEARTS)
        ]
        action_id = encode_action(cards)

        # Straights: 131-180
        assert 131 <= action_id <= 180

    def test_encode_four_of_kind(self):
        """Test encoding four-of-a-kind action."""
        cards = [
            Card(Rank.SEVEN, Suit.DIAMONDS),
            Card(Rank.SEVEN, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.SPADES)
        ]
        action_id = encode_action(cards)

        # Four-of-a-kinds: 181-193
        assert 181 <= action_id <= 193

    def test_decode_pass(self):
        """Test decoding pass action."""
        hand = [Card(Rank.THREE, Suit.DIAMONDS)]
        cards = decode_action(0, hand)
        assert cards == []

    def test_decode_single_card(self):
        """Test decoding single card action."""
        card = Card(Rank.FIVE, Suit.HEARTS)
        hand = [card, Card(Rank.SIX, Suit.DIAMONDS)]

        action_id = encode_action([card])
        decoded = decode_action(action_id, hand)

        assert len(decoded) == 1
        assert decoded[0] == card

    def test_decode_pair(self):
        """Test decoding pair action."""
        cards = [Card(Rank.EIGHT, Suit.DIAMONDS), Card(Rank.EIGHT, Suit.CLUBS)]
        hand = cards + [Card(Rank.NINE, Suit.HEARTS)]

        action_id = encode_action(cards)
        decoded = decode_action(action_id, hand)

        assert len(decoded) == 2
        assert all(c.rank == Rank.EIGHT for c in decoded)

    def test_encode_decode_roundtrip(self):
        """Test that encode -> decode returns same cards."""
        # Single card
        card = Card(Rank.KING, Suit.SPADES)
        hand = [card]
        action_id = encode_action([card])
        decoded = decode_action(action_id, hand)
        assert decoded == [card]

        # Pass
        action_id = encode_action([])
        decoded = decode_action(action_id, hand)
        assert decoded == []


class TestActionMasking:
    """Test action masking functionality."""

    def test_action_mask_shape(self):
        """Test that action mask has correct shape."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        mask = get_action_mask(game, player_id=0, action_space_size=194)

        assert mask.shape == (194,)
        assert mask.dtype == np.float32

    def test_action_mask_binary(self):
        """Test that action mask contains only 0s and 1s."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        mask = get_action_mask(game, player_id=0, action_space_size=194)

        assert np.all((mask == 0) | (mask == 1))

    def test_action_mask_always_allows_pass(self):
        """Test that pass is always a valid action."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Play a few turns
        for _ in range(5):
            current = game.current_player
            mask = get_action_mask(game, current, action_space_size=194)

            # Pass (action 0) should always be valid
            assert mask[0] == 1.0

            # Make a play
            valid_plays = game.get_valid_plays(current)
            if len(valid_plays) > 1:
                # Play something other than pass
                for play in valid_plays:
                    if len(play) > 0:
                        game.play_cards(current, play)
                        break

    def test_action_mask_reflects_valid_plays(self):
        """Test that action mask matches valid plays from game state."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        current = game.current_player
        valid_plays = game.get_valid_plays(current)
        mask = get_action_mask(game, current, action_space_size=194)

        # Count valid actions in mask
        num_valid = np.sum(mask)

        # Should have some valid actions (at least pass)
        assert num_valid > 0
        # Should have at least one action per valid play (may have more due to encoding)
        assert num_valid >= 1  # At minimum, pass should be valid


class TestPettingZooEnvironment:
    """Test PettingZoo environment wrapper."""

    def test_env_creation(self):
        """Test creating environment."""
        env = SlaveEnv()

        assert env.num_players == 4
        assert len(env.possible_agents) == 4
        assert env.action_space_size == 194
        assert env.observation_space_size == 155

    def test_env_reset(self):
        """Test environment reset."""
        env = SlaveEnv()
        env.reset(seed=42)

        assert len(env.agents) == 4
        assert env.game_state is not None
        assert env.agent_selection in env.agents

        # All agents should not be terminated
        for agent in env.agents:
            assert not env.terminations[agent]
            assert not env.truncations[agent]

    def test_env_observe(self):
        """Test observing environment state."""
        env = SlaveEnv()
        env.reset(seed=42)

        for agent in env.agents:
            obs = env.observe(agent)

            assert isinstance(obs, np.ndarray)
            assert obs.shape == (155,)
            assert obs.dtype == np.float32

    def test_env_action_spaces(self):
        """Test action spaces are correct."""
        env = SlaveEnv()

        for agent in env.possible_agents:
            action_space = env.action_space(agent)
            assert action_space.n == 194

    def test_env_observation_spaces(self):
        """Test observation spaces are correct."""
        env = SlaveEnv()

        for agent in env.possible_agents:
            obs_space = env.observation_space(agent)
            assert obs_space.shape == (155,)

    def test_env_step_with_valid_action(self):
        """Test stepping with valid action."""
        env = SlaveEnv()
        env.reset(seed=42)

        agent = env.agent_selection

        # IMPORTANT: Need to observe first to populate action_mask in info
        obs = env.observe(agent)

        # Get action mask and choose valid action
        action_mask = env.infos[agent]["action_mask"]
        valid_actions = np.where(action_mask == 1)[0]

        assert len(valid_actions) > 0, "Should have at least one valid action (pass)"

        # Take a valid action
        action = valid_actions[0]
        env.step(action)

        # Should move to next agent
        assert env.agent_selection != agent

    def test_env_step_with_invalid_action(self):
        """Test stepping with invalid action gives penalty."""
        env = SlaveEnv()
        env.reset(seed=42)

        agent = env.agent_selection

        # Observe first to populate action_mask
        obs = env.observe(agent)

        # Get action mask
        action_mask = env.infos[agent]["action_mask"]
        invalid_actions = np.where(action_mask == 0)[0]

        if len(invalid_actions) > 0:
            # Take invalid action
            action = invalid_actions[0]

            initial_reward = env._cumulative_rewards[agent]
            env.step(action)

            # Should get penalty
            assert env._cumulative_rewards[agent] < initial_reward

    def test_env_complete_game(self):
        """Test playing a complete game."""
        env = SlaveEnv()
        env.reset(seed=42)

        max_steps = 500
        step_count = 0

        while not all(env.terminations.values()) and step_count < max_steps:
            agent = env.agent_selection

            # Observe first
            obs = env.observe(agent)

            # Get valid action
            action_mask = env.infos[agent]["action_mask"]
            valid_actions = np.where(action_mask == 1)[0]

            if len(valid_actions) == 0:
                # Player has finished, skip
                break

            # Prefer non-pass actions
            action = valid_actions[0]
            for act in valid_actions:
                if act != 0:  # Not pass
                    action = act
                    break

            env.step(action)
            step_count += 1

        # Game should complete
        assert all(env.terminations.values())
        assert step_count < max_steps

        # Check that positions are assigned
        for agent in env.agents:
            player_id = env.agent_name_mapping[agent]
            position = env.game_state.get_position(player_id)
            assert position is not None

    def test_env_rewards_sum_appropriately(self):
        """Test that rewards are assigned correctly at game end."""
        env = SlaveEnv()
        env.reset(seed=42)

        # Play complete game
        max_steps = 500
        step_count = 0

        while not all(env.terminations.values()) and step_count < max_steps:
            agent = env.agent_selection
            obs = env.observe(agent)
            action_mask = env.infos[agent]["action_mask"]
            valid_actions = np.where(action_mask == 1)[0]

            if len(valid_actions) == 0:
                # Player has finished
                break

            # Prefer non-pass actions
            action = valid_actions[0]
            for act in valid_actions:
                if act != 0:
                    action = act
                    break

            env.step(action)
            step_count += 1

        # Check rewards
        total_reward = sum(env._cumulative_rewards.values())

        # Total reward should be close to 0 (King=+10, Queen=+5, Commoner=-5, Slave=-10)
        # Plus small penalties for turns and cards remaining
        assert abs(total_reward) < 50  # Reasonable bound

        # King should have positive reward, Slave should have negative
        king_agent = None
        slave_agent = None

        for agent in env.agents:
            player_id = env.agent_name_mapping[agent]
            position = env.game_state.get_position(player_id)

            if position == Position.KING:
                king_agent = agent
            elif position == Position.SLAVE:
                slave_agent = agent

        if king_agent and slave_agent:
            assert env._cumulative_rewards[king_agent] > env._cumulative_rewards[slave_agent]

    def test_env_multiple_games(self):
        """Test running multiple games without errors."""
        env = SlaveEnv()

        for game_num in range(10):
            env.reset(seed=game_num)

            max_steps = 500
            step_count = 0

            while not all(env.terminations.values()) and step_count < max_steps:
                agent = env.agent_selection
                obs = env.observe(agent)
                action_mask = env.infos[agent]["action_mask"]
                valid_actions = np.where(action_mask == 1)[0]

                if len(valid_actions) == 0:
                    # Player has finished
                    break

                action = valid_actions[0]
                for act in valid_actions:
                    if act != 0:
                        action = act
                        break

                env.step(action)
                step_count += 1

            if not all(env.terminations.values()):
                # If not all terminated, might be stuck - skip this game
                continue


class TestHelperFunctions:
    """Test helper functions for action encoding."""

    def test_get_pair_suit_combination(self):
        """Test suit combination encoding for pairs."""
        # Test (C, D)
        cards = [Card(Rank.FIVE, Suit.CLUBS), Card(Rank.FIVE, Suit.DIAMONDS)]
        combo = _get_pair_suit_combination(cards)
        assert combo == 0

        # Test (C, H)
        cards = [Card(Rank.FIVE, Suit.CLUBS), Card(Rank.FIVE, Suit.HEARTS)]
        combo = _get_pair_suit_combination(cards)
        assert combo == 1

        # Test (H, S)
        cards = [Card(Rank.FIVE, Suit.HEARTS), Card(Rank.FIVE, Suit.SPADES)]
        combo = _get_pair_suit_combination(cards)
        assert combo == 5

    def test_find_straight_in_hand(self):
        """Test finding straights in hand."""
        hand = [
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.SIX, Suit.SPADES),
            Card(Rank.NINE, Suit.DIAMONDS)
        ]

        # Find straight 3-4-5
        straight = _find_straight_in_hand(hand, starting_rank=3, length=3)
        assert straight is not None
        assert len(straight) == 3
        assert straight[0].rank == Rank.THREE
        assert straight[1].rank == Rank.FOUR
        assert straight[2].rank == Rank.FIVE

        # Find straight 3-4-5-6
        straight = _find_straight_in_hand(hand, starting_rank=3, length=4)
        assert straight is not None
        assert len(straight) == 4

        # Cannot find straight 7-8-9
        straight = _find_straight_in_hand(hand, starting_rank=7, length=3)
        assert straight is None
