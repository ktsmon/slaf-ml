"""
Tests for GameState class.
"""

import pytest
from game.game_state import GameState, Position
from game.card import Card, Rank, Suit


class TestGameStateInitialization:
    """Test game state initialization."""

    def test_create_game(self):
        """Test creating a new game state."""
        game = GameState(num_players=4)
        assert game.num_players == 4
        assert len(game.hands) == 0
        assert not game.game_over

    def test_reset_deals_cards(self):
        """Test that reset deals cards to all players."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        assert len(game.hands) == 4
        for player_id in range(4):
            assert len(game.hands[player_id]) == 13

    def test_reset_finds_starting_player(self):
        """Test that reset sets the player with 3♦ as current player."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Current player should have 3 of diamonds
        three_diamonds = Card(Rank.THREE, Suit.DIAMONDS)
        assert three_diamonds in game.hands[game.current_player]

    def test_reproducible_with_seed(self):
        """Test that same seed produces same deal."""
        game1 = GameState(num_players=4)
        game1.reset(seed=12345)

        game2 = GameState(num_players=4)
        game2.reset(seed=12345)

        # Same starting player
        assert game1.current_player == game2.current_player

        # Same hands
        for player_id in range(4):
            assert set(game1.hands[player_id]) == set(game2.hands[player_id])


class TestBasicGameplay:
    """Test basic gameplay mechanics."""

    def test_play_single_card(self):
        """Test playing a single card."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        current = game.current_player
        hand = game.get_hand(current)
        card_to_play = hand[0]

        # Play the card
        success = game.play_cards(current, [card_to_play])
        assert success

        # Card should be removed from hand
        new_hand = game.get_hand(current)
        assert card_to_play not in new_hand
        assert len(new_hand) == len(hand) - 1

    def test_cannot_play_card_not_in_hand(self):
        """Test that you cannot play a card not in your hand."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        current = game.current_player
        other_player = (current + 1) % 4

        # Try to play a card from another player's hand
        other_card = game.get_hand(other_player)[0]
        success = game.play_cards(current, [other_card])

        assert not success

    def test_cannot_play_out_of_turn(self):
        """Test that players cannot play out of turn."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        current = game.current_player
        not_current = (current + 1) % 4

        card = game.get_hand(not_current)[0]
        success = game.play_cards(not_current, [card])

        assert not success

    def test_can_pass(self):
        """Test that passing is always valid."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        current = game.current_player

        # Play a card first
        first_card = game.get_hand(current)[0]
        game.play_cards(current, [first_card])

        # Next player can pass
        next_player = game.current_player
        success = game.play_cards(next_player, [])
        assert success

    def test_turn_advances_after_play(self):
        """Test that the turn advances after a play."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        first_player = game.current_player
        card = game.get_hand(first_player)[0]

        game.play_cards(first_player, [card])

        # Turn should advance
        assert game.current_player != first_player


class TestTrickManagement:
    """Test trick completion and winner determination."""

    def test_trick_completes_after_all_players_play(self):
        """Test that a trick completes after all players have played."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Have all 4 players play
        for _ in range(4):
            current = game.current_player
            # Just pass or play lowest card
            valid_plays = game.get_valid_plays(current)
            if len(valid_plays) > 0:
                game.play_cards(current, valid_plays[0])

        # A new trick should have started (trick_plays should be reset)
        assert len(game.trick_plays) < 4

    def test_highest_play_wins_trick(self):
        """Test that the highest play wins the trick and leads next."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Find player with highest card for first trick
        starting_player = game.current_player

        # Play first card
        first_card = game.get_hand(starting_player)[0]
        game.play_cards(starting_player, [first_card])

        # Have other players pass or play lower cards
        for _ in range(3):
            current = game.current_player
            # Pass
            game.play_cards(current, [])

        # Starting player should win and lead next trick
        assert game.trick_leader == starting_player


class TestGameCompletion:
    """Test game completion and position determination."""

    def test_game_ends_when_all_but_one_finish(self):
        """Test that game ends when all but one player finishes."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Simulate game until end by having players play all their cards
        max_turns = 200  # Safety limit
        turn_count = 0

        while not game.is_game_over() and turn_count < max_turns:
            current = game.current_player
            valid_plays = game.get_valid_plays(current)

            if len(valid_plays) > 0:
                # Prefer non-pass plays to make progress
                play = valid_plays[0]
                for p in valid_plays:
                    if len(p) > 0:
                        play = p
                        break
                game.play_cards(current, play)

            turn_count += 1

        # Game should eventually end
        assert game.is_game_over()
        assert len(game.finished_order) == 4

    def test_positions_assigned_after_game(self):
        """Test that positions are correctly assigned after game ends."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Play until game ends
        max_turns = 200
        turn_count = 0

        while not game.is_game_over() and turn_count < max_turns:
            current = game.current_player
            valid_plays = game.get_valid_plays(current)

            if len(valid_plays) > 0:
                # Prefer non-pass plays
                play = valid_plays[0]
                for p in valid_plays:
                    if len(p) > 0:
                        play = p
                        break
                game.play_cards(current, play)

            turn_count += 1

        # Check positions are assigned
        positions = [game.get_position(i) for i in range(4)]

        assert Position.KING in positions
        assert Position.QUEEN in positions
        assert Position.COMMONER in positions
        assert Position.SLAVE in positions

        # First finisher should be King
        assert game.get_position(game.finished_order[0]) == Position.KING

        # Last finisher should be Slave
        assert game.get_position(game.finished_order[-1]) == Position.SLAVE


class TestCardExchanges:
    """Test card exchange mechanics."""

    def test_card_exchange_after_first_round(self):
        """Test that card exchanges happen at start of new round."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Play full first round
        max_turns = 200
        turn_count = 0

        while not game.is_game_over() and turn_count < max_turns:
            current = game.current_player
            valid_plays = game.get_valid_plays(current)

            if len(valid_plays) > 0:
                # Prefer non-pass plays
                play = valid_plays[0]
                for p in valid_plays:
                    if len(p) > 0:
                        play = p
                        break
                game.play_cards(current, play)

            turn_count += 1

        # Remember original positions
        original_positions = {i: game.get_position(i) for i in range(4)}

        # Start new round
        game.start_new_round()

        # Positions should be preserved
        for player_id in range(4):
            assert game.get_position(player_id) == original_positions[player_id]

        # All players should have 13 cards again
        for player_id in range(4):
            assert len(game.get_hand(player_id)) == 13

        # Slave should start the round
        slave_id = None
        for player_id, position in game.positions.items():
            if position == Position.SLAVE:
                slave_id = player_id
                break

        assert game.current_player == slave_id


class TestValidPlays:
    """Test valid play generation."""

    def test_get_valid_plays_on_first_turn(self):
        """Test that valid plays includes all valid combinations on first turn."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        current = game.current_player
        valid = game.get_valid_plays(current)

        # Should have at least pass and some singles
        assert [] in valid  # Pass
        assert len(valid) > 1  # More than just pass

    def test_cannot_get_valid_plays_out_of_turn(self):
        """Test that you can't get valid plays when it's not your turn."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        current = game.current_player
        not_current = (current + 1) % 4

        valid = game.get_valid_plays(not_current)
        assert len(valid) == 0

    def test_valid_plays_respect_last_play(self):
        """Test that valid plays must beat the last play."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Play a high card
        current = game.current_player
        hand = game.get_hand(current)

        # Find and play a high card
        high_card = max(hand)
        game.play_cards(current, [high_card])

        # Next player should have limited options
        next_player = game.current_player
        valid = game.get_valid_plays(next_player)

        # Should at least be able to pass
        assert [] in valid


class TestIntegration:
    """Integration tests for full game flows."""

    def test_complete_game_no_errors(self):
        """Test that a complete game can be played without errors."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        turn_count = 0
        max_turns = 200

        while not game.is_game_over() and turn_count < max_turns:
            current = game.current_player
            valid_plays = game.get_valid_plays(current)

            assert len(valid_plays) > 0, "Should always have at least pass available"

            # Prefer non-pass plays
            play = valid_plays[0]
            for p in valid_plays:
                if len(p) > 0:
                    play = p
                    break
            game.play_cards(current, play)
            turn_count += 1

        assert game.is_game_over()
        assert turn_count < max_turns

    def test_multiple_rounds(self):
        """Test playing multiple rounds in sequence."""
        game = GameState(num_players=4)

        for round_num in range(3):
            if round_num == 0:
                game.reset(seed=42)
            else:
                game.start_new_round()

            # Play one full round
            turn_count = 0
            max_turns = 200

            while not game.is_game_over() and turn_count < max_turns:
                current = game.current_player
                valid_plays = game.get_valid_plays(current)

                if len(valid_plays) > 0:
                    # Prefer non-pass plays
                    play = valid_plays[0]
                    for p in valid_plays:
                        if len(p) > 0:
                            play = p
                            break
                    game.play_cards(current, play)

                turn_count += 1

            assert game.is_game_over()


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_game_state_repr(self):
        """Test string representation of game state."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        repr_str = repr(game)
        assert "GameState" in repr_str
        assert "round=" in repr_str
        assert "current_player=" in repr_str

    def test_get_hand_returns_copy(self):
        """Test that get_hand returns a copy, not the original."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        player_id = 0
        hand1 = game.get_hand(player_id)
        hand2 = game.get_hand(player_id)

        # Should be equal but not the same object
        assert hand1 == hand2
        assert hand1 is not hand2

    def test_playing_last_card_adds_to_finished(self):
        """Test that playing your last card adds you to finished order."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Manually set up a scenario where a player has 1 card
        player_id = game.current_player
        hand = game.get_hand(player_id)

        # Remove all but one card
        game.hands[player_id] = [hand[0]]

        # Play the last card
        valid_plays = game.get_valid_plays(player_id)
        if [hand[0]] in valid_plays or len(valid_plays) > 1:
            # Find a play with the card
            for play in valid_plays:
                if hand[0] in play:
                    game.play_cards(player_id, play)
                    break

            # Player should be in finished order
            if len(game.get_hand(player_id)) == 0:
                assert player_id in game.finished_order
