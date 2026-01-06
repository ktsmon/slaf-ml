"""
Tests for strategic card exchange functionality.
"""

import pytest
from game.game_state import GameState, Position
from game.card import Card, Rank, Suit


class TestStrategicCardExchange:
    """Test strategic card exchange mechanics."""

    def test_king_can_choose_which_cards_to_give(self):
        """Test that King can strategically choose which cards to give."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Play full first round
        max_turns = 200
        turn_count = 0

        while not game.is_game_over() and turn_count < max_turns:
            current = game.current_player
            valid_plays = game.get_valid_plays(current)

            if len(valid_plays) > 0:
                play = valid_plays[0]
                for p in valid_plays:
                    if len(p) > 0:
                        play = p
                        break
                game.play_cards(current, play)

            turn_count += 1

        # Find King player
        king_id = None
        for player_id, position in game.positions.items():
            if position == Position.KING:
                king_id = player_id
                break

        assert king_id is not None

        # Start new round but don't exchange yet - just deal
        game.is_first_round = False
        from game.deck import Deck
        deck = Deck()
        deck.shuffle(seed=123)
        hands = deck.deal(game.num_players)
        game.hands = {i: hands[i] for i in range(game.num_players)}

        # King can see what they will receive
        cards_from_slave = game.get_cards_to_receive_from_slave()
        assert cards_from_slave is not None
        assert len(cards_from_slave) == 2

        # King's current hand
        king_hand = game.get_hand(king_id)
        assert len(king_hand) == 13

        # King chooses to give specific cards (not necessarily lowest)
        # For example, if King has three 3s, keep them and give 4s or 5s instead
        sorted_hand = sorted(king_hand)

        # Strategic choice: give cards at positions 2 and 3 instead of 0 and 1
        king_gives = [sorted_hand[2], sorted_hand[3]]

        # Perform exchange with strategic choice
        game._exchange_cards(king_gives=king_gives)

        # Verify the specific cards were removed
        new_king_hand = game.get_hand(king_id)
        assert sorted_hand[2] not in new_king_hand
        assert sorted_hand[3] not in new_king_hand

        # Verify King received the slave's cards
        assert cards_from_slave[0] in new_king_hand
        assert cards_from_slave[1] in new_king_hand

        # Verify hand size is still 13
        assert len(new_king_hand) == 13

    def test_queen_can_choose_which_card_to_give(self):
        """Test that Queen can strategically choose which card to give."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Play full first round
        max_turns = 200
        turn_count = 0

        while not game.is_game_over() and turn_count < max_turns:
            current = game.current_player
            valid_plays = game.get_valid_plays(current)

            if len(valid_plays) > 0:
                play = valid_plays[0]
                for p in valid_plays:
                    if len(p) > 0:
                        play = p
                        break
                game.play_cards(current, play)

            turn_count += 1

        # Find Queen player
        queen_id = None
        for player_id, position in game.positions.items():
            if position == Position.QUEEN:
                queen_id = player_id
                break

        assert queen_id is not None

        # Start new round but don't exchange yet - just deal
        game.is_first_round = False
        from game.deck import Deck
        deck = Deck()
        deck.shuffle(seed=456)
        hands = deck.deal(game.num_players)
        game.hands = {i: hands[i] for i in range(game.num_players)}

        # Queen can see what they will receive
        card_from_commoner = game.get_cards_to_receive_from_commoner()
        assert card_from_commoner is not None

        # Queen's current hand
        queen_hand = game.get_hand(queen_id)
        assert len(queen_hand) == 13

        # Queen chooses to give a specific card (not necessarily lowest)
        sorted_hand = sorted(queen_hand)
        queen_gives = sorted_hand[1]  # Give 2nd lowest instead of lowest

        # Perform exchange with strategic choice
        game._exchange_cards(queen_gives=queen_gives)

        # Verify the specific card was removed
        new_queen_hand = game.get_hand(queen_id)
        assert queen_gives not in new_queen_hand

        # Verify Queen received the commoner's card
        assert card_from_commoner in new_queen_hand

        # Verify hand size is still 13
        assert len(new_queen_hand) == 13

    def test_default_exchange_still_works(self):
        """Test that default exchange (lowest cards) still works when no choice given."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Play full first round
        max_turns = 200
        turn_count = 0

        while not game.is_game_over() and turn_count < max_turns:
            current = game.current_player
            valid_plays = game.get_valid_plays(current)

            if len(valid_plays) > 0:
                play = valid_plays[0]
                for p in valid_plays:
                    if len(p) > 0:
                        play = p
                        break
                game.play_cards(current, play)

            turn_count += 1

        # Find King
        king_id = None
        for player_id, position in game.positions.items():
            if position == Position.KING:
                king_id = player_id
                break

        # Start new round with default exchange (no parameters)
        game.start_new_round()

        # Verify exchange happened (King should have 13 cards)
        assert len(game.get_hand(king_id)) == 13

        # Game should be ready to play
        assert not game.is_game_over()

    def test_invalid_cards_fallback_to_default(self):
        """Test that invalid card choices fallback to default behavior."""
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Play full first round
        max_turns = 200
        turn_count = 0

        while not game.is_game_over() and turn_count < max_turns:
            current = game.current_player
            valid_plays = game.get_valid_plays(current)

            if len(valid_plays) > 0:
                play = valid_plays[0]
                for p in valid_plays:
                    if len(p) > 0:
                        play = p
                        break
                game.play_cards(current, play)

            turn_count += 1

        # Find King
        king_id = None
        for player_id, position in game.positions.items():
            if position == Position.KING:
                king_id = player_id
                break

        # Start new round setup
        game.is_first_round = False
        from game.deck import Deck
        deck = Deck()
        deck.shuffle(seed=789)
        hands = deck.deal(game.num_players)
        game.hands = {i: hands[i] for i in range(game.num_players)}

        king_hand_before = sorted(game.get_hand(king_id))

        # Try to give cards that King doesn't have
        invalid_cards = [Card(Rank.TWO, Suit.SPADES), Card(Rank.ACE, Suit.HEARTS)]

        # Should fallback to giving lowest 2 cards
        game._exchange_cards(king_gives=invalid_cards)

        # Verify exchange still happened
        new_king_hand = game.get_hand(king_id)
        assert len(new_king_hand) == 13

    def test_strategic_example_keeping_triplets(self):
        """
        Test strategic example: King has three 3s, so keeps them and gives 4s/5s instead.
        """
        game = GameState(num_players=4)
        game.reset(seed=42)

        # Play full first round
        max_turns = 200
        turn_count = 0

        while not game.is_game_over() and turn_count < max_turns:
            current = game.current_player
            valid_plays = game.get_valid_plays(current)

            if len(valid_plays) > 0:
                play = valid_plays[0]
                for p in valid_plays:
                    if len(p) > 0:
                        play = p
                        break
                game.play_cards(current, play)

            turn_count += 1

        # Find King
        king_id = None
        for player_id, position in game.positions.items():
            if position == Position.KING:
                king_id = player_id
                break

        # Manually set up a scenario where King has multiple cards of same rank
        game.is_first_round = False

        # Create a custom hand with three 3s
        custom_king_hand = [
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.THREE, Suit.CLUBS),
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.FOUR, Suit.DIAMONDS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.SPADES),
            Card(Rank.SIX, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.CLUBS),
            Card(Rank.EIGHT, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.SPADES),
            Card(Rank.TEN, Suit.HEARTS),
            Card(Rank.JACK, Suit.CLUBS),
            Card(Rank.QUEEN, Suit.DIAMONDS),
        ]

        game.hands[king_id] = custom_king_hand

        # Set up other hands
        from game.deck import Deck
        deck = Deck()
        deck.shuffle(seed=100)
        hands = deck.deal(game.num_players)
        for pid in range(game.num_players):
            if pid != king_id:
                game.hands[pid] = hands[pid]

        # King decides to keep the three 3s and give the two 4s instead
        strategic_choice = [Card(Rank.FOUR, Suit.DIAMONDS), Card(Rank.FOUR, Suit.CLUBS)]

        # Perform exchange
        game._exchange_cards(king_gives=strategic_choice)

        # Verify King still has all three 3s
        final_hand = game.get_hand(king_id)
        threes_count = sum(1 for card in final_hand if card.rank == Rank.THREE)
        assert threes_count == 3

        # Verify King gave away the 4s
        fours_count = sum(1 for card in final_hand if card.rank == Rank.FOUR)
        assert fours_count == 0
