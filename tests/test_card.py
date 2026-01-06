"""
Tests for the Card class and related functionality.
"""

import pytest
from game.card import Card, Rank, Suit, create_deck


class TestCardComparison:
    """Test card comparison logic."""

    def test_rank_comparison_same_suit(self):
        """Test that higher ranks beat lower ranks with same suit."""
        three_spades = Card(Rank.THREE, Suit.SPADES)
        king_spades = Card(Rank.KING, Suit.SPADES)
        ace_spades = Card(Rank.ACE, Suit.SPADES)
        two_spades = Card(Rank.TWO, Suit.SPADES)

        # 2 is highest
        assert two_spades > ace_spades
        assert two_spades > king_spades
        assert two_spades > three_spades

        # A > K > 3
        assert ace_spades > king_spades
        assert ace_spades > three_spades
        assert king_spades > three_spades

    def test_suit_comparison_same_rank(self):
        """Test that suit ordering works: Spades > Hearts > Clubs > Diamonds."""
        ace_diamonds = Card(Rank.ACE, Suit.DIAMONDS)
        ace_clubs = Card(Rank.ACE, Suit.CLUBS)
        ace_hearts = Card(Rank.ACE, Suit.HEARTS)
        ace_spades = Card(Rank.ACE, Suit.SPADES)

        # Spades is highest
        assert ace_spades > ace_hearts
        assert ace_spades > ace_clubs
        assert ace_spades > ace_diamonds

        # Hearts > Clubs > Diamonds
        assert ace_hearts > ace_clubs
        assert ace_hearts > ace_diamonds
        assert ace_clubs > ace_diamonds

    def test_rank_beats_suit(self):
        """Test that rank comparison takes precedence over suit."""
        three_spades = Card(Rank.THREE, Suit.SPADES)
        four_diamonds = Card(Rank.FOUR, Suit.DIAMONDS)

        # Even though Spades > Diamonds, 4 > 3 so four wins
        assert four_diamonds > three_spades
        assert three_spades < four_diamonds

    def test_equality(self):
        """Test that cards with same rank and suit are equal."""
        card1 = Card(Rank.KING, Suit.HEARTS)
        card2 = Card(Rank.KING, Suit.HEARTS)
        card3 = Card(Rank.KING, Suit.SPADES)

        assert card1 == card2
        assert card1 != card3

    def test_all_comparison_operators(self):
        """Test all comparison operators work correctly."""
        small_card = Card(Rank.THREE, Suit.DIAMONDS)
        medium_card = Card(Rank.KING, Suit.CLUBS)
        large_card = Card(Rank.TWO, Suit.SPADES)

        # Less than
        assert small_card < medium_card < large_card

        # Greater than
        assert large_card > medium_card > small_card

        # Less than or equal
        assert small_card <= medium_card
        assert small_card <= small_card

        # Greater than or equal
        assert large_card >= medium_card
        assert large_card >= large_card


class TestCardEncoding:
    """Test card integer encoding and decoding."""

    def test_encoding_uniqueness(self):
        """Test that all 52 cards have unique integer IDs."""
        deck = create_deck()
        ids = [card.to_int() for card in deck]

        assert len(ids) == 52
        assert len(set(ids)) == 52  # All unique
        assert min(ids) == 0
        assert max(ids) == 51

    def test_encoding_decoding_roundtrip(self):
        """Test that encoding and decoding are inverse operations."""
        deck = create_deck()

        for card in deck:
            card_id = card.to_int()
            decoded_card = Card.from_int(card_id)
            assert decoded_card == card

    def test_specific_card_encodings(self):
        """Test specific card encodings match expected values."""
        # 3♦ should be 0
        three_diamonds = Card(Rank.THREE, Suit.DIAMONDS)
        assert three_diamonds.to_int() == 0

        # 3♠ should be 3
        three_spades = Card(Rank.THREE, Suit.SPADES)
        assert three_spades.to_int() == 3

        # 2♠ should be 51
        two_spades = Card(Rank.TWO, Suit.SPADES)
        assert two_spades.to_int() == 51

    def test_from_int_invalid_id(self):
        """Test that invalid card IDs raise ValueError."""
        with pytest.raises(ValueError):
            Card.from_int(-1)

        with pytest.raises(ValueError):
            Card.from_int(52)

        with pytest.raises(ValueError):
            Card.from_int(100)


class TestCardRepresentation:
    """Test card string representation."""

    def test_repr(self):
        """Test that card representation is human-readable."""
        card = Card(Rank.ACE, Suit.SPADES)
        assert repr(card) == "A♠"

        card = Card(Rank.TEN, Suit.HEARTS)
        assert repr(card) == "10♥"

        card = Card(Rank.THREE, Suit.DIAMONDS)
        assert repr(card) == "3♦"

        card = Card(Rank.TWO, Suit.CLUBS)
        assert repr(card) == "2♣"

    def test_str(self):
        """Test that str() returns the same as repr()."""
        card = Card(Rank.KING, Suit.HEARTS)
        assert str(card) == repr(card)


class TestCardHashability:
    """Test that cards can be used in sets and as dict keys."""

    def test_cards_in_set(self):
        """Test that cards can be added to sets."""
        card1 = Card(Rank.ACE, Suit.SPADES)
        card2 = Card(Rank.ACE, Suit.SPADES)
        card3 = Card(Rank.KING, Suit.SPADES)

        card_set = {card1, card2, card3}
        assert len(card_set) == 2  # card1 and card2 are the same

    def test_cards_as_dict_keys(self):
        """Test that cards can be used as dictionary keys."""
        card_counts = {}
        card = Card(Rank.QUEEN, Suit.HEARTS)

        card_counts[card] = 1
        card_counts[card] += 1

        assert card_counts[card] == 2


class TestCreateDeck:
    """Test deck creation function."""

    def test_deck_size(self):
        """Test that a full deck has 52 cards."""
        deck = create_deck()
        assert len(deck) == 52

    def test_deck_uniqueness(self):
        """Test that all cards in deck are unique."""
        deck = create_deck()
        card_set = set(deck)
        assert len(card_set) == 52

    def test_deck_contains_all_ranks_and_suits(self):
        """Test that deck contains all rank-suit combinations."""
        deck = create_deck()

        for rank in Rank:
            for suit in Suit:
                expected_card = Card(rank, suit)
                assert expected_card in deck

    def test_deck_has_correct_rank_distribution(self):
        """Test that deck has exactly 4 cards of each rank."""
        deck = create_deck()
        rank_counts = {}

        for card in deck:
            rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1

        for rank in Rank:
            assert rank_counts[rank] == 4

    def test_deck_has_correct_suit_distribution(self):
        """Test that deck has exactly 13 cards of each suit."""
        deck = create_deck()
        suit_counts = {}

        for card in deck:
            suit_counts[card.suit] = suit_counts.get(card.suit, 0) + 1

        for suit in Suit:
            assert suit_counts[suit] == 13
