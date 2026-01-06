"""
Tests for the Deck class.
"""

import pytest
from game.deck import Deck
from game.card import Card, Rank, Suit


class TestDeckCreation:
    """Test deck initialization."""

    def test_new_deck_has_52_cards(self):
        """Test that a new deck has 52 cards."""
        deck = Deck()
        assert len(deck) == 52

    def test_new_deck_has_all_unique_cards(self):
        """Test that all cards in a new deck are unique."""
        deck = Deck()
        card_set = set(deck.cards)
        assert len(card_set) == 52

    def test_deck_repr(self):
        """Test deck string representation."""
        deck = Deck()
        assert repr(deck) == "Deck(52 cards)"


class TestDeckShuffle:
    """Test deck shuffling."""

    def test_shuffle_changes_order(self):
        """Test that shuffling changes the card order."""
        deck1 = Deck()
        deck2 = Deck()

        # Get original order
        original_order = [card.to_int() for card in deck1.cards]

        # Shuffle with different seeds
        deck1.shuffle(seed=42)
        deck2.shuffle(seed=123)

        shuffled_order1 = [card.to_int() for card in deck1.cards]
        shuffled_order2 = [card.to_int() for card in deck2.cards]

        # Orders should be different
        assert shuffled_order1 != original_order
        assert shuffled_order2 != original_order
        assert shuffled_order1 != shuffled_order2

    def test_shuffle_with_same_seed_produces_same_order(self):
        """Test that shuffling with the same seed produces the same order."""
        deck1 = Deck()
        deck2 = Deck()

        deck1.shuffle(seed=42)
        deck2.shuffle(seed=42)

        order1 = [card.to_int() for card in deck1.cards]
        order2 = [card.to_int() for card in deck2.cards]

        assert order1 == order2

    def test_shuffle_preserves_all_cards(self):
        """Test that shuffling doesn't lose or duplicate cards."""
        deck = Deck()
        original_cards = set(deck.cards)

        deck.shuffle()
        shuffled_cards = set(deck.cards)

        assert original_cards == shuffled_cards
        assert len(deck.cards) == 52


class TestDeckDeal:
    """Test dealing cards to players."""

    def test_deal_to_4_players(self):
        """Test dealing 52 cards to 4 players gives 13 cards each."""
        deck = Deck()
        hands = deck.deal(num_players=4)

        assert len(hands) == 4
        for hand in hands:
            assert len(hand) == 13

    def test_deal_distributes_all_cards(self):
        """Test that dealing gives out all cards in the deck."""
        deck = Deck()
        hands = deck.deal(num_players=4)

        # Collect all dealt cards
        all_dealt_cards = []
        for hand in hands:
            all_dealt_cards.extend(hand)

        # Should have all 52 cards
        assert len(all_dealt_cards) == 52

        # Should be unique
        assert len(set(all_dealt_cards)) == 52

    def test_deal_sorts_hands(self):
        """Test that dealt hands are sorted."""
        deck = Deck()
        deck.shuffle(seed=42)
        hands = deck.deal(num_players=4)

        for hand in hands:
            # Check that hand is sorted
            sorted_hand = sorted(hand)
            assert hand == sorted_hand

    def test_deal_with_different_shuffles_gives_different_hands(self):
        """Test that different shuffles produce different deals."""
        deck1 = Deck()
        deck1.shuffle(seed=42)
        hands1 = deck1.deal(num_players=4)

        deck2 = Deck()
        deck2.shuffle(seed=123)
        hands2 = deck2.deal(num_players=4)

        # At least one player should have a different hand
        different = False
        for i in range(4):
            if set(hands1[i]) != set(hands2[i]):
                different = True
                break

        assert different

    def test_deal_with_invalid_num_players(self):
        """Test that dealing to an invalid number of players raises error."""
        deck = Deck()

        # 52 cards cannot be evenly divided by 3 or 5
        with pytest.raises(ValueError):
            deck.deal(num_players=3)

        with pytest.raises(ValueError):
            deck.deal(num_players=5)

    def test_deal_to_2_players(self):
        """Test dealing to 2 players works correctly."""
        deck = Deck()
        hands = deck.deal(num_players=2)

        assert len(hands) == 2
        for hand in hands:
            assert len(hand) == 26


class TestDeckReset:
    """Test deck reset functionality."""

    def test_reset_restores_full_deck(self):
        """Test that reset restores the deck to 52 cards."""
        deck = Deck()
        deck.shuffle()
        deck.deal(num_players=4)

        deck.reset()

        assert len(deck) == 52

    def test_reset_gives_unshuffled_deck(self):
        """Test that reset gives back the original card order."""
        deck1 = Deck()
        original_order = [card.to_int() for card in deck1.cards]

        deck1.shuffle(seed=42)
        deck1.reset()

        reset_order = [card.to_int() for card in deck1.cards]

        assert reset_order == original_order


class TestDeckIntegration:
    """Integration tests for full deck workflow."""

    def test_full_game_workflow(self):
        """Test a complete shuffle-deal-reset cycle."""
        deck = Deck()

        # Start with 52 cards
        assert len(deck) == 52

        # Shuffle
        deck.shuffle(seed=42)
        assert len(deck) == 52

        # Deal to 4 players
        hands = deck.deal(num_players=4)
        assert len(hands) == 4

        # Verify all cards are accounted for
        all_cards = []
        for hand in hands:
            all_cards.extend(hand)
        assert len(set(all_cards)) == 52

        # Reset and verify
        deck.reset()
        assert len(deck) == 52

    def test_reproducible_deal_with_seed(self):
        """Test that using the same seed produces the same deal."""
        deck1 = Deck()
        deck1.shuffle(seed=12345)
        hands1 = deck1.deal(num_players=4)

        deck2 = Deck()
        deck2.shuffle(seed=12345)
        hands2 = deck2.deal(num_players=4)

        # Each player should have the exact same cards
        for i in range(4):
            assert set(hands1[i]) == set(hands2[i])
