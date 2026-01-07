"""
Card representation for the Slave card game.

In this game:
- Rank ordering: 2 (highest) > A > K > Q > J > 10 > 9 > 8 > 7 > 6 > 5 > 4 > 3 (lowest)
- Suit ordering: Spades (highest) > Hearts > Diamonds > Clubs (lowest)
- Cards are compared first by rank, then by suit if ranks are equal
"""

from enum import IntEnum
from typing import Optional


class Rank(IntEnum):
    """Card ranks with their ordering values."""
    THREE = 3
    FOUR = 4
    FIVE = 5
    SIX = 6
    SEVEN = 7
    EIGHT = 8
    NINE = 9
    TEN = 10
    JACK = 11
    QUEEN = 12
    KING = 13
    ACE = 14
    TWO = 15  # Highest rank in Slave game


class Suit(IntEnum):
    """Card suits with their ordering values."""
    CLUBS = 1  # Lowest
    DIAMONDS = 2
    HEARTS = 3
    SPADES = 4  # Highest


class Card:
    """
    Represents a single playing card.

    Attributes:
        rank: The card's rank (3-2, where 2 is highest)
        suit: The card's suit (♦♣♥♠)
    """

    # Unicode symbols for suits
    SUIT_SYMBOLS = {
        Suit.DIAMONDS: '♦',
        Suit.CLUBS: '♣',
        Suit.HEARTS: '♥',
        Suit.SPADES: '♠'
    }

    # Rank symbols for display
    RANK_SYMBOLS = {
        Rank.THREE: '3', Rank.FOUR: '4', Rank.FIVE: '5',
        Rank.SIX: '6', Rank.SEVEN: '7', Rank.EIGHT: '8',
        Rank.NINE: '9', Rank.TEN: '10', Rank.JACK: 'J',
        Rank.QUEEN: 'Q', Rank.KING: 'K', Rank.ACE: 'A',
        Rank.TWO: '2'
    }

    def __init__(self, rank: Rank, suit: Suit):
        """
        Initialize a card.

        Args:
            rank: The card's rank
            suit: The card's suit
        """
        self.rank = rank
        self.suit = suit

    def __lt__(self, other: 'Card') -> bool:
        """Compare if this card is less than another (rank first, then suit)."""
        if self.rank != other.rank:
            return self.rank < other.rank
        return self.suit < other.suit

    def __gt__(self, other: 'Card') -> bool:
        """Compare if this card is greater than another (rank first, then suit)."""
        if self.rank != other.rank:
            return self.rank > other.rank
        return self.suit > other.suit

    def __eq__(self, other: object) -> bool:
        """Check if two cards are equal."""
        if not isinstance(other, Card):
            return False
        return self.rank == other.rank and self.suit == other.suit

    def __le__(self, other: 'Card') -> bool:
        """Compare if this card is less than or equal to another."""
        return self < other or self == other

    def __ge__(self, other: 'Card') -> bool:
        """Compare if this card is greater than or equal to another."""
        return self > other or self == other

    def __hash__(self) -> int:
        """Make cards hashable so they can be used in sets and as dict keys."""
        return hash((self.rank, self.suit))

    def to_int(self) -> int:
        """
        Convert card to unique integer ID (0-51).

        Encoding: (rank_value - 3) * 4 + (suit_value - 1)
        This ensures 3♣ = 0 and 2♠ = 51

        Returns:
            Integer from 0 to 51
        """
        rank_index = self.rank - 3  # 3->0, 4->1, ..., 2->12
        suit_index = self.suit - 1  # CLUBS->0, DIAMONDS->1, HEARTS->2, SPADES->3
        return rank_index * 4 + suit_index

    @staticmethod
    def from_int(card_id: int) -> 'Card':
        """
        Create a card from an integer ID (0-51).

        Args:
            card_id: Integer from 0 to 51

        Returns:
            Card object

        Raises:
            ValueError: If card_id is not in range [0, 51]
        """
        if not 0 <= card_id <= 51:
            raise ValueError(f"Card ID must be between 0 and 51, got {card_id}")

        suit_index = card_id % 4
        rank_index = card_id // 4

        rank = Rank(rank_index + 3)
        suit = Suit(suit_index + 1)

        return Card(rank, suit)

    def __repr__(self) -> str:
        """Return a string representation of the card."""
        rank_str = self.RANK_SYMBOLS[self.rank]
        suit_str = self.SUIT_SYMBOLS[self.suit]
        return f"{rank_str}{suit_str}"

    def __str__(self) -> str:
        """Return a string representation of the card."""
        return self.__repr__()


def create_deck() -> list[Card]:
    """
    Create a standard 52-card deck.

    Returns:
        List of all 52 cards
    """
    deck = []
    for rank in Rank:
        for suit in Suit:
            deck.append(Card(rank, suit))
    return deck
