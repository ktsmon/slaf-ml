"""
Deck management for the Slave card game.

Handles shuffling and dealing cards to players.
"""

import random
from typing import Optional
from game.card import Card, create_deck


class Deck:
    """
    Represents a deck of 52 playing cards.

    Provides methods for shuffling and dealing cards to players.
    """

    def __init__(self):
        """Initialize a standard 52-card deck."""
        self.cards = create_deck()

    def shuffle(self, seed: Optional[int] = None) -> None:
        """
        Shuffle the deck.

        Args:
            seed: Optional random seed for reproducible shuffles
        """
        if seed is not None:
            random.seed(seed)
        random.shuffle(self.cards)

    def deal(self, num_players: int = 4) -> list[list[Card]]:
        """
        Deal cards evenly to players.

        Args:
            num_players: Number of players to deal to (default: 4)

        Returns:
            List of hands, where each hand is a list of Cards

        Raises:
            ValueError: If the deck size is not evenly divisible by num_players
        """
        if len(self.cards) % num_players != 0:
            raise ValueError(
                f"Cannot deal {len(self.cards)} cards evenly to {num_players} players"
            )

        cards_per_player = len(self.cards) // num_players
        hands = []

        for i in range(num_players):
            start_idx = i * cards_per_player
            end_idx = start_idx + cards_per_player
            hand = self.cards[start_idx:end_idx]
            # Sort each hand by card value for convenience
            hand.sort()
            hands.append(hand)

        return hands

    def reset(self) -> None:
        """Reset the deck to a full, unshuffled state."""
        self.cards = create_deck()

    def __len__(self) -> int:
        """Return the number of cards remaining in the deck."""
        return len(self.cards)

    def __repr__(self) -> str:
        """Return a string representation of the deck."""
        return f"Deck({len(self.cards)} cards)"
