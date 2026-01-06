"""
Game rules and play validation for the Slave card game.

This module implements the complete rule set including:
- Play type detection (single, pair, straight, four-of-a-kind)
- Play validation
- Play comparison
- Valid action generation
"""

from enum import Enum
from typing import Optional
from game.card import Card, Rank


class PlayType(Enum):
    """Types of plays in the Slave card game."""
    PASS = "pass"
    SINGLE = "single"
    PAIR = "pair"
    STRAIGHT = "straight"
    FOUR_OF_KIND = "four_of_kind"


class Play:
    """
    Represents a play made by a player.

    Attributes:
        player_id: ID of the player who made this play
        cards: List of cards played (empty for PASS)
        play_type: Type of play
    """

    def __init__(self, player_id: int, cards: list[Card]):
        """
        Initialize a play.

        Args:
            player_id: ID of the player making the play
            cards: List of cards in the play (empty for pass)
        """
        self.player_id = player_id
        self.cards = sorted(cards)  # Always keep cards sorted
        self.play_type = self._determine_play_type()

    def _determine_play_type(self) -> PlayType:
        """Determine the type of play based on the cards."""
        if len(self.cards) == 0:
            return PlayType.PASS
        elif len(self.cards) == 1:
            return PlayType.SINGLE
        elif len(self.cards) == 2:
            return PlayType.PAIR
        elif len(self.cards) == 4:
            return PlayType.FOUR_OF_KIND
        elif len(self.cards) >= 3:
            return PlayType.STRAIGHT
        else:
            raise ValueError(f"Invalid number of cards: {len(self.cards)}")

    def is_valid(self) -> bool:
        """
        Check if this play is valid according to game rules.

        Returns:
            True if the play is valid, False otherwise
        """
        if self.play_type == PlayType.PASS:
            return True

        if self.play_type == PlayType.SINGLE:
            return True

        if self.play_type == PlayType.PAIR:
            # Must have exactly 2 cards of the same rank
            return len(self.cards) == 2 and self.cards[0].rank == self.cards[1].rank

        if self.play_type == PlayType.FOUR_OF_KIND:
            # Must have exactly 4 cards of the same rank
            if len(self.cards) != 4:
                return False
            first_rank = self.cards[0].rank
            return all(card.rank == first_rank for card in self.cards)

        if self.play_type == PlayType.STRAIGHT:
            # Must have at least 3 consecutive cards
            if len(self.cards) < 3:
                return False

            # Check if cards are consecutive
            ranks = [card.rank for card in self.cards]
            for i in range(len(ranks) - 1):
                # Special case: 2 cannot be in the middle of a straight
                if ranks[i] == Rank.TWO:
                    return False
                # Check consecutive
                if ranks[i + 1] != ranks[i] + 1:
                    return False

            return True

        return False

    def get_highest_card(self) -> Optional[Card]:
        """
        Get the highest card in this play.

        Returns:
            The highest card, or None if this is a pass
        """
        if self.play_type == PlayType.PASS or len(self.cards) == 0:
            return None
        return max(self.cards)

    def __repr__(self) -> str:
        """Return string representation of the play."""
        if self.play_type == PlayType.PASS:
            return f"Play(player={self.player_id}, PASS)"
        cards_str = ", ".join(str(card) for card in self.cards)
        return f"Play(player={self.player_id}, {self.play_type.value}, [{cards_str}])"


def can_beat(play: Play, last_play: Optional[Play]) -> bool:
    """
    Check if a play can beat the last play.

    Args:
        play: The play to check
        last_play: The previous play to beat (None if starting a new trick)

    Returns:
        True if the play is valid and beats the last play
    """
    # Pass is always valid
    if play.play_type == PlayType.PASS:
        return True

    # If no last play, any valid play is allowed
    if last_play is None or last_play.play_type == PlayType.PASS:
        return play.is_valid()

    # Play must be valid first
    if not play.is_valid():
        return False

    # Four of a kind beats pairs
    if play.play_type == PlayType.FOUR_OF_KIND and last_play.play_type == PlayType.PAIR:
        return True

    # Straights beat singles but lose to pairs
    if play.play_type == PlayType.STRAIGHT and last_play.play_type == PlayType.SINGLE:
        return True
    if play.play_type == PlayType.STRAIGHT and last_play.play_type == PlayType.PAIR:
        return False

    # Otherwise, must be the same type
    if play.play_type != last_play.play_type:
        return False

    # For same type, compare highest cards
    play_high = play.get_highest_card()
    last_high = last_play.get_highest_card()

    if play_high is None or last_high is None:
        return False

    return play_high > last_high


def get_valid_plays(hand: list[Card], last_play: Optional[Play]) -> list[list[Card]]:
    """
    Get all valid plays that can be made from a hand.

    Args:
        hand: The player's current hand
        last_play: The last play made (None if starting a trick)

    Returns:
        List of valid card combinations that can be played
    """
    valid_plays = []

    # Pass is always valid
    valid_plays.append([])

    if len(hand) == 0:
        return valid_plays

    # If no last play, we can play anything valid
    if last_play is None or last_play.play_type == PlayType.PASS:
        # Add all singles
        for card in hand:
            valid_plays.append([card])

        # Add all pairs
        valid_plays.extend(_get_all_pairs(hand))

        # Add all straights
        valid_plays.extend(_get_all_straights(hand))

        # Add all four-of-a-kinds
        valid_plays.extend(_get_all_four_of_kinds(hand))

    else:
        # We need to beat the last play
        last_type = last_play.play_type
        last_high = last_play.get_highest_card()

        if last_type == PlayType.SINGLE:
            # Can play higher singles or straights
            for card in hand:
                if card > last_high:
                    valid_plays.append([card])

            # Can also play straights
            for straight in _get_all_straights(hand):
                valid_plays.append(straight)

        elif last_type == PlayType.PAIR:
            # Can play higher pairs or four-of-a-kinds
            for pair in _get_all_pairs(hand):
                if max(pair) > last_high:
                    valid_plays.append(pair)

            # Can also play four-of-a-kinds
            valid_plays.extend(_get_all_four_of_kinds(hand))

        elif last_type == PlayType.STRAIGHT:
            # Can only play higher straights of the same length
            last_length = len(last_play.cards)
            for straight in _get_all_straights(hand):
                if len(straight) == last_length and max(straight) > last_high:
                    valid_plays.append(straight)

        elif last_type == PlayType.FOUR_OF_KIND:
            # Can only play higher four-of-a-kinds
            for four in _get_all_four_of_kinds(hand):
                if max(four) > last_high:
                    valid_plays.append(four)

    return valid_plays


def _get_all_pairs(hand: list[Card]) -> list[list[Card]]:
    """Get all possible pairs from a hand."""
    pairs = []
    rank_groups = {}

    # Group cards by rank
    for card in hand:
        if card.rank not in rank_groups:
            rank_groups[card.rank] = []
        rank_groups[card.rank].append(card)

    # Find all pairs
    for rank, cards in rank_groups.items():
        if len(cards) >= 2:
            # Take the highest pair (by suit)
            sorted_cards = sorted(cards, reverse=True)
            pairs.append([sorted_cards[0], sorted_cards[1]])

    return pairs


def _get_all_four_of_kinds(hand: list[Card]) -> list[list[Card]]:
    """Get all possible four-of-a-kinds from a hand."""
    fours = []
    rank_groups = {}

    # Group cards by rank
    for card in hand:
        if card.rank not in rank_groups:
            rank_groups[card.rank] = []
        rank_groups[card.rank].append(card)

    # Find all four-of-a-kinds
    for rank, cards in rank_groups.items():
        if len(cards) == 4:
            fours.append(cards)

    return fours


def _get_all_straights(hand: list[Card]) -> list[list[Card]]:
    """Get all possible straights from a hand."""
    straights = []

    if len(hand) < 3:
        return straights

    # Get unique ranks (sorted)
    unique_ranks = sorted(set(card.rank for card in hand))

    # Remove TWO as it cannot be in a straight
    unique_ranks = [r for r in unique_ranks if r != Rank.TWO]

    if len(unique_ranks) < 3:
        return straights

    # Find all consecutive sequences of length 3 or more
    for start_idx in range(len(unique_ranks)):
        for end_idx in range(start_idx + 2, len(unique_ranks)):
            # Check if consecutive
            is_consecutive = True
            for i in range(start_idx, end_idx):
                if unique_ranks[i + 1] != unique_ranks[i] + 1:
                    is_consecutive = False
                    break

            if is_consecutive:
                # Build the straight by taking the highest card of each rank
                straight_cards = []
                for rank in unique_ranks[start_idx:end_idx + 1]:
                    # Get highest card of this rank
                    cards_of_rank = [c for c in hand if c.rank == rank]
                    straight_cards.append(max(cards_of_rank))

                straights.append(straight_cards)

    return straights


def determine_trick_winner(plays: list[Play]) -> int:
    """
    Determine which player won the trick.

    Args:
        plays: List of plays made by each player

    Returns:
        Player ID of the winner
    """
    # Find the highest valid play
    best_play = None
    best_player = -1

    for play in plays:
        if play.play_type == PlayType.PASS:
            continue

        if best_play is None:
            best_play = play
            best_player = play.player_id
        else:
            # Create a new play to test if it beats the current best
            if can_beat(play, best_play):
                best_play = play
                best_player = play.player_id

    return best_player


def has_three_of_diamonds(hand: list[Card]) -> bool:
    """
    Check if a hand contains the 3 of diamonds.

    Args:
        hand: List of cards

    Returns:
        True if the hand contains 3♦
    """
    from game.card import Rank, Suit
    three_diamonds = Card(Rank.THREE, Suit.DIAMONDS)
    return three_diamonds in hand
