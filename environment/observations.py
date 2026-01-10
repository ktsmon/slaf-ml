"""
Observation space encoding for the Slave card game.

This module converts game state into neural network input features.
"""

from typing import Optional
import numpy as np

from game.card import Card, Rank, Suit
from game.game_state import GameState, Position
from game.rules import Play, PlayType


def encode_observation(game_state: GameState, player_id: int) -> np.ndarray:
    """
    Encode game state as observation vector for a specific player.

    Observation components:
    1. Own hand (52 binary) - which cards the player holds
    2. Played cards (52 binary) - which cards are out of play
    3. Last play encoding:
       - Play type (5 one-hot)
       - Card ranks in play (13 features)
       - Leading suit (4 one-hot)
    4. Game context:
       - Current positions (4x4 one-hot: K/Q/C/S for each player)
       - Cards remaining per player (4 integers normalized)
       - Round number (1 integer)
       - Current player indicator (4 binary)
       - Player's own position (4 one-hot)

    Total: 52 + 52 + 5 + 13 + 4 + 16 + 4 + 1 + 4 + 4 = 155 features

    Args:
        game_state: Current game state
        player_id: ID of the player to encode observation for

    Returns:
        Observation vector as numpy array
    """
    features = []

    # 1. Own hand (52 binary features)
    own_hand = game_state.get_hand(player_id)
    hand_encoding = np.zeros(52, dtype=np.float32)
    for card in own_hand:
        hand_encoding[card.to_int()] = 1.0
    features.append(hand_encoding)

    # 2. Played cards (52 binary features)
    played_cards = np.zeros(52, dtype=np.float32)
    # Cards that have been played are those not in any player's hand
    all_hands_cards = set()
    for pid in range(game_state.num_players):
        all_hands_cards.update(game_state.get_hand(pid))

    # All 52 cards
    from game.deck import Deck
    all_cards = Deck().cards
    for card in all_cards:
        if card not in all_hands_cards:
            played_cards[card.to_int()] = 1.0
    features.append(played_cards)

    # 3. Last play encoding
    last_play_type = np.zeros(5, dtype=np.float32)  # 5 play types
    last_play_ranks = np.zeros(13, dtype=np.float32)  # 13 ranks
    last_play_suit = np.zeros(4, dtype=np.float32)  # 4 suits

    if game_state.last_play is not None:
        play = game_state.last_play
        if len(play.cards) > 0:
            # Play type one-hot
            play_type_map = {
                PlayType.SINGLE: 0,
                PlayType.PAIR: 1,
                PlayType.THREE_OF_KIND: 2,
                PlayType.FOUR_OF_KIND: 3,
                PlayType.PASS: 4
            }
            last_play_type[play_type_map[play.play_type]] = 1.0

            # Ranks in play
            for card in play.cards:
                rank_idx = card.rank.value - 3  # Rank 3 -> index 0
                last_play_ranks[rank_idx] = 1.0

            # Leading suit (highest card's suit)
            leading_card = max(play.cards)
            suit_map = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
            last_play_suit[suit_map[leading_card.suit]] = 1.0
        else:
            # Pass
            last_play_type[4] = 1.0
    else:
        # No last play (start of trick)
        last_play_type[4] = 1.0

    features.append(last_play_type)
    features.append(last_play_ranks)
    features.append(last_play_suit)

    # 4. Game context
    # Positions for all players (4x4 one-hot)
    positions_encoding = np.zeros(16, dtype=np.float32)
    position_map = {
        Position.KING: 0,
        Position.QUEEN: 1,
        Position.COMMONER: 2,
        Position.SLAVE: 3,
        None: 0  # First round - no position yet, default to King encoding
    }

    for pid in range(game_state.num_players):
        pos = game_state.get_position(pid)
        pos_idx = position_map[pos]
        positions_encoding[pid * 4 + pos_idx] = 1.0

    features.append(positions_encoding)

    # Cards remaining per player (4 integers normalized to 0-1)
    cards_remaining = np.zeros(4, dtype=np.float32)
    for pid in range(game_state.num_players):
        num_cards = len(game_state.get_hand(pid))
        cards_remaining[pid] = num_cards / 13.0  # Normalize by max cards
    features.append(cards_remaining)

    # Round number (normalized)
    round_number = np.array([game_state.round_number / 10.0], dtype=np.float32)
    features.append(round_number)

    # Current player indicator (4 binary)
    current_player_indicator = np.zeros(4, dtype=np.float32)
    current_player_indicator[game_state.current_player] = 1.0
    features.append(current_player_indicator)

    # Player's own position (4 one-hot)
    own_position = np.zeros(4, dtype=np.float32)
    own_pos = game_state.get_position(player_id)
    own_position[position_map[own_pos]] = 1.0
    features.append(own_position)

    # Concatenate all features
    observation = np.concatenate(features)

    return observation


def get_action_mask(game_state: GameState, player_id: int, action_space_size: int = 194) -> np.ndarray:
    """
    Get mask of valid actions for a player.

    This prevents the agent from selecting invalid actions.

    Args:
        game_state: Current game state
        player_id: ID of the player
        action_space_size: Total number of possible actions

    Returns:
        Binary mask where 1 = valid action, 0 = invalid action
    """
    mask = np.zeros(action_space_size, dtype=np.float32)

    # Get valid plays from game state
    valid_plays = game_state.get_valid_plays(player_id)

    # Encode each valid play as an action ID and set mask to 1
    for play in valid_plays:
        action_id = encode_action(play)
        if action_id < action_space_size:
            mask[action_id] = 1.0

    return mask


def encode_action(cards: list[Card]) -> int:
    """
    Encode a card play as an action ID.

    Action space structure:
    - 0: Pass
    - 1-52: Single cards (card_id + 1)
    - 53-130: Pairs (78 possible pairs)
    - 131-143: Three-of-a-kinds (13 ranks)
    - 144-156: Four-of-a-kinds (13 ranks)

    Total: 157 actions

    Args:
        cards: List of cards to play

    Returns:
        Action ID (integer)
    """
    if len(cards) == 0:
        return 0  # Pass

    if len(cards) == 1:
        # Single card: action_id = card.to_int() + 1
        return cards[0].to_int() + 1

    if len(cards) == 2:
        # Pair: encode as 53 + pair_id
        # Pair ID based on rank (0-12) and suits combination
        cards_sorted = sorted(cards)
        rank = cards_sorted[0].rank.value
        pair_id = (rank - 3) * 6 + _get_pair_suit_combination(cards_sorted)
        return 53 + pair_id

    if len(cards) == 3:
        # Three-of-a-kind: encode as 131 + rank_id
        rank = cards[0].rank.value
        rank_id = rank - 3
        return 131 + rank_id

    if len(cards) == 4:
        # Four-of-a-kind: encode as 144 + rank_id
        rank = cards[0].rank.value
        rank_id = rank - 3
        return 144 + rank_id

    # Default fallback for unhandled combinations
    return 0


def decode_action(action_id: int, hand: list[Card]) -> list[Card]:
    """
    Decode an action ID to a list of cards to play.

    Args:
        action_id: Action ID to decode
        hand: Player's current hand

    Returns:
        List of cards to play
    """
    if action_id == 0:
        return []  # Pass

    if 1 <= action_id <= 52:
        # Single card
        card_id = action_id - 1
        for card in hand:
            if card.to_int() == card_id:
                return [card]
        return []  # Card not in hand

    if 53 <= action_id <= 130:
        # Pair
        pair_id = action_id - 53
        rank_value = (pair_id // 6) + 3
        rank = Rank(rank_value)

        # Find two cards of this rank in hand
        cards_of_rank = [card for card in hand if card.rank == rank]
        if len(cards_of_rank) >= 2:
            # Return highest two cards by suit
            return sorted(cards_of_rank, reverse=True)[:2]
        return []

    if 131 <= action_id <= 143:
        # Three-of-a-kind
        rank_value = (action_id - 131) + 3
        rank = Rank(rank_value)

        # Find three cards of this rank
        cards_of_rank = [card for card in hand if card.rank == rank]
        if len(cards_of_rank) >= 3:
            # Return highest three cards by suit
            return sorted(cards_of_rank, reverse=True)[:3]
        return []

    if 144 <= action_id <= 156:
        # Four-of-a-kind
        rank_value = (action_id - 144) + 3
        rank = Rank(rank_value)

        # Find four cards of this rank
        cards_of_rank = [card for card in hand if card.rank == rank]
        if len(cards_of_rank) == 4:
            return cards_of_rank
        return []

    return []


def _get_pair_suit_combination(cards: list[Card]) -> int:
    """
    Get suit combination index for a pair.

    There are 6 possible combinations for choosing 2 suits from 4:
    (C,D)=0, (C,H)=1, (C,S)=2, (D,H)=3, (D,S)=4, (H,S)=5

    Args:
        cards: Two cards forming a pair

    Returns:
        Combination index (0-5)
    """
    suits = sorted([card.suit for card in cards])
    suit_pairs = [
        (Suit.CLUBS, Suit.DIAMONDS),
        (Suit.CLUBS, Suit.HEARTS),
        (Suit.CLUBS, Suit.SPADES),
        (Suit.DIAMONDS, Suit.HEARTS),
        (Suit.DIAMONDS, Suit.SPADES),
        (Suit.HEARTS, Suit.SPADES)
    ]

    for idx, pair in enumerate(suit_pairs):
        if suits[0] == pair[0] and suits[1] == pair[1]:
            return idx

    return 0


