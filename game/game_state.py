"""
Game state management for the Slave card game.

Manages the full game including:
- Player hands and positions
- Turn order and trick management
- Card exchanges between rounds
- Position determination (King/Queen/Commoner/Slave)
"""

from typing import Optional
from enum import Enum
from game.card import Card, Rank, Suit
from game.deck import Deck
from game.rules import Play, PlayType, can_beat, get_valid_plays, has_three_of_diamonds


class Position(Enum):
    """Player positions in the Slave card game."""
    KING = "King"
    QUEEN = "Queen"
    COMMONER = "Commoner"
    SLAVE = "Slave"


class GameState:
    """
    Manages the complete game state for a Slave card game.

    Attributes:
        num_players: Number of players (default: 4)
        hands: Dictionary mapping player ID to their hand
        positions: Dictionary mapping player ID to their position
        finished_order: List of player IDs in order they finished
        current_player: ID of the current player
        last_play: The most recent play made
        trick_plays: List of plays in the current trick
        trick_leader: Player who started the current trick
        round_number: Current round number (0-indexed)
        is_first_round: Whether this is the first round
    """

    def __init__(self, num_players: int = 4):
        """
        Initialize a new game.

        Args:
            num_players: Number of players (default: 4)
        """
        self.num_players = num_players
        self.hands: dict[int, list[Card]] = {}
        self.positions: dict[int, Optional[Position]] = {}
        self.finished_order: list[int] = []
        self.current_player: int = 0
        self.last_play: Optional[Play] = None
        self.trick_plays: list[Play] = []
        self.trick_leader: int = 0
        self.round_number: int = 0
        self.is_first_round: bool = True
        self.game_over: bool = False

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the game to initial state and deal new hands.

        Args:
            seed: Optional seed for reproducible games
        """
        # Create and shuffle deck
        deck = Deck()
        deck.shuffle(seed=seed)

        # Deal cards
        hands = deck.deal(self.num_players)
        self.hands = {i: hands[i] for i in range(self.num_players)}

        # Reset game state
        self.positions = {i: None for i in range(self.num_players)}
        self.finished_order = []
        self.last_play = None
        self.trick_plays = []
        self.round_number = 0
        self.is_first_round = True
        self.game_over = False

        # Find who has 3 of diamonds and set as starting player
        for player_id, hand in self.hands.items():
            if has_three_of_diamonds(hand):
                self.current_player = player_id
                self.trick_leader = player_id
                break

    def start_new_round(self, king_gives: Optional[list[Card]] = None,
                        queen_gives: Optional[Card] = None) -> None:
        """
        Start a new round with card exchanges.

        This should be called after the first round completes.

        Args:
            king_gives: Optional list of 2 cards for King to give. If None, gives lowest 2 cards.
            queen_gives: Optional card for Queen to give. If None, gives lowest card.
        """
        if self.is_first_round:
            self.is_first_round = False

        # Create and shuffle deck
        deck = Deck()
        deck.shuffle()

        # Deal new hands
        hands = deck.deal(self.num_players)
        self.hands = {i: hands[i] for i in range(self.num_players)}

        # Perform card exchanges based on positions
        self._exchange_cards(king_gives=king_gives, queen_gives=queen_gives)

        # Slave starts the round
        for player_id, position in self.positions.items():
            if position == Position.SLAVE:
                self.current_player = player_id
                self.trick_leader = player_id
                break

        # Reset round state
        self.finished_order = []
        self.last_play = None
        self.trick_plays = []
        self.round_number += 1
        self.game_over = False

    def _exchange_cards(self, king_gives: Optional[list[Card]] = None,
                        queen_gives: Optional[Card] = None) -> None:
        """
        Exchange cards between King/Slave and Queen/Commoner.

        King and Slave exchange 2 cards.
        Queen and Commoner exchange 1 card.

        Args:
            king_gives: Optional list of 2 cards for King to give. If None, gives lowest 2 cards.
            queen_gives: Optional card for Queen to give. If None, gives lowest card.
        """
        # Find players by position
        king_id = None
        queen_id = None
        commoner_id = None
        slave_id = None

        for player_id, position in self.positions.items():
            if position == Position.KING:
                king_id = player_id
            elif position == Position.QUEEN:
                queen_id = player_id
            elif position == Position.COMMONER:
                commoner_id = player_id
            elif position == Position.SLAVE:
                slave_id = player_id

        # Exchange between King and Slave (2 cards)
        if king_id is not None and slave_id is not None:
            # Slave gives 2 best cards to King (mandatory)
            slave_hand = sorted(self.hands[slave_id], reverse=True)
            slave_gives = slave_hand[:2]

            # King chooses which 2 cards to give to Slave
            if king_gives is None:
                # Default: give 2 lowest cards
                king_hand = sorted(self.hands[king_id])
                king_gives_cards = king_hand[:2]
            else:
                # Validate that King has these cards
                king_gives_cards = king_gives
                for card in king_gives_cards:
                    if card not in self.hands[king_id]:
                        # Invalid cards, fallback to lowest
                        king_hand = sorted(self.hands[king_id])
                        king_gives_cards = king_hand[:2]
                        break

            # Perform exchange
            for card in slave_gives:
                self.hands[slave_id].remove(card)
                self.hands[king_id].append(card)

            for card in king_gives_cards:
                self.hands[king_id].remove(card)
                self.hands[slave_id].append(card)

            # Re-sort hands
            self.hands[king_id].sort()
            self.hands[slave_id].sort()

        # Exchange between Queen and Commoner (1 card)
        if queen_id is not None and commoner_id is not None:
            # Commoner gives 1 best card to Queen (mandatory)
            commoner_hand = sorted(self.hands[commoner_id], reverse=True)
            commoner_gives_card = commoner_hand[0]

            # Queen chooses which card to give to Commoner
            if queen_gives is None:
                # Default: give lowest card
                queen_hand = sorted(self.hands[queen_id])
                queen_gives_card = queen_hand[0]
            else:
                # Validate that Queen has this card
                if queen_gives not in self.hands[queen_id]:
                    # Invalid card, fallback to lowest
                    queen_hand = sorted(self.hands[queen_id])
                    queen_gives_card = queen_hand[0]
                else:
                    queen_gives_card = queen_gives

            # Perform exchange
            self.hands[commoner_id].remove(commoner_gives_card)
            self.hands[queen_id].append(commoner_gives_card)

            self.hands[queen_id].remove(queen_gives_card)
            self.hands[commoner_id].append(queen_gives_card)

            # Re-sort hands
            self.hands[queen_id].sort()
            self.hands[commoner_id].sort()

    def play_cards(self, player_id: int, cards: list[Card]) -> bool:
        """
        Attempt to play cards for a player.

        Args:
            player_id: ID of the player making the play
            cards: List of cards to play (empty for pass)

        Returns:
            True if the play was successful, False otherwise
        """
        if player_id != self.current_player:
            return False

        if self.game_over:
            return False

        # Check if player still has cards
        if len(self.hands[player_id]) == 0 and len(cards) > 0:
            return False

        # Verify player has all the cards they're trying to play
        for card in cards:
            if card not in self.hands[player_id]:
                return False

        # Create the play
        play = Play(player_id=player_id, cards=cards)

        # Check if play is valid and can beat last play
        if not can_beat(play, self.last_play):
            return False

        # Valid play - execute it
        if play.play_type != PlayType.PASS:
            # Remove cards from hand
            for card in cards:
                self.hands[player_id].remove(card)

            # Update last play
            self.last_play = play

            # Check if player finished
            if len(self.hands[player_id]) == 0:
                self.finished_order.append(player_id)

                # Check for King demotion rule
                if not self.is_first_round:
                    self._check_king_demotion()

                # Check if game is over (all but one player finished)
                if len(self.finished_order) >= self.num_players - 1:
                    self._end_round()
                    return True

        # Add play to trick
        self.trick_plays.append(play)

        # Check if trick is complete (all active players have played)
        num_active_players = self.num_players - len(self.finished_order)
        if len(self.trick_plays) >= num_active_players:
            self._end_trick()
        else:
            # Move to next player who hasn't finished
            self._next_player()

        return True

    def _next_player(self) -> None:
        """Move to the next player who hasn't finished."""
        next_player = (self.current_player + 1) % self.num_players
        while next_player in self.finished_order:
            next_player = (next_player + 1) % self.num_players
        self.current_player = next_player

    def _end_trick(self) -> None:
        """
        End the current trick and determine the winner.

        The winner of the trick leads the next trick.
        """
        # Find who played the highest non-pass play
        best_play = None
        winner = None

        for play in self.trick_plays:
            if play.play_type == PlayType.PASS:
                continue

            if best_play is None:
                best_play = play
                winner = play.player_id
            elif can_beat(play, best_play):
                best_play = play
                winner = play.player_id

        # Clear trick state
        self.trick_plays = []
        self.last_play = None

        # Winner leads next trick (if they haven't finished)
        if winner is not None and winner not in self.finished_order:
            self.current_player = winner
            self.trick_leader = winner
        else:
            # If winner has finished, next active player leads
            self._next_player()
            self.trick_leader = self.current_player

    def _check_king_demotion(self) -> None:
        """
        Check if King demotion rule applies.

        If a non-King finishes first, the previous King becomes the Slave.
        """
        if len(self.finished_order) != 1:
            return

        first_finisher = self.finished_order[0]

        # Find previous King
        previous_king = None
        for player_id, position in self.positions.items():
            if position == Position.KING:
                previous_king = player_id
                break

        # If first finisher is not the King, demote the King
        if previous_king is not None and first_finisher != previous_king:
            # King becomes Slave in next round
            # (This will be handled in position determination)
            pass

    def _end_round(self) -> None:
        """
        End the round and determine final positions.
        """
        # Add the last remaining player to finished order
        for player_id in range(self.num_players):
            if player_id not in self.finished_order:
                self.finished_order.append(player_id)
                break

        # Determine positions based on finish order
        self._determine_positions()

        self.game_over = True

    def _determine_positions(self) -> None:
        """
        Determine player positions based on finish order.

        1st place = King
        2nd place = Queen
        3rd place = Commoner
        4th place = Slave

        Special rule: If a non-King finishes first, the previous King becomes Slave.
        """
        if len(self.finished_order) != self.num_players:
            return

        # Check for King demotion
        if not self.is_first_round:
            first_finisher = self.finished_order[0]
            previous_king = None

            for player_id, position in self.positions.items():
                if position == Position.KING:
                    previous_king = player_id
                    break

            if previous_king is not None and first_finisher != previous_king:
                # Apply King demotion: previous King becomes Slave
                self.positions[first_finisher] = Position.KING
                self.positions[previous_king] = Position.SLAVE

                # Assign other positions
                remaining = [p for p in self.finished_order if p not in [first_finisher, previous_king]]
                if len(remaining) >= 1:
                    self.positions[remaining[0]] = Position.QUEEN
                if len(remaining) >= 2:
                    self.positions[remaining[1]] = Position.COMMONER

                return

        # Normal position assignment
        positions = [Position.KING, Position.QUEEN, Position.COMMONER, Position.SLAVE]
        for i, player_id in enumerate(self.finished_order):
            if i < len(positions):
                self.positions[player_id] = positions[i]

    def get_valid_plays(self, player_id: int) -> list[list[Card]]:
        """
        Get all valid plays for a player.

        Args:
            player_id: ID of the player

        Returns:
            List of valid card combinations
        """
        if player_id != self.current_player:
            return []

        if self.game_over:
            return []

        hand = self.hands.get(player_id, [])
        return get_valid_plays(hand, self.last_play)

    def is_game_over(self) -> bool:
        """Check if the game is over."""
        return self.game_over

    def get_hand(self, player_id: int) -> list[Card]:
        """Get a player's hand."""
        return self.hands.get(player_id, []).copy()

    def get_position(self, player_id: int) -> Optional[Position]:
        """Get a player's position."""
        return self.positions.get(player_id)

    def get_cards_to_receive_from_slave(self) -> Optional[list[Card]]:
        """
        Get the 2 cards that the King will receive from the Slave.

        This is useful for strategic decision-making during card exchange.
        Should be called after dealing but before exchange.

        Returns:
            List of 2 cards (the Slave's best cards), or None if not applicable
        """
        # Find King and Slave
        slave_id = None
        for player_id, position in self.positions.items():
            if position == Position.SLAVE:
                slave_id = player_id
                break

        if slave_id is None or slave_id not in self.hands:
            return None

        # Return the 2 best cards from Slave's hand
        slave_hand = sorted(self.hands[slave_id], reverse=True)
        return slave_hand[:2]

    def get_cards_to_receive_from_commoner(self) -> Optional[Card]:
        """
        Get the card that the Queen will receive from the Commoner.

        This is useful for strategic decision-making during card exchange.
        Should be called after dealing but before exchange.

        Returns:
            The card (Commoner's best card), or None if not applicable
        """
        # Find Commoner
        commoner_id = None
        for player_id, position in self.positions.items():
            if position == Position.COMMONER:
                commoner_id = player_id
                break

        if commoner_id is None or commoner_id not in self.hands:
            return None

        # Return the best card from Commoner's hand
        commoner_hand = sorted(self.hands[commoner_id], reverse=True)
        return commoner_hand[0]

    def __repr__(self) -> str:
        """Return string representation of game state."""
        return (f"GameState(round={self.round_number}, "
                f"current_player={self.current_player}, "
                f"finished={len(self.finished_order)}/{self.num_players})")
