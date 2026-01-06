"""
Tests for game rules and play validation.
"""

import pytest
from game.card import Card, Rank, Suit
from game.rules import Play, PlayType, can_beat, get_valid_plays, determine_trick_winner, has_three_of_diamonds


class TestPlayTypeDetection:
    """Test play type detection."""

    def test_pass_detection(self):
        """Test that empty cards is detected as PASS."""
        play = Play(player_id=0, cards=[])
        assert play.play_type == PlayType.PASS

    def test_single_detection(self):
        """Test single card detection."""
        play = Play(player_id=0, cards=[Card(Rank.FIVE, Suit.HEARTS)])
        assert play.play_type == PlayType.SINGLE

    def test_pair_detection(self):
        """Test pair detection."""
        play = Play(player_id=0, cards=[
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES)
        ])
        assert play.play_type == PlayType.PAIR

    def test_straight_detection(self):
        """Test straight detection."""
        play = Play(player_id=0, cards=[
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.FOUR, Suit.SPADES),
            Card(Rank.FIVE, Suit.DIAMONDS)
        ])
        assert play.play_type == PlayType.STRAIGHT

    def test_four_of_kind_detection(self):
        """Test four-of-a-kind detection."""
        play = Play(player_id=0, cards=[
            Card(Rank.SEVEN, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.SPADES),
            Card(Rank.SEVEN, Suit.DIAMONDS),
            Card(Rank.SEVEN, Suit.CLUBS)
        ])
        assert play.play_type == PlayType.FOUR_OF_KIND


class TestPlayValidation:
    """Test play validation logic."""

    def test_pass_is_valid(self):
        """Test that pass is always valid."""
        play = Play(player_id=0, cards=[])
        assert play.is_valid()

    def test_single_is_valid(self):
        """Test that any single card is valid."""
        play = Play(player_id=0, cards=[Card(Rank.ACE, Suit.SPADES)])
        assert play.is_valid()

    def test_valid_pair(self):
        """Test that matching pair is valid."""
        play = Play(player_id=0, cards=[
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.NINE, Suit.CLUBS)
        ])
        assert play.is_valid()

    def test_invalid_pair_different_ranks(self):
        """Test that non-matching cards are not a valid pair."""
        play = Play(player_id=0, cards=[
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.TEN, Suit.CLUBS)
        ])
        assert not play.is_valid()

    def test_valid_straight(self):
        """Test that consecutive cards form valid straight."""
        play = Play(player_id=0, cards=[
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.SEVEN, Suit.DIAMONDS)
        ])
        assert play.is_valid()

    def test_invalid_straight_not_consecutive(self):
        """Test that non-consecutive cards are not a valid straight."""
        play = Play(player_id=0, cards=[
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.SIX, Suit.CLUBS),
            Card(Rank.EIGHT, Suit.DIAMONDS)
        ])
        assert not play.is_valid()

    def test_invalid_straight_with_two(self):
        """Test that straights cannot contain a 2 in the middle."""
        play = Play(player_id=0, cards=[
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.TWO, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS)
        ])
        assert not play.is_valid()

    def test_valid_four_of_kind(self):
        """Test that four matching cards are valid."""
        play = Play(player_id=0, cards=[
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.JACK, Suit.CLUBS),
            Card(Rank.JACK, Suit.DIAMONDS),
            Card(Rank.JACK, Suit.SPADES)
        ])
        assert play.is_valid()

    def test_invalid_four_of_kind_different_ranks(self):
        """Test that four cards with different ranks are not valid."""
        play = Play(player_id=0, cards=[
            Card(Rank.JACK, Suit.HEARTS),
            Card(Rank.JACK, Suit.CLUBS),
            Card(Rank.JACK, Suit.DIAMONDS),
            Card(Rank.QUEEN, Suit.SPADES)
        ])
        assert not play.is_valid()


class TestCanBeat:
    """Test play comparison logic."""

    def test_pass_always_valid(self):
        """Test that pass can always be played."""
        pass_play = Play(player_id=0, cards=[])
        last_play = Play(player_id=1, cards=[Card(Rank.ACE, Suit.SPADES)])
        assert can_beat(pass_play, last_play)

    def test_first_play_any_valid(self):
        """Test that any valid play can start a trick."""
        play = Play(player_id=0, cards=[Card(Rank.THREE, Suit.DIAMONDS)])
        assert can_beat(play, None)

    def test_higher_single_beats_lower_single(self):
        """Test that higher single beats lower single."""
        last_play = Play(player_id=0, cards=[Card(Rank.FIVE, Suit.HEARTS)])
        winning_play = Play(player_id=1, cards=[Card(Rank.KING, Suit.DIAMONDS)])
        losing_play = Play(player_id=1, cards=[Card(Rank.FOUR, Suit.SPADES)])

        assert can_beat(winning_play, last_play)
        assert not can_beat(losing_play, last_play)

    def test_suit_tiebreaker_for_singles(self):
        """Test that suit breaks ties for same rank singles."""
        last_play = Play(player_id=0, cards=[Card(Rank.KING, Suit.DIAMONDS)])
        winning_play = Play(player_id=1, cards=[Card(Rank.KING, Suit.SPADES)])
        losing_play = Play(player_id=1, cards=[Card(Rank.KING, Suit.CLUBS)])

        # Spades > Diamonds, so should win
        assert can_beat(winning_play, last_play)
        # Clubs < Diamonds, so should lose (even though Clubs > Diamonds is wrong, Diamonds is lowest)
        # Actually Clubs > Diamonds, so this should win too
        assert can_beat(losing_play, last_play)

    def test_higher_pair_beats_lower_pair(self):
        """Test that higher pair beats lower pair."""
        last_play = Play(player_id=0, cards=[
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.CLUBS)
        ])
        winning_play = Play(player_id=1, cards=[
            Card(Rank.NINE, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.SPADES)
        ])

        assert can_beat(winning_play, last_play)

    def test_single_cannot_beat_pair(self):
        """Test that a single card cannot beat a pair."""
        last_play = Play(player_id=0, cards=[
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.THREE, Suit.CLUBS)
        ])
        high_single = Play(player_id=1, cards=[Card(Rank.TWO, Suit.SPADES)])

        assert not can_beat(high_single, last_play)

    def test_straight_beats_single(self):
        """Test that a straight beats any single."""
        last_play = Play(player_id=0, cards=[Card(Rank.ACE, Suit.SPADES)])
        straight = Play(player_id=1, cards=[
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.DIAMONDS)
        ])

        assert can_beat(straight, last_play)

    def test_straight_loses_to_pair(self):
        """Test that a straight loses to any pair."""
        last_play = Play(player_id=0, cards=[
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.THREE, Suit.CLUBS)
        ])
        straight = Play(player_id=1, cards=[
            Card(Rank.QUEEN, Suit.HEARTS),
            Card(Rank.KING, Suit.CLUBS),
            Card(Rank.ACE, Suit.DIAMONDS)
        ])

        assert not can_beat(straight, last_play)

    def test_four_of_kind_beats_pair(self):
        """Test that four-of-a-kind beats any pair."""
        last_play = Play(player_id=0, cards=[
            Card(Rank.ACE, Suit.HEARTS),
            Card(Rank.ACE, Suit.CLUBS)
        ])
        four_of_kind = Play(player_id=1, cards=[
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.THREE, Suit.CLUBS),
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.THREE, Suit.SPADES)
        ])

        assert can_beat(four_of_kind, last_play)

    def test_higher_straight_beats_lower_straight(self):
        """Test that higher straight beats lower straight of same length."""
        last_play = Play(player_id=0, cards=[
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.DIAMONDS)
        ])
        winning_play = Play(player_id=1, cards=[
            Card(Rank.NINE, Suit.HEARTS),
            Card(Rank.TEN, Suit.CLUBS),
            Card(Rank.JACK, Suit.DIAMONDS)
        ])

        assert can_beat(winning_play, last_play)

    def test_different_length_straights_cannot_beat(self):
        """Test that straights of different lengths cannot beat each other."""
        last_play = Play(player_id=0, cards=[
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.FOUR, Suit.CLUBS),
            Card(Rank.FIVE, Suit.DIAMONDS)
        ])
        longer_straight = Play(player_id=1, cards=[
            Card(Rank.SIX, Suit.HEARTS),
            Card(Rank.SEVEN, Suit.CLUBS),
            Card(Rank.EIGHT, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.SPADES)
        ])

        assert not can_beat(longer_straight, last_play)


class TestGetValidPlays:
    """Test valid action generation."""

    def test_empty_hand_only_pass(self):
        """Test that empty hand can only pass."""
        valid = get_valid_plays([], None)
        assert valid == [[]]

    def test_first_play_includes_all_types(self):
        """Test that first play allows all play types."""
        hand = [
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.THREE, Suit.CLUBS),
            Card(Rank.FOUR, Suit.DIAMONDS),
            Card(Rank.FIVE, Suit.SPADES)
        ]
        valid = get_valid_plays(hand, None)

        # Should include: pass, 4 singles, 1 pair, 1 straight
        assert [] in valid  # Pass
        assert len([p for p in valid if len(p) == 1]) == 4  # 4 singles
        assert len([p for p in valid if len(p) == 2]) == 1  # 1 pair
        assert len([p for p in valid if len(p) == 3]) == 1  # 1 straight

    def test_must_play_higher_single(self):
        """Test that only higher singles are valid."""
        hand = [
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.CLUBS),
            Card(Rank.KING, Suit.DIAMONDS)
        ]
        last_play = Play(player_id=0, cards=[Card(Rank.FIVE, Suit.HEARTS)])
        valid = get_valid_plays(hand, last_play)

        # Should include: pass, King single, and any straights
        single_plays = [p for p in valid if len(p) == 1]
        assert len(single_plays) == 1
        assert single_plays[0][0].rank == Rank.KING

    def test_must_play_higher_pair(self):
        """Test that only higher pairs are valid."""
        hand = [
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.THREE, Suit.CLUBS),
            Card(Rank.NINE, Suit.DIAMONDS),
            Card(Rank.NINE, Suit.SPADES),
            Card(Rank.KING, Suit.HEARTS),
            Card(Rank.KING, Suit.CLUBS)
        ]
        last_play = Play(player_id=0, cards=[
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.CLUBS)
        ])
        valid = get_valid_plays(hand, last_play)

        # Should include: pass, 9 pair, King pair, and four-of-kinds
        pair_plays = [p for p in valid if len(p) == 2]
        assert len(pair_plays) == 2
        pair_ranks = [p[0].rank for p in pair_plays]
        assert Rank.NINE in pair_ranks
        assert Rank.KING in pair_ranks
        assert Rank.THREE not in pair_ranks


class TestDetermineTrickWinner:
    """Test trick winner determination."""

    def test_only_player_wins(self):
        """Test that if only one player plays, they win."""
        plays = [
            Play(player_id=0, cards=[Card(Rank.FIVE, Suit.HEARTS)]),
            Play(player_id=1, cards=[]),
            Play(player_id=2, cards=[]),
            Play(player_id=3, cards=[])
        ]
        winner = determine_trick_winner(plays)
        assert winner == 0

    def test_highest_play_wins(self):
        """Test that the highest play wins."""
        plays = [
            Play(player_id=0, cards=[Card(Rank.FIVE, Suit.HEARTS)]),
            Play(player_id=1, cards=[Card(Rank.NINE, Suit.CLUBS)]),
            Play(player_id=2, cards=[Card(Rank.SEVEN, Suit.DIAMONDS)]),
            Play(player_id=3, cards=[])
        ]
        winner = determine_trick_winner(plays)
        assert winner == 1

    def test_four_of_kind_beats_pair(self):
        """Test that four-of-a-kind beats a pair."""
        plays = [
            Play(player_id=0, cards=[
                Card(Rank.ACE, Suit.HEARTS),
                Card(Rank.ACE, Suit.CLUBS)
            ]),
            Play(player_id=1, cards=[
                Card(Rank.FOUR, Suit.HEARTS),
                Card(Rank.FOUR, Suit.CLUBS),
                Card(Rank.FOUR, Suit.DIAMONDS),
                Card(Rank.FOUR, Suit.SPADES)
            ])
        ]
        winner = determine_trick_winner(plays)
        assert winner == 1


class TestHasThreeOfDiamonds:
    """Test detection of 3 of diamonds."""

    def test_has_three_of_diamonds(self):
        """Test detection when hand contains 3♦."""
        hand = [
            Card(Rank.THREE, Suit.DIAMONDS),
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES)
        ]
        assert has_three_of_diamonds(hand)

    def test_does_not_have_three_of_diamonds(self):
        """Test detection when hand does not contain 3♦."""
        hand = [
            Card(Rank.THREE, Suit.HEARTS),
            Card(Rank.FIVE, Suit.HEARTS),
            Card(Rank.KING, Suit.SPADES)
        ]
        assert not has_three_of_diamonds(hand)

    def test_empty_hand(self):
        """Test detection with empty hand."""
        assert not has_three_of_diamonds([])
