"""
Example demonstrating strategic card exchange in the Slave card game.

This shows how King and Queen can strategically choose which cards to give
during the exchange phase, rather than being forced to give their lowest cards.
"""

from game.game_state import GameState, Position
from game.card import Card, Rank

def play_game_round(game):
    """Helper to play one round of the game."""
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


def main():
    print("=" * 60)
    print("Strategic Card Exchange Example")
    print("=" * 60)

    # Create game and play first round
    game = GameState(num_players=4)
    game.reset(seed=42)

    print("\n1. Playing first round to establish positions...")
    play_game_round(game)

    # Show positions
    print("\nRound 1 Results:")
    for player_id in range(4):
        position = game.get_position(player_id)
        print(f"  Player {player_id}: {position.value}")

    # Find King and Queen
    king_id = None
    queen_id = None
    for player_id, position in game.positions.items():
        if position == Position.KING:
            king_id = player_id
        elif position == Position.QUEEN:
            queen_id = player_id

    # Start new round - deal cards but don't exchange yet
    print("\n2. Starting Round 2...")
    game.is_first_round = False
    from game.deck import Deck
    deck = Deck()
    deck.shuffle(seed=100)
    hands = deck.deal(game.num_players)
    game.hands = {i: hands[i] for i in range(game.num_players)}

    # King examines their hand
    print(f"\n3. King (Player {king_id}) examines hand before exchange:")
    king_hand = sorted(game.get_hand(king_id))
    print(f"   Hand: {len(king_hand)} cards, ranks: {sorted([c.rank.value for c in king_hand])}")

    # King checks what they will receive
    cards_from_slave = game.get_cards_to_receive_from_slave()
    print(f"   Will receive from Slave: ranks {[c.rank.value for c in cards_from_slave]}")

    # King makes strategic decision
    print("\n4. King's Strategic Decision:")

    # Count cards by rank
    rank_counts = {}
    for card in king_hand:
        rank_counts[card.rank] = rank_counts.get(card.rank, 0) + 1

    # Find if King has any triplets or pairs they want to keep
    triplets = [rank for rank, count in rank_counts.items() if count >= 3]
    pairs = [rank for rank, count in rank_counts.items() if count >= 2]

    if triplets:
        print(f"   King has triplet(s) of: {', '.join(str(r) for r in triplets)}")
        print(f"   Strategy: Keep the triplets, give away other low cards")

        # Give cards that aren't part of triplets
        cards_to_give = []
        for card in king_hand:
            if card.rank not in triplets and len(cards_to_give) < 2:
                cards_to_give.append(card)

        print(f"   Giving: ranks {[c.rank.value for c in cards_to_give]}")
    elif pairs:
        print(f"   King has pair(s) of: {', '.join(str(r) for r in pairs)}")
        print(f"   Strategy: Keep the pairs, give away other low cards")

        # Give cards that aren't part of pairs
        cards_to_give = []
        for card in king_hand:
            if card.rank not in pairs and len(cards_to_give) < 2:
                cards_to_give.append(card)

        if len(cards_to_give) < 2:
            # Need to give some pair cards
            for card in king_hand:
                if len(cards_to_give) < 2 and card not in cards_to_give:
                    cards_to_give.append(card)

        print(f"   Giving: ranks {[c.rank.value for c in cards_to_give]}")
    else:
        # No triplets or pairs, use default strategy
        cards_to_give = king_hand[:2]
        print(f"   No triplets or pairs, using default (lowest 2 cards)")
        print(f"   Giving: ranks {[c.rank.value for c in cards_to_give]}")

    # Queen also makes a strategic decision
    print(f"\n5. Queen (Player {queen_id}) examines hand before exchange:")
    queen_hand = sorted(game.get_hand(queen_id))
    print(f"   Hand: {len(queen_hand)} cards, ranks: {sorted([c.rank.value for c in queen_hand])}")

    card_from_commoner = game.get_cards_to_receive_from_commoner()
    print(f"   Will receive from Commoner: rank {card_from_commoner.rank.value}")

    # Queen's strategy: keep any valuable low pairs
    queen_rank_counts = {}
    for card in queen_hand:
        queen_rank_counts[card.rank] = queen_rank_counts.get(card.rank, 0) + 1

    queen_pairs = [rank for rank, count in queen_rank_counts.items() if count >= 2]

    if queen_pairs and queen_pairs[0] in [Rank.THREE, Rank.FOUR, Rank.FIVE]:
        # Has low pairs, give a different card
        queen_gives = None
        for card in queen_hand:
            if card.rank not in queen_pairs:
                queen_gives = card
                break
        if queen_gives is None:
            queen_gives = queen_hand[0]
        print(f"   Strategy: Keep low pair of rank {queen_pairs[0].value}, give rank {queen_gives.rank.value}")
    else:
        queen_gives = queen_hand[0]
        print(f"   Strategy: No valuable pairs, give lowest card rank {queen_gives.rank.value}")

    # Perform the strategic exchange
    print("\n6. Performing exchange...")
    game._exchange_cards(king_gives=cards_to_give, queen_gives=queen_gives)

    # Show results
    print("\n7. After Exchange:")
    new_king_hand = sorted(game.get_hand(king_id))
    print(f"   King's new hand: {len(new_king_hand)} cards, ranks: {sorted([c.rank.value for c in new_king_hand])}")

    # Verify strategic decision worked
    if triplets:
        kept_triplets = sum(1 for card in new_king_hand if card.rank in triplets)
        original_triplets = sum(1 for card in king_hand if card.rank in triplets)
        print(f"   ✓ Successfully kept {kept_triplets}/{original_triplets} triplet cards")

    print("\n" + "=" * 60)
    print("Strategic exchange allows King and Queen to:")
    print("  • Keep valuable combinations (pairs, triplets)")
    print("  • Optimize their hand composition")
    print("  • Make strategic decisions based on game state")
    print("=" * 60)


if __name__ == "__main__":
    main()
