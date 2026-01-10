"""
Greedy agent for the Slave card game.

This agent uses simple rule-based heuristics:
- Always plays the lowest valid card/combination
- Prefers singles over other play types
- Never passes unless forced to
"""

import numpy as np
from agents.base_agent import BaseAgent


class GreedyAgent(BaseAgent):
    """
    Agent that uses simple greedy heuristics.

    Strategy:
    1. Never pass unless it's the only option
    2. Prefer singles > pairs > straights > four-of-a-kind
    3. Within each category, play the lowest cards first
    """

    def __init__(self, player_id: int, name: str = None):
        """
        Initialize a greedy agent.

        Args:
            player_id: Integer ID of the player (0-3)
            name: Optional name for the agent
        """
        super().__init__(player_id, name or f"Greedy_{player_id}")

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select action using simple greedy heuristics.

        Args:
            observation: Encoded game state
            action_mask: Binary mask of valid actions

        Returns:
            Selected action ID
        """
        # Get valid actions
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            return 0  # Should never happen

        if len(valid_actions) == 1:
            return valid_actions[0]

        # Categorize actions by type
        pass_action = None
        singles = []
        pairs = []
        threes = []
        fours = []

        for action in valid_actions:
            if action == 0:
                pass_action = action
            elif 1 <= action <= 52:
                singles.append(action)
            elif 53 <= action <= 130:
                pairs.append(action)
            elif 131 <= action <= 143:
                threes.append(action)
            elif 144 <= action <= 156:
                fours.append(action)

        # Priority: singles > pairs > threes > fours > pass
        # Within each category, choose lowest (which represents lowest card/combination)

        if singles:
            return min(singles)  # Lowest card ID = lowest rank
        elif pairs:
            return min(pairs)
        elif threes:
            return min(threes)
        elif fours:
            return min(fours)
        elif pass_action is not None:
            return pass_action

        # Fallback (should never reach here)
        return valid_actions[0]

    def reset(self):
        """Reset agent state."""
        pass
