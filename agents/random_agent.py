"""
Random agent for the Slave card game.

This agent selects random valid actions uniformly.
Useful as a baseline for evaluating other agents.
"""

import numpy as np
from agents.base_agent import BaseAgent


class RandomAgent(BaseAgent):
    """
    Agent that plays random valid moves.

    This agent uniformly selects from all valid actions without any strategy.
    It serves as a baseline to compare against more sophisticated agents.
    """

    def __init__(self, player_id: int, name: str = None, seed: int = None):
        """
        Initialize a random agent.

        Args:
            player_id: Integer ID of the player (0-3)
            name: Optional name for the agent
            seed: Optional random seed for reproducibility
        """
        super().__init__(player_id, name or f"Random_{player_id}")
        self.rng = np.random.RandomState(seed)

    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select a random valid action.

        Args:
            observation: Encoded game state (not used by random agent)
            action_mask: Binary mask of valid actions

        Returns:
            Randomly selected valid action ID
        """
        # Get indices of valid actions (where mask == 1)
        valid_actions = np.where(action_mask == 1)[0]

        if len(valid_actions) == 0:
            # Should not happen with proper masking, but fallback to pass
            return 0

        # Select random valid action
        return self.rng.choice(valid_actions)

    def reset(self):
        """Reset agent state (random agent has no state to reset)."""
        pass
