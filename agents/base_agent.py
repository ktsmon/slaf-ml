"""
Base agent interface for the Slave card game.

All agents (random, greedy, RL) should inherit from this base class.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseAgent(ABC):
    """
    Abstract base class for all agents playing the Slave card game.

    Agents must implement the select_action method to choose actions
    based on observations and valid action masks.
    """

    def __init__(self, player_id: int, name: Optional[str] = None):
        """
        Initialize a base agent.

        Args:
            player_id: Integer ID of the player (0-3)
            name: Optional name for the agent
        """
        self.player_id = player_id
        self.name = name or f"Agent_{player_id}"

    @abstractmethod
    def select_action(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """
        Select an action based on the current observation.

        Args:
            observation: Encoded game state observation (155 features)
            action_mask: Binary mask of valid actions (194 actions)

        Returns:
            Action ID (0-193) to execute
        """
        pass

    def reset(self):
        """
        Reset agent state at the start of a new episode.

        Override this method if the agent maintains internal state
        that needs to be reset between episodes.
        """
        pass

    def __repr__(self) -> str:
        """String representation of the agent."""
        return f"{self.__class__.__name__}(player_id={self.player_id}, name='{self.name}')"
