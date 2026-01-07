"""
PettingZoo AEC (Agent Environment Cycle) wrapper for the Slave card game.
"""

from typing import Optional, Dict
import numpy as np
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers
from pettingzoo.utils.agent_selector import AgentSelector

from game.game_state import GameState, Position
from game.card import Card
from environment.observations import encode_observation, get_action_mask, decode_action


class SlaveEnv(AECEnv):
    """
    PettingZoo environment for the Slave card game.

    This implements the AEC (Agent Environment Cycle) API for multi-agent RL.
    """

    metadata = {
        "render_modes": ["human"],
        "name": "slave_v0",
        "is_parallelizable": False,
    }

    def __init__(self, render_mode: Optional[str] = None, num_rounds: int = 1):
        """
        Initialize the Slave card game environment.

        Args:
            render_mode: Rendering mode ("human" or None)
            num_rounds: Number of rounds to play before terminating (default: 1)
        """
        super().__init__()

        self.num_players = 4
        self.possible_agents = [f"player_{i}" for i in range(self.num_players)]
        self.agent_name_mapping = {name: i for i, name in enumerate(self.possible_agents)}

        self.render_mode = render_mode
        self.num_rounds = num_rounds

        # Action space: 194 possible actions
        # 0: Pass
        # 1-52: Singles
        # 53-130: Pairs
        # 131-180: Straights
        # 181-193: Four-of-a-kinds
        self.action_space_size = 194

        # Observation space: 155 features
        self.observation_space_size = 155

        # Define spaces
        self._action_spaces = {
            agent: spaces.Discrete(self.action_space_size)
            for agent in self.possible_agents
        }

        self._observation_spaces = {
            agent: spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.observation_space_size,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }

        # Game state
        self.game_state: Optional[GameState] = None

        # Agent tracking
        self.agents = []
        self._agent_selector = None

        # Rewards and terminations
        self._cumulative_rewards: Dict[str, float] = {}
        self._rewards: Dict[str, float] = {}
        self.terminations: Dict[str, bool] = {}
        self.truncations: Dict[str, bool] = {}
        self.infos: Dict[str, dict] = {}

    def observation_space(self, agent: str) -> spaces.Space:
        """Return observation space for an agent."""
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> spaces.Space:
        """Return action space for an agent."""
        return self._action_spaces[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """
        Reset the environment to start a new game.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            None (observations obtained via observe())
        """
        # Initialize game state
        self.game_state = GameState(num_players=self.num_players)
        self.game_state.reset(seed=seed)

        # Reset agent tracking
        self.agents = self.possible_agents.copy()
        self._agent_selector = AgentSelector(self.agents)

        # Set agent_selection to match current player in game
        current_agent = self.possible_agents[self.game_state.current_player]
        self._agent_selector.reinit(self.agents)
        # Advance selector to current player
        while self._agent_selector.next() != current_agent:
            pass
        self.agent_selection = current_agent

        # Reset rewards and terminations
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self._rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Reset num_moves counter
        self.num_moves = 0

        # Track rounds completed
        self.rounds_completed = 0

    def observe(self, agent: str) -> np.ndarray:
        """
        Return observation for a specific agent.

        Args:
            agent: Agent name

        Returns:
            Observation array
        """
        player_id = self.agent_name_mapping[agent]
        observation = encode_observation(self.game_state, player_id)

        # Add action mask to info
        action_mask = get_action_mask(self.game_state, player_id, self.action_space_size)
        self.infos[agent]["action_mask"] = action_mask

        return observation

    def step(self, action: int):
        """
        Execute an action for the current agent.

        Args:
            action: Action ID to execute
        """
        if self.terminations[self.agent_selection] or self.truncations[self.agent_selection]:
            # Agent is done, just return
            return self._was_dead_step(action)

        agent = self.agent_selection
        player_id = self.agent_name_mapping[agent]

        # Decode action to cards
        hand = self.game_state.get_hand(player_id)
        cards = decode_action(action, hand)

        # Validate action
        action_mask = get_action_mask(self.game_state, player_id, self.action_space_size)
        if action_mask[action] == 0:
            # Invalid action - give penalty and don't execute
            self._rewards[agent] = -1.0
            self._accumulate_rewards()

            # Move to next agent without changing game state
            self.agent_selection = self._agent_selector.next()
            return

        # Execute action
        success = self.game_state.play_cards(player_id, cards)

        if not success:
            # Action failed (shouldn't happen with proper masking)
            self._rewards[agent] = -1.0
            self._accumulate_rewards()
            self.agent_selection = self._agent_selector.next()
            return

        # Small negative reward per turn (encourages finishing quickly)
        self._rewards[agent] = -0.01

        # Check if game is over
        if self.game_state.is_game_over():
            self._handle_game_end()
            return

        # Move to next agent
        self.num_moves += 1
        self._accumulate_rewards()

        # Update agent_selection to match game's current_player
        # (game_state handles skipping finished players)
        self.agent_selection = self.possible_agents[self.game_state.current_player]

        # Truncate if game is too long (safety measure)
        if self.num_moves > 500:
            for ag in self.agents:
                self.truncations[ag] = True

    def _handle_game_end(self):
        """
        Handle end of a round - check if game should continue or terminate.
        """
        # Assign rewards based on finishing position
        position_rewards = {
            Position.KING: 10.0,
            Position.QUEEN: 5.0,
            Position.COMMONER: -5.0,
            Position.SLAVE: -10.0
        }

        for agent in self.agents:
            player_id = self.agent_name_mapping[agent]
            position = self.game_state.get_position(player_id)

            # Base reward from position
            self._rewards[agent] = position_rewards[position]

            # Additional penalty for cards remaining in hand
            cards_remaining = len(self.game_state.get_hand(player_id))
            self._rewards[agent] += cards_remaining * -0.01

        self._accumulate_rewards()

        # Increment rounds completed
        self.rounds_completed += 1

        # Check if we should continue to next round or terminate
        if self.rounds_completed < self.num_rounds:
            # Continue to next round
            self._start_next_round()
        else:
            # Game is over - mark all agents as terminated
            for agent in self.agents:
                self.terminations[agent] = True

    def _start_next_round(self):
        """
        Start the next round of play.
        """
        # Start new round in game state (uses default strategy: King/Queen give lowest cards)
        self.game_state.start_new_round()

        # Reset environment state for new round
        self._rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}

        # Reset move counter for new round
        self.num_moves = 0

        # Update agent selection to match game's current player
        self.agent_selection = self.possible_agents[self.game_state.current_player]

    def _accumulate_rewards(self):
        """Accumulate rewards for all agents."""
        for agent in self.agents:
            self._cumulative_rewards[agent] += self._rewards[agent]

    def _was_dead_step(self, action):
        """Handle step when agent is already terminated."""
        # Agent is already done, return without changing state
        return

    def render(self):
        """
        Render the environment (human-readable output).
        """
        if self.render_mode != "human":
            return

        print("\n" + "=" * 60)
        print(f"Round {self.game_state.round_number}, Move {self.num_moves}")
        print("=" * 60)

        # Show current player
        current_agent = self.agent_selection
        current_player_id = self.agent_name_mapping[current_agent]
        print(f"\nCurrent player: {current_agent} (Player {current_player_id})")

        # Show hands
        for agent in self.agents:
            player_id = self.agent_name_mapping[agent]
            hand = self.game_state.get_hand(player_id)
            position = self.game_state.get_position(player_id)
            position_str = position.value if position else "None"

            print(f"\n{agent} ({position_str}): {len(hand)} cards")

            if player_id == current_player_id:
                # Show current player's hand
                hand_str = ", ".join([f"{card.rank.name}{card.suit.name[0]}" for card in sorted(hand)])
                print(f"  Hand: {hand_str}")

        # Show last play
        if self.game_state.last_play:
            play = self.game_state.last_play
            if len(play.cards) > 0:
                cards_str = ", ".join([f"{card.rank.name}{card.suit.name[0]}" for card in play.cards])
                print(f"\nLast play: Player {play.player_id} played {cards_str}")
            else:
                print(f"\nLast play: Player {play.player_id} passed")
        else:
            print("\nNo last play (start of trick)")

        print("=" * 60)

    def close(self):
        """Clean up resources."""
        pass


def env(**kwargs):
    """
    Create a wrapped SlaveEnv environment.

    This is the recommended way to create the environment.
    """
    env = SlaveEnv(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env
