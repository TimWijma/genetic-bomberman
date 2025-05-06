'''This is the base abstraction for agents in pommerman.
All agents should inherent from this class'''
from pommerman import constants
from genetic.common_types import Direction, PommermanBoard
from .. import characters


class BaseAgent:
    """Parent abstract Agent."""

    def __init__(self, character=characters.Bomber):
        self._character = character

    def __getattr__(self, attr):
        return getattr(self._character, attr)

    def act(self, obs, action_space):
        raise NotImplementedError()

    def episode_end(self, reward):
        """This is called at the end of the episode to let the agent know that
        the episode has ended and what is the reward.

        Args:
          reward: The single reward scalar to this agent.
        """
        pass

    def init_agent(self, id_, game_type):
        self._character = self._character(id_, game_type)

    @staticmethod
    def has_user_input():
        return False

    def shutdown(self):
        pass
    
    def _check_direction_safety(self, obs: PommermanBoard, direction: Direction) -> bool:
        # Check if the direction is safe
        board = obs['board']
        x, y = obs['position']
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy

        # Check if the new position is within bounds
        if new_x < 0 or new_x >= board.shape[0] or new_y < 0 or new_y >= board.shape[1]:
            return False
        
        # Check if the new position is free
        # TODO: Add support for powerups
        if board[new_x, new_y] != constants.Item.Passage:
            return False

        return True
