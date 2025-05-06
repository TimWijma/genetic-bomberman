from enum import Enum
from typing import TypedDict
from pommerman import constants
from pommerman.agents.base_agent import BaseAgent
from gym.spaces import Discrete
import numpy as np

class PommermanBoard(TypedDict, total=False):
    board: np.ndarray # 11x11 array
    bomb_blast_strength: np.ndarray
    bomb_life: np.ndarray
    bomb_moving_direction: np.ndarray
    flame_life: np.ndarray
    game_type: int
    game_env: str
    position: tuple # (x, y) coordinates
    blast_strength: int # Blast strength of the agent
    can_kick: bool 
    teammate: constants.Item # Teammate's ID
    ammo: int # Number of bombs
    enemies: list # List of enemies
    step_count: int # Current step in the episode
    alive: list # List of booleans indicating if each agent is alive

class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

class TestAgent(BaseAgent):
    def act(self, obs: PommermanBoard, action_space: Discrete):
        # return action_space.sample()

        agent_id = self._character.agent_id
        if agent_id == 1:
            return 0
        
        return action_space.sample()

    def _check_direction_safety(obs: PommermanBoard, direction: Direction) -> bool:
        # Check if the direction is safe
        board = obs['board']
        x, y = obs['position']
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy

        # Check if the new position is within bounds
        if new_x < 0 or new_x >= board.shape[0] or new_y < 0 or new_y >= board.shape[1]:
            return False
        
        # Check if the new position is a wall or a bomb
        if board[new_x, new_y] == constants.Item.WALL or board[new_x, new_y] == constants.Item.BOMB:
            return False
        
        # Check if the new position is a bomb blast
        if obs['bomb_blast_strength'][new_x, new_y] > 0:
            return False
        
        # Check if the new position is a flame
        if obs['flame_life'][new_x, new_y] > 0:
            return False
        
        # Check if the new position is another agent
        for i in range(len(obs['alive'])):
            if obs['alive'][i] and (obs['position'][i][0], obs['position'][i][1]) == (new_x, new_y):
                return False

        return True
