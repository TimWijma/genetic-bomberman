from enum import Enum
from typing import TypedDict
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
    teammate: object
    ammo: int # Number of bombs
    enemies: list # List of enemies
    step_count: int # Current step in the episode
    alive: list # List of booleans indicating if each agent is alive

class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)
