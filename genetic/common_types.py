from enum import Enum
from typing import List, TypedDict
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

class GameResult(TypedDict):
    winners: List[int]
    survival_steps: List[int]
    total_steps: int

class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)


class ConditionType(Enum):
    IS_BOMB_IN_RANGE = 0,
    IS_BOMB_UP = 1,
    IS_BOMB_DOWN = 2,
    IS_BOMB_LEFT = 3,
    IS_BOMB_RIGHT = 4,
    IS_WOOD_UP = 5,
    IS_WOOD_DOWN = 6,
    IS_WOOD_LEFT = 7,
    IS_WOOD_RIGHT = 8,
    CAN_MOVE_UP = 9,
    CAN_MOVE_DOWN = 10,
    CAN_MOVE_LEFT = 11,
    CAN_MOVE_RIGHT = 12,
    IS_TRAPPED = 13,
    HAS_BOMB = 14,
    IS_ENEMY_UP = 15,
    IS_ENEMY_DOWN = 16,
    IS_ENEMY_LEFT = 17,
    IS_ENEMY_RIGHT = 18,
    IS_ENEMY_IN_RANGE = 19,

class OperatorType(Enum):
    AND = 0,
    OR = 1,
    
class ActionType(Enum):
    DO_NOTHING = 0
    MOVE_UP = 1
    MOVE_DOWN = 2
    MOVE_LEFT = 3
    MOVE_RIGHT = 4
    PLACE_BOMB = 5

class Rule:
    def __init__(self, conditions: List[ConditionType], operators: List[OperatorType], action: ActionType):
        self.conditions = conditions
        self.operators = operators
        self.action = action
        
    def __str__(self):
        conditions_str = " ".join(
            f"{cond.name} {op.name}" if i < len(self.operators) else cond.name
            for i, (cond, op) in enumerate(zip(self.conditions, self.operators + [None]))
        )
        return f"IF {conditions_str} THEN {self.action.name}"

    def __repr__(self):
        return self.__str__()