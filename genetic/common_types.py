from enum import Enum
from typing import Dict, List, Optional, Tuple, TypedDict
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

class AgentResult:
    def __init__(
        self, 
        agent_id: int, 
        agent_type, 
        winner: bool, 
        step_count: int, 
        visited_tiles: int, 
        bombs_placed: int, 
        individual_index: int, 
        kills: List[int], 
        wood_exploded: int,
        average_distance: float,
        is_alive: bool,
    ):
        self.id = agent_id
        self.agent_type = agent_type
        self.winner = winner
        self.step_count = step_count
        self.visited_tiles = visited_tiles
        self.bombs_placed = bombs_placed
        self.individual_index = individual_index
        self.kills = kills
        self.wood_exploded = wood_exploded
        self.average_distance = average_distance
        self.is_alive = is_alive
        
    def __str__(self):
        return (f"Agent {self.id} ({self.agent_type}):\n"
                f"  Winner:         {self.winner}\n"
                f"  Steps:          {self.step_count}\n"
                f"  Visited Tiles:  {self.visited_tiles}\n"
                f"  Bombs Placed:   {self.bombs_placed}\n"
                f"  Kills:          {self.kills}\n"
                f"  Wood Exploded:  {self.wood_exploded}\n"
                f"  Average Distance: {self.average_distance}\n"
                f"  Alive:          {self.is_alive}\n")

    def __repr__(self):
        return self.__str__()

class GameResult:
    def __init__(self, agent_results: List[AgentResult], total_steps: int):
        self.agent_results = agent_results
        self.total_steps = total_steps

    def __str__(self):
        result_str = f"Total Steps: {self.total_steps}\n"
        for i, agent_result in enumerate(self.agent_results):
            result_str += f"{agent_result}"
        return result_str

    def __repr__(self):
        return self.__str__()
    
    def to_json(self):
        return {
            "total_steps": self.total_steps,
            "agents": [
                {
                    "winner": agent_result.winner,
                    "step_count": agent_result.step_count,
                    "visited_tiles": agent_result.visited_tiles,
                    "bombs_placed": agent_result.bombs_placed,
                    "individual_index": agent_result.individual_index,
                    "kills": agent_result.kills,
                    "wood_exploded": agent_result.wood_exploded,
                }
                for agent_result in self.agent_results
            ],
        }

class Direction(Enum):
    UP = (0, -1)
    RIGHT = (1, 0)
    DOWN = (0, 1)
    LEFT = (-1, 0)

class ConditionType(Enum):
    # # IS_BOMB_IN_RANGE = 0,
    # IS_BOMB_UP = 1,
    # IS_BOMB_DOWN = 2,
    # IS_BOMB_LEFT = 3,
    # IS_BOMB_RIGHT = 4,
    # IS_WOOD_IN_RANGE = 5,
    # CAN_MOVE_UP = 6,
    # CAN_MOVE_DOWN = 7,
    # CAN_MOVE_LEFT = 8,
    # CAN_MOVE_RIGHT = 9,
    # IS_TRAPPED = 10,
    # HAS_BOMB = 11,
    # IS_ENEMY_IN_RANGE = 12,
    # IS_BOMB_ON_PLAYER = 13,
    # IS_ENEMY_UP = 14,
    # IS_ENEMY_DOWN = 15,
    # IS_ENEMY_LEFT = 16,
    # IS_ENEMY_RIGHT = 17,
    # # IS_SAFE_TO_PLACE_BOMB = 18,

    IS_IN_BLAST_RADIUS = 0,
    IS_ADJACENT_TO_WOOD = 1,
    IS_ADJACENT_TO_BOMB = 2,
    CAN_MOVE_UP = 3,
    CAN_MOVE_DOWN = 4,
    CAN_MOVE_LEFT = 5,
    CAN_MOVE_RIGHT = 6,
    IS_TRAPPED = 7,
    HAS_BOMB = 8,
    IS_ENEMY_IN_BLAST_RANGE = 9,
    IS_BOMB_ON_PLAYER = 10,
    IS_ENEMY_UP = 11,
    IS_ENEMY_DOWN = 12,
    IS_ENEMY_LEFT = 13,
    IS_ENEMY_RIGHT = 14,
    # --- NEW CONDITIONS BASED ON SimpleAgent ---
    IS_SAFE_TO_PLACE_BOMB = 15, # Uses _maybe_bomb
    IS_WOOD_IN_RANGE = 16, # Uses _nearest_position + _djikstra
    IS_ENEMY_IN_RANGE = 17, # Uses _nearest_position + _djikstra
    IS_POWERUP_IN_RANGE = 18, # Uses _nearest_position + _djikstra
    DISTANCE_TO_ENEMY_GT = 19, # Uses _djikstra.dist
    DISTANCE_TO_ENEMY_LT = 20, # Uses _djikstra.dist
    DISTANCE_TO_WOOD_GT = 21, # Uses _djikstra.dist
    DISTANCE_TO_WOOD_LT = 22, # Uses _djikstra.dist
    CAN_REACH_SAFE_SPOT = 23, # Uses _find_safe_directions
    # --- END NEW CONDITIONS ---


class Condition:
    def __init__(self, condition_type: ConditionType, negation: bool = False, value: Optional[int] = None):
        self.condition_type = condition_type
        self.negation = negation
        self.value = value

    def __str__(self):
        value_str = f" {self.value}" if self.value is not None else ""
        return f"{'NOT ' if self.negation else ''}{self.condition_type.name}{value_str}".strip()

    def __repr__(self):
        return self.__str__()

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

    MOVE_TOWARDS_ENEMY = 6
    MOVE_TOWARDS_WOOD = 7
    MOVE_TOWARDS_POWERUP = 8
    MOFE_TO_SAFESPOT = 9

class Rule:
    def __init__(self, conditions: List[Condition], operators: List[OperatorType], action: ActionType):
        self.conditions = conditions
        self.operators = operators
        self.action = action
        
    def __str__(self):
        parts = []
        for i, cond in enumerate(self.conditions):
            parts.append(str(cond))
            if i < len(self.operators):
                parts.append(self.operators[i].name)
        return f"IF {' '.join(parts)} THEN {self.action.name}"

    def __repr__(self):
        return self.__str__()