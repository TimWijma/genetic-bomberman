import random
from typing import List, Set, Tuple, TypedDict
from pommerman.agents.base_agent import BaseAgent
from gym.spaces import Discrete
from pommerman import characters, constants
from genetic.common_types import ActionType, Direction, OperatorType, PommermanBoard, Rule, ConditionType
import numpy as np

class ProcessedBoard(TypedDict):
    enemies: List[Tuple[int, int]]
    bombs: List[Tuple[int, int]]
    wood: List[Tuple[int, int]]
    is_rigid: np.ndarray
    is_wood: np.ndarray
    is_obstacle: np.ndarray

class GeneticAgent(BaseAgent):
    def __init__(self, rules: List[Rule], individual_index = -1, character=characters.Bomber):
        super().__init__(character)

        self.rules = rules
        self.individual_index = individual_index
        self.step_count = 0
        self.visited_tiles = set()
        self.bombs_placed = 0
        self.total_distance = 0
    
    def act(self, obs: PommermanBoard, action_space: Discrete):
        self.step_count += 1
        self.visited_tiles.add(obs['position'])

        processed_board = self.process_board(obs)

        self._distance_to_enemies(obs, processed_board)

        action = self.evaluate(obs, processed_board)

        if action is None:
            return 0

        if action == ActionType.PLACE_BOMB and obs['ammo'] > 0:
            self.bombs_placed += 1

        return action.value

    def reset_state(self):
        self.step_count = 0
        self.visited_tiles = set()
        self.bombs_placed = 0
        self.total_distance = 0
        self.average_distance = 0

    def episode_end(self, reward):
        self.average_distance = self.total_distance / self.step_count if self.step_count > 0 else 0

        return super().episode_end(reward)

    def process_board(self, obs: PommermanBoard) -> ProcessedBoard:
        board = obs['board']
        board_blast_strength = obs['bomb_blast_strength']

        enemy_mask = np.zeros(board.shape, dtype=bool)
        for enemy_object in obs['enemies']:
            enemy_mask |= (board == enemy_object.value)
        enemy_positions = np.argwhere(enemy_mask)
        
        bomb_mask = (board_blast_strength > 0)
        bomb_positions = np.argwhere(bomb_mask)

        # wood_positions = np.argwhere(board == constants.Item.Wood.value)

        is_rigid_mask = (board == constants.Item.Rigid.value)
        is_wood_mask = (board == constants.Item.Wood.value)
        is_obstacle = is_rigid_mask | is_wood_mask
        
        return {
            'enemies': [tuple(pos) for pos in enemy_positions],
            'bombs': [tuple(pos) for pos in bomb_positions],
            # 'wood': wood_positions,
            'is_rigid': is_rigid_mask,
            'is_wood': is_wood_mask,
            'is_obstacle': is_obstacle,
        }
        
    def evaluate(self, obs: PommermanBoard, processed_board: ProcessedBoard):
        bomb_conditions = {
            ConditionType.IS_BOMB_UP: self._is_bomb_in_direction(obs, processed_board, Direction.UP),
            ConditionType.IS_BOMB_DOWN: self._is_bomb_in_direction(obs, processed_board, Direction.DOWN),
            ConditionType.IS_BOMB_LEFT: self._is_bomb_in_direction(obs, processed_board, Direction.LEFT),
            ConditionType.IS_BOMB_RIGHT: self._is_bomb_in_direction(obs, processed_board, Direction.RIGHT),
        }
        is_bomb_in_range = bomb_conditions[ConditionType.IS_BOMB_DOWN] or \
            bomb_conditions[ConditionType.IS_BOMB_UP] or \
            bomb_conditions[ConditionType.IS_BOMB_LEFT] or \
            bomb_conditions[ConditionType.IS_BOMB_RIGHT]

        conditions = {
            ConditionType.IS_BOMB_IN_RANGE: is_bomb_in_range,
            ConditionType.IS_BOMB_UP: bomb_conditions[ConditionType.IS_BOMB_UP],
            ConditionType.IS_BOMB_DOWN: bomb_conditions[ConditionType.IS_BOMB_DOWN],
            ConditionType.IS_BOMB_LEFT: bomb_conditions[ConditionType.IS_BOMB_LEFT],
            ConditionType.IS_BOMB_RIGHT: bomb_conditions[ConditionType.IS_BOMB_RIGHT],
            ConditionType.IS_WOOD_IN_RANGE: self._is_wood_in_range(obs, processed_board),
            ConditionType.CAN_MOVE_UP: self._can_move(obs, Direction.UP),
            ConditionType.CAN_MOVE_DOWN: self._can_move(obs, Direction.DOWN),
            ConditionType.CAN_MOVE_LEFT: self._can_move(obs, Direction.LEFT),
            ConditionType.CAN_MOVE_RIGHT: self._can_move(obs, Direction.RIGHT),
            ConditionType.IS_TRAPPED: self._is_trapped(obs),
            ConditionType.HAS_BOMB: obs['ammo'] > 0,
            ConditionType.IS_ENEMY_IN_RANGE: self._is_enemy_in_range(obs, processed_board),
            ConditionType.IS_BOMB_ON_PLAYER: self._is_bomb_on_player(obs),
        }

        satisfied_rules = []
        for rule in self.rules:
            if len(rule.conditions) == 0:
                continue
            
            # If there is 1 condition, check if it is satisfied 
            if len(rule.conditions) == 1:
                result = conditions.get(rule.conditions[0], False)
                
                if result:
                    satisfied_rules.append(rule)

                continue

            # If there are multiple conditions, check if all of them are satisfied
            if len(rule.conditions) > 1:
                result = conditions.get(rule.conditions[0], False)
                
                for i in range(len(rule.operators)):
                    # Break if there are no more conditions
                    if i + 1 >= len(rule.conditions):
                        break
                    
                    next_condition = conditions.get(rule.conditions[i + 1], False)
                    operator = rule.operators[i]
                    if operator == OperatorType.AND:
                        result = result and next_condition
                    elif operator == OperatorType.OR:
                        result = result or next_condition
                    else:
                        raise ValueError(f"Unknown operator: {operator}")
                    
                if result:
                    satisfied_rules.append(rule)
                    continue

        if len(satisfied_rules) > 0:
            rule = random.choice(satisfied_rules)
            return rule.action

    def _can_move(self, obs: PommermanBoard, direction: Direction) -> bool:
        board = obs['board']
        y, x = obs['position']
        dx, dy = direction.value
        new_y, new_x = y + dy, x + dx
        
        rows, cols = board.shape

        if not (0 <= new_y < rows and 0 <= new_x < cols):
            return False

        if board[new_y, new_x] == constants.Item.Passage.value:
            return True

        # TODO: Add support for powerups.

        return False # Rigid, Wood, Bomb, Agent, etc.
    
    def _is_bomb_on_player(self, obs: PommermanBoard):
        y, x = obs['position']
        blast_strength = obs['bomb_blast_strength']
        
        return blast_strength[y, x] > 0
    
    def _is_bomb_in_direction(self, obs: PommermanBoard, processed_board: ProcessedBoard, direction: Direction):
        y, x = obs['position']
        
        bomb_blast_strength_map = obs['bomb_blast_strength']
        bomb_coords = processed_board['bombs']
        is_obstacle = processed_board['is_obstacle']

        is_horizontal = (direction == Direction.LEFT or direction == Direction.RIGHT)
        
        for bomb_y, bomb_x in bomb_coords:
            # Check if the bomb is in the same row or column as the player
            if (is_horizontal and bomb_y != y) or (not is_horizontal and bomb_x != x):
                continue
            
            # Check if the bomb is in the specified direction
            if (direction == Direction.UP and bomb_y >= y) or \
                (direction == Direction.DOWN and bomb_y <= y) or \
                (direction == Direction.LEFT and bomb_x >= x) or \
                (direction == Direction.RIGHT and bomb_x <= x):
                continue    

            bomb_strength = bomb_blast_strength_map[bomb_y, bomb_x]
            if is_horizontal:
                dist = abs(bomb_x - x)
                if dist < bomb_strength:
                    path_slice = is_obstacle[y, min(x, bomb_x) + 1 : max(x, bomb_x)]
                    if not np.any(path_slice):
                        return True
            else:
                dist = abs(bomb_y - y)
                if dist < bomb_strength:
                    path_slice = is_obstacle[min(y, bomb_y) + 1 : max(y, bomb_y), x]
                    if not np.any(path_slice):
                        return True
        return False

    def _is_wood_in_range(self, obs: PommermanBoard, processed_board: ProcessedBoard):
        y, x = obs['position']
        player_blast_strength = obs['blast_strength']
        board = obs['board']
        is_rigid = processed_board['is_rigid']
        rows, cols = board.shape

        directions = [
            (0, 1),   # Right
            (0, -1),  # Left
            (1, 0),   # Down
            (-1, 0)   # Up
        ]

        for dy, dx in directions:
            for i in range(1, player_blast_strength + 1):
                new_y = y + dy * i
                new_x = x + dx * i

                if not (0 <= new_y < rows and 0 <= new_x < cols):
                    break
                
                if is_rigid[new_y, new_x]:
                    break

                # Check if the new position is a wood tile
                if board[new_y, new_x] == constants.Item.Wood.value:
                    return True

        return False
    
    def _is_trapped(self, obs: PommermanBoard):
        board = obs['board']
        y, x = obs['position']
        
        rows, cols = board.shape
        
        directions_offsets = [
            (0, 1),   # Right
            (0, -1),  # Left
            (1, 0),   # Down
            (-1, 0)   # Up
        ]
        
        for dy, dx in directions_offsets:
            new_y, new_x = y + dy, x + dx
            
            if 0 <= new_y < rows and 0 <= new_x < cols:
                if board[new_y, new_x] == constants.Item.Passage.value:
                    return False
        
        return True
    
    # Method that checks if there an an enemy within blast range in a given direction
    def _is_enemy_in_range(self, obs: PommermanBoard, processed_board: ProcessedBoard):
        y, x = obs['position']

        player_blast_strength = obs['blast_strength']
        enemy_coords = processed_board['enemies']

        is_obstacle = processed_board['is_obstacle']

        for enemy_y, enemy_x in enemy_coords:
            if enemy_y == y and enemy_x != x:
                min_x, max_x = min(x, enemy_x), max(x, enemy_x)
                if not np.any(is_obstacle[enemy_y, min_x + 1:max_x]):
                    if (abs(enemy_x - x) < player_blast_strength):
                        return True
            elif enemy_x == x and enemy_y != y:
                min_y, max_y = min(y, enemy_y), max(y, enemy_y)
                if not np.any(is_obstacle[min_y + 1:max_y, enemy_x]):
                    if (abs(enemy_y - y) < player_blast_strength):
                        return True

        return False
    
    def position_in_bounds(self, obs: PommermanBoard, position: tuple):
        board = obs['board']
        y, x = position
        return 0 <= x < board.shape[1] and 0 <= y < board.shape[0]

    def manhattan_distance(self, pos1: tuple, pos2: tuple):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _distance_to_enemies(self, obs: PommermanBoard, processed_board: ProcessedBoard):
        y, x = obs['position']
        enemy_coords = processed_board['enemies']

        distances = [self.manhattan_distance((y, x), (enemy_y, enemy_x)) for enemy_y, enemy_x in enemy_coords]
        average_distance = np.mean(distances)
        self.total_distance += average_distance