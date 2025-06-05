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
            return ActionType.DO_NOTHING

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

        is_rigid_mask = (board == constants.Item.Rigid.value)
        is_wood_mask = (board == constants.Item.Wood.value)
        is_flame_mask = (board == constants.Item.Flames.value)
        is_obstacle = is_rigid_mask | is_wood_mask
        is_bomb_mask = bomb_mask | is_flame_mask
        is_obstacle_with_bombs = is_obstacle | is_bomb_mask
        
        return {
            'enemies': [tuple(pos) for pos in enemy_positions],
            'bombs': [tuple(pos) for pos in bomb_positions],
            # 'wood': wood_positions,
            'is_rigid': is_rigid_mask,
            'is_wood': is_wood_mask,
            'is_bomb': is_bomb_mask,
            'is_flame': is_flame_mask,
            'is_obstacle': is_obstacle,
            'is_obstacle_with_bombs': is_obstacle_with_bombs,
        }
        
    def evaluate(self, obs: PommermanBoard, processed_board: ProcessedBoard) -> ActionType:
        evaluated_conditions = {}

        # Iterate through the rules and evaluate conditions
        # Return the action of the first rule that is satisfied
        for rule in self.rules:
            if not rule.conditions:
                continue

            for condition in rule.conditions:
                if condition not in evaluated_conditions:
                    result = self.evaluate_condition(obs, processed_board, condition)
                    evaluated_conditions[condition] = result

            current_condition_values = [evaluated_conditions.get(cond, False) for cond in rule.conditions]

            # If there is 1 condition, check if it is satisfied 
            if len(rule.conditions) == 1:
                if current_condition_values[0]:
                    return rule.action

            # If there are multiple conditions, check if all of them are satisfied
            elif len(rule.conditions) > 1:
                operators_for_evaluation = list(rule.operators)

                i = 0
                while i < len(operators_for_evaluation):
                    if operators_for_evaluation[i] == OperatorType.AND:
                        current_condition_values[i] = current_condition_values[i] and current_condition_values[i + 1]
                        
                        current_condition_values.pop(i + 1)
                        operators_for_evaluation.pop(i)
                    else:
                        i += 1

                result = current_condition_values[0]
                for i in range(len(operators_for_evaluation)):
                    operator = operators_for_evaluation[i]
                    if operator == OperatorType.OR:
                        result = result or current_condition_values[i + 1]
                    else:
                        raise ValueError(f"Unknown operator: {operator}")

                if result:
                    return rule.action

        # If no rule is satisfied, return a default action
        return ActionType.DO_NOTHING

    def evaluate_condition(self, obs: PommermanBoard, processed_board: ProcessedBoard, condition: ConditionType) -> bool:
        if condition == ConditionType.IS_BOMB_UP:
            return self._is_bomb_in_direction(obs, processed_board, Direction.UP)
        elif condition == ConditionType.IS_BOMB_DOWN:
            return self._is_bomb_in_direction(obs, processed_board, Direction.DOWN)
        elif condition == ConditionType.IS_BOMB_LEFT:
            return self._is_bomb_in_direction(obs, processed_board, Direction.LEFT)
        elif condition == ConditionType.IS_BOMB_RIGHT:
            return self._is_bomb_in_direction(obs, processed_board, Direction.RIGHT)
        elif condition == ConditionType.IS_WOOD_IN_RANGE:
            return self._is_wood_in_range(obs, processed_board)
        elif condition == ConditionType.CAN_MOVE_UP:
            return self._can_move(obs, Direction.UP)
        elif condition == ConditionType.CAN_MOVE_DOWN:
            return self._can_move(obs, Direction.DOWN)
        elif condition == ConditionType.CAN_MOVE_LEFT:
            return self._can_move(obs, Direction.LEFT)
        elif condition == ConditionType.CAN_MOVE_RIGHT:
            return self._can_move(obs, Direction.RIGHT)
        elif condition == ConditionType.IS_TRAPPED:
            return self._is_trapped(obs)
        elif condition == ConditionType.HAS_BOMB:
            return obs['ammo'] > 0
        elif condition == ConditionType.IS_ENEMY_IN_RANGE:
            return self._is_enemy_in_range(obs, processed_board)
        elif condition == ConditionType.IS_BOMB_ON_PLAYER:
            return self._is_bomb_on_player(obs)
        elif condition == ConditionType.IS_ENEMY_UP:
            return self._is_enemy_in_direction(obs, processed_board, Direction.UP)
        elif condition == ConditionType.IS_ENEMY_DOWN:
            return self._is_enemy_in_direction(obs, processed_board, Direction.DOWN)
        elif condition == ConditionType.IS_ENEMY_LEFT:
            return self._is_enemy_in_direction(obs, processed_board, Direction.LEFT)
        elif condition == ConditionType.IS_ENEMY_RIGHT:
            return self._is_enemy_in_direction(obs, processed_board, Direction.RIGHT)

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
    
    def _is_enemy_in_direction(self, obs: PommermanBoard, processed_board: ProcessedBoard, direction: Direction):
        y, x = obs['position']
        enemy_coords = processed_board['enemies']
        is_obstacle_with_bombs = processed_board['is_obstacle_with_bombs']

        # Check if there is a direct line of sight to an enemy in the specified direction
        is_horizontal = (direction == Direction.LEFT or direction == Direction.RIGHT)
        
        for enemy_y, enemy_x in enemy_coords:
            if (is_horizontal and enemy_y != y) or (not is_horizontal and enemy_x != x):
                continue

            # Check if the enemy is in the specified direction
            if (direction == Direction.UP and enemy_y >= y) or \
                (direction == Direction.DOWN and enemy_y <= y) or \
                (direction == Direction.LEFT and enemy_x >= x) or \
                (direction == Direction.RIGHT and enemy_x <= x):
                continue

            if is_horizontal:
                min_x, max_x = min(x, enemy_x), max(x, enemy_x)
                if not np.any(is_obstacle_with_bombs[y, min_x + 1:max_x]):
                    return True
            else:
                min_y, max_y = min(y, enemy_y), max(y, enemy_y)
                if not np.any(is_obstacle_with_bombs[min_y + 1:max_y, enemy_x]):
                    return True

        return False
    
    # Method that checks if there an an enemy within blast range in a given direction
    def _is_enemy_in_range(self, obs: PommermanBoard, processed_board: ProcessedBoard):
        y, x = obs['position']

        player_blast_strength = obs['blast_strength']
        enemy_coords = processed_board['enemies']

        is_obstacle_with_bombs = processed_board['is_obstacle_with_bombs']

        for enemy_y, enemy_x in enemy_coords:
            if enemy_y == y and enemy_x != x:
                min_x, max_x = min(x, enemy_x), max(x, enemy_x)
                if not np.any(is_obstacle_with_bombs[enemy_y, min_x + 1:max_x]):
                    if (abs(enemy_x - x) < player_blast_strength):
                        return True
            elif enemy_x == x and enemy_y != y:
                min_y, max_y = min(y, enemy_y), max(y, enemy_y)
                if not np.any(is_obstacle_with_bombs[min_y + 1:max_y, enemy_x]):
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