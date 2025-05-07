from typing import List
from pommerman.agents.base_agent import BaseAgent
from gym.spaces import Discrete
from pommerman import characters, constants
from genetic.common_types import ActionType, Direction, OperatorType, PommermanBoard, Rule, ConditionType
import numpy as np

class GeneticAgent(BaseAgent):
    def __init__(self, rules: List[Rule], individual_index = -1, character=characters.Bomber):
        super().__init__(character)

        self.rules = rules
        self.individual_index = individual_index
        self.step_count = 0
        self.visited_tiles = set()
        self.bombs_placed = 0
    
    def act(self, obs: PommermanBoard, action_space: Discrete):
        self.step_count += 1
        self.visited_tiles.add(obs['position'])

        action = self.evaluate(obs)
        
        if action is None:
            return 0

        if action == ActionType.PLACE_BOMB and obs['ammo'] > 0:
            self.bombs_placed += 1
        
        return action.value

    def evaluate(self, obs: PommermanBoard):
        conditions = {
            ConditionType.IS_BOMB_IN_RANGE: self._is_bomb_in_range(obs),
            ConditionType.IS_BOMB_UP: self._is_bomb_in_direction(obs, Direction.UP),
            ConditionType.IS_BOMB_DOWN: self._is_bomb_in_direction(obs, Direction.DOWN),
            ConditionType.IS_BOMB_LEFT: self._is_bomb_in_direction(obs, Direction.LEFT),
            ConditionType.IS_BOMB_RIGHT: self._is_bomb_in_direction(obs, Direction.RIGHT),
            ConditionType.IS_WOOD_UP: self._is_wood_in_direction(obs, Direction.UP),
            ConditionType.IS_WOOD_DOWN: self._is_wood_in_direction(obs, Direction.DOWN),
            ConditionType.IS_WOOD_LEFT: self._is_wood_in_direction(obs, Direction.LEFT),
            ConditionType.IS_WOOD_RIGHT: self._is_wood_in_direction(obs, Direction.RIGHT),
            ConditionType.CAN_MOVE_UP: self._can_move(obs, Direction.UP),
            ConditionType.CAN_MOVE_DOWN: self._can_move(obs, Direction.DOWN),
            ConditionType.CAN_MOVE_LEFT: self._can_move(obs, Direction.LEFT),
            ConditionType.CAN_MOVE_RIGHT: self._can_move(obs, Direction.RIGHT),
            ConditionType.IS_TRAPPED: self._is_trapped(obs),
            ConditionType.HAS_BOMB: obs['ammo'] > 0,
            ConditionType.IS_ENEMY_UP: self._is_enemy_in_direction(obs, Direction.UP),
            ConditionType.IS_ENEMY_DOWN: self._is_enemy_in_direction(obs, Direction.DOWN),
            ConditionType.IS_ENEMY_LEFT: self._is_enemy_in_direction(obs, Direction.LEFT),
            ConditionType.IS_ENEMY_RIGHT: self._is_enemy_in_direction(obs, Direction.RIGHT),
        }
        
        # print({
        #     'up': conditions[ConditionType.IS_ENEMY_UP],
        #     'down': conditions[ConditionType.IS_ENEMY_DOWN],
        #     'left': conditions[ConditionType.IS_ENEMY_LEFT],
        #     'right': conditions[ConditionType.IS_ENEMY_RIGHT],
        # })
        
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
                    # print(f"Satisfied rules: {satisfied_rules}")
                    continue

        # print(f"Satisfied rules: {satisfied_rules}")

        if len(satisfied_rules) > 0:
            rule = np.random.choice(satisfied_rules)
            # print(f"Chosen rule: {rule}")
            return rule.action

    def _can_move(self, obs: PommermanBoard, direction: Direction) -> bool:
        board = obs['board']
        y, x = obs['position']
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy
        
        if not self.position_in_bounds(obs, (new_y, new_x)):
            # print("Out of bounds")
            return False
        
        target_value = board[new_y, new_x]

        if target_value == constants.Item.Passage.value:
            return True

        # TODO: Add support for powerups

        return False

    # Check if the agent is in a tile that will be hit by a bomb
    def _is_bomb_in_range(self, obs: PommermanBoard):
        board = obs['board']
        y, x = obs['position']
        blast_strength = obs['bomb_blast_strength']
        
        bomb_positions = np.where(blast_strength > 0)
        bomb_coords = list(zip(bomb_positions[1], bomb_positions[0]))

        for bomb_x, bomb_y in bomb_coords:
            bomb_strenght = blast_strength[bomb_y, bomb_x]
            
            # Check if the bomb is in the same row or column as the player
            # Checks if there is a wall between the bomb and the player
            # Then checks if the bomb is strong enough to hit the player
            if bomb_y == y and bomb_x != x:
                min_x, max_x = min(x, bomb_x), max(x, bomb_x)
                if not any(board[bomb_y, i] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(min_x + 1, max_x)):
                    if (abs(bomb_x - x) < bomb_strenght):
                        # print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
            elif bomb_x == x and bomb_y != y:
                min_y, max_y = min(y, bomb_y), max(y, bomb_y)
                if not any(board[i, bomb_x] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(min_y + 1, max_y)):
                    if (abs(bomb_y - y) < bomb_strenght):
                        # print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
                            
        return False
    
    def _is_bomb_in_direction(self, obs: PommermanBoard, direction: Direction):
        board = obs['board']
        y, x = obs['position']
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy
        
        # # Check if the new position is within bounds
        if not self.position_in_bounds(obs, (new_y, new_x)):
            return False

        blast_strength = obs['bomb_blast_strength']
        
        bomb_positions = np.where(blast_strength > 0)
        bomb_coords = list(zip(bomb_positions[1], bomb_positions[0]))

        for bomb_x, bomb_y in bomb_coords:
            bomb_strength = blast_strength[bomb_y, bomb_x]
            if direction == Direction.UP and bomb_x == x and bomb_y < y:
                if not any(board[i, x] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(bomb_y + 1, y)):
                    if (y - bomb_y < bomb_strength):
                        # print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
            elif direction == Direction.DOWN and bomb_x == x and bomb_y > y:
                if not any(board[i, x] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(y + 1, bomb_y)):
                    if (bomb_y - y < bomb_strength):
                        # print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
            elif direction == Direction.LEFT and bomb_y == y and bomb_x < x:
                if not any(board[y, i] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(bomb_x + 1, x)):
                    if (x - bomb_x < bomb_strength):
                        # print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
            elif direction == Direction.RIGHT and bomb_y == y and bomb_x > x:
                if not any(board[y, i] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(x + 1, bomb_x)):
                    if (bomb_x - x < bomb_strength):
                        # print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
                    
        return False
    
    def _is_wood_in_direction(self, obs: PommermanBoard, direction: Direction):
        board = obs['board']
        y, x = obs['position']
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy
        
        # Check if the new position is within bounds
        if not self.position_in_bounds(obs, (new_y, new_x)):
            return False

        target_value = board[new_y, new_x]

        if target_value == constants.Item.Wood.value:
            return True

        return False
    
    def _is_trapped(self, obs: PommermanBoard):
        board = obs['board']
        y, x = obs['position']
        
        for direction in Direction:
            dx, dy = direction.value
            new_x, new_y = x + dx, y + dy
            
            # Check if the new position is within bounds
            if not self.position_in_bounds(obs, (new_y, new_x)):
                continue

            target_value = board[new_y, new_x]

            if target_value == constants.Item.Passage.value:
                return False
            
        return True
    
    # Method that checks if there an an enemy within blast range in a given direction
    def _is_enemy_in_direction(self, obs: PommermanBoard, direction: Direction):
        board = obs['board']
        y, x = obs['position']
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy
        
        # Check if the new position is within bounds
        if not self.position_in_bounds(obs, (new_y, new_x)):
            return False

        blast_strength = obs['blast_strength']
        enemies = [enemy.value for enemy in obs['enemies']]
        enemy_positions = np.where(np.isin(board, enemies))
        enemy_coords = list(zip(enemy_positions[1], enemy_positions[0]))

        
        for enemy_x, enemy_y in enemy_coords:
            if direction == Direction.UP and enemy_x == x and enemy_y < y:
                if not any(board[i, x] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(enemy_y + 1, y)):
                    if (y - enemy_y < blast_strength):
                        # print(f"Enemy at ({enemy_x}, {enemy_y}) will hit agent at ({x}, {y})")
                        return True
            elif direction == Direction.DOWN and enemy_x == x and enemy_y > y:
                if not any(board[i, x] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(y + 1, enemy_y)):
                    if (enemy_y - y < blast_strength):
                        # print(f"Enemy at ({enemy_x}, {enemy_y}) will hit agent at ({x}, {y})")
                        return True
            elif direction == Direction.LEFT and enemy_y == y and enemy_x < x:
                if not any(board[y, i] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(enemy_x + 1, x)):
                    if (x - enemy_x < blast_strength):
                        # print(f"Enemy at ({enemy_x}, {enemy_y}) will hit agent at ({x}, {y})")
                        return True
            elif direction == Direction.RIGHT and enemy_y == y and enemy_x > x:
                if not any(board[y, i] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(x + 1, enemy_x)):
                    if (enemy_x - x < blast_strength):
                        # print(f"Enemy at ({enemy_x}, {enemy_y}) will hit agent at ({x}, {y})")
                        return True
        
        return False
    
    def position_in_bounds(self, obs: PommermanBoard, position: tuple):
        board = obs['board']
        y, x = position
        return 0 <= x < board.shape[1] and 0 <= y < board.shape[0]