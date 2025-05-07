from pommerman.agents.base_agent import BaseAgent
from gym.spaces import Discrete
from pommerman import constants
from genetic.common_types import Direction, PommermanBoard
import numpy as np

class TestAgent(BaseAgent):
    def act(self, obs: PommermanBoard, action_space: Discrete):
        agent_id = self._character.agent_id
        if agent_id == 0:
            print('\n'.join([
                "--- Conditions ---",
                f'is_bomb_in_range: {self._is_bomb_in_range(obs)}',
                f'is_bomb_direction(UP): {self._is_bomb_in_direction(obs, Direction.UP)}',
                f'is_bomb_direction(DOWN): {self._is_bomb_in_direction(obs, Direction.DOWN)}',
                f'is_bomb_direction(LEFT): {self._is_bomb_in_direction(obs, Direction.LEFT)}',
                f'is_bomb_direction(RIGHT): {self._is_bomb_in_direction(obs, Direction.RIGHT)}',
                f'can_move(UP): {self._can_move(obs, Direction.UP)}',
                f'can_move(DOWN): {self._can_move(obs, Direction.DOWN)}',
                f'can_move(LEFT): {self._can_move(obs, Direction.LEFT)}',
                f'can_move(RIGHT): {self._can_move(obs, Direction.RIGHT)}',
                   ]))
            if obs['step_count'] in [0, 1, 2, 3]:
                return 4
                
            return 0
        
        return action_space.sample()

    def _can_move(self, obs: PommermanBoard, direction: Direction) -> bool:
        board = obs['board']
        y, x = obs['position']
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy
        
        if not self.position_in_bounds(obs, (new_y, new_x)):
            print("Out of bounds")
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
                        print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
            elif bomb_x == x and bomb_y != y:
                min_y, max_y = min(y, bomb_y), max(y, bomb_y)
                if not any(board[i, bomb_x] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(min_y + 1, max_y)):
                    if (abs(bomb_y - y) < bomb_strenght):
                        print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
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
                        print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
            elif direction == Direction.DOWN and bomb_x == x and bomb_y > y:
                if not any(board[i, x] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(y + 1, bomb_y)):
                    if (bomb_y - y < bomb_strength):
                        print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
            elif direction == Direction.LEFT and bomb_y == y and bomb_x < x:
                if not any(board[y, i] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(bomb_x + 1, x)):
                    if (x - bomb_x < bomb_strength):
                        print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
            elif direction == Direction.RIGHT and bomb_y == y and bomb_x > x:
                if not any(board[y, i] in [constants.Item.Wood.value, constants.Item.Rigid.value] for i in range(x + 1, bomb_x)):
                    if (bomb_x - x < bomb_strength):
                        print(f"Bomb at ({bomb_x}, {bomb_y}) will hit agent at ({x}, {y})")
                        return True
                    
        return False
    
    def position_in_bounds(self, obs: PommermanBoard, position: tuple):
        board = obs['board']
        y, x = position
        return 0 <= x < board.shape[1] and 0 <= y < board.shape[0]