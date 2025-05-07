from pommerman.agents.base_agent import BaseAgent
from gym.spaces import Discrete
from pommerman import constants
from genetic.common_types import Direction, PommermanBoard
import numpy as np

class TestAgent(BaseAgent):
    def act(self, obs: PommermanBoard, action_space: Discrete):
        # return action_space.sample()

        agent_id = self._character.agent_id
        if agent_id == 0:
            print({
                'up': self._is_bomb_in_direction(obs, Direction.UP),
                'down': self._is_bomb_in_direction(obs, Direction.DOWN),
                'left': self._is_bomb_in_direction(obs, Direction.LEFT),
                'right': self._is_bomb_in_direction(obs, Direction.RIGHT),
                   })
            if obs['step_count'] in [0, 1, 2, 3]:
                return 4
                
            return 0
        
        return action_space.sample()

    def _can_move(self, obs: PommermanBoard, direction: Direction) -> bool:
        # Check if the direction is safe
        board = obs['board']
        y, x = obs['position']
        dx, dy = direction.value
        new_x, new_y = x + dx, y + dy
        
        # Check if the new position is within bounds
        if new_x < 0 or new_x >= board.shape[1] or new_y < 0 or new_y >= board.shape[0]:
            print("Out of bounds")
            return False

        # Check if the new position is free
        # TODO: Add support for powerups
        if board[new_y, new_x] != constants.Item.Passage.value:
            print(f"Not a passage, found {board[new_y, new_x]}")
            return False

        return True



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
        
        # Check if the new position is within bounds
        if new_x < 0 or new_x >= board.shape[1] or new_y < 0 or new_y >= board.shape[0]:
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