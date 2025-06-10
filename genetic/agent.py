from collections import defaultdict
import queue
import random
from typing import Any, Dict, List, Optional, Set, Tuple, TypedDict
from pommerman.agents.base_agent import BaseAgent
from gym.spaces import Discrete
from pommerman import characters, constants, utility
from genetic.common_types import Condition, ActionType, Direction, OperatorType, PommermanBoard, Rule, ConditionType
import numpy as np

class ProcessedBoard(TypedDict):
    enemies: List[Tuple[int, int]]
    bombs: List[Tuple[int, int]]
    is_rigid: np.ndarray
    is_wood: np.ndarray
    is_bomb: np.ndarray
    is_flame: np.ndarray
    is_obstacle: np.ndarray
    is_obstacle_with_bombs: np.ndarray

class GeneticAgent(BaseAgent):
    def __init__(self, rules: List[Rule], individual_index = -1, character=characters.Bomber):
        super().__init__(character)

        self.rules = rules
        self.individual_index = individual_index
        self.step_count = 0
        self.visited_tiles = set()
        self.bombs_placed = 0
        self.total_distance = 0

        self._recently_visited_positions = []
        self._recently_visited_length = 6
        
        self._prev_direction = None

    def act(self, obs: PommermanBoard, action_space: Discrete):
        self.step_count += 1
        self.visited_tiles.add(obs['position'])

        processed_board = self.process_board(obs)

        current_bombs = self.convert_bombs(np.array(obs['bomb_blast_strength']))

        my_position = tuple(obs['position'])
        board = np.array(obs['board'])
        enemies_enums = [constants.Item(e) for e in obs['enemies']] # Get enemy enums for Dijkstra
        items, dist, prev = self._djikstra(
            board, my_position, current_bombs, enemies_enums, depth=10)

        # self._distance_to_enemies(obs, processed_board)

        # action = self.evaluate(obs, processed_board)
        action = self.evaluate(obs, processed_board, items, dist, prev, current_bombs, enemies_enums)

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
        self._recently_visited_positions = [] # Reset for new episode
        self._prev_direction = None # Reset for new episode

    def episode_end(self, reward):
        self.average_distance = self.total_distance / self.step_count if self.step_count > 0 else 0

        return super().episode_end(reward)

    def process_board(self, obs: PommermanBoard) -> ProcessedBoard:
        board = obs['board']
        enemies = obs['enemies']
        my_position = obs['position']

        enemy_coords = []
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] in enemies:
                    enemy_coords.append((r, c))

        bomb_coords = []
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] == constants.Item.Bomb.value:
                    bomb_coords.append((r, c))

        wood_coords = []
        for r in range(board.shape[0]):
            for c in range(board.shape[1]):
                if board[r, c] == constants.Item.Wood.value:
                    wood_coords.append((r, c))

        is_rigid = (board == constants.Item.Rigid.value)
        is_wood = (board == constants.Item.Wood.value)
        is_obstacle = is_rigid | is_wood | (board == constants.Item.Bomb.value)
        is_obstacle_with_bombs = is_rigid | is_wood

        return ProcessedBoard(
            enemies=enemy_coords,
            bombs=bomb_coords,
            wood=wood_coords,
            is_rigid=is_rigid,
            is_wood=is_wood,
            is_obstacle=is_obstacle,
            is_obstacle_with_bombs=is_obstacle_with_bombs
        )

        # board = obs['board']
        # board_blast_strength = obs['bomb_blast_strength']

        # enemy_mask = np.zeros(board.shape, dtype=bool)
        # for enemy_object in obs['enemies']:
        #     enemy_mask |= (board == enemy_object.value)
        # enemy_positions = np.argwhere(enemy_mask)
        
        # bomb_mask = (board_blast_strength > 0)
        # bomb_positions = np.argwhere(bomb_mask)

        # is_rigid_mask = (board == constants.Item.Rigid.value)
        # is_wood_mask = (board == constants.Item.Wood.value)
        # is_flame_mask = (board == constants.Item.Flames.value)
        # is_obstacle = is_rigid_mask | is_wood_mask
        # is_bomb_mask = bomb_mask | is_flame_mask
        # is_obstacle_with_bombs = is_obstacle | is_bomb_mask
        
        # return {
        #     'enemies': [tuple(pos) for pos in enemy_positions],
        #     'bombs': [tuple(pos) for pos in bomb_positions],
        #     # 'wood': wood_positions,
        #     'is_rigid': is_rigid_mask,
        #     'is_wood': is_wood_mask,
        #     'is_bomb': is_bomb_mask,
        #     'is_flame': is_flame_mask,
        #     'is_obstacle': is_obstacle,
        #     'is_obstacle_with_bombs': is_obstacle_with_bombs,
        # }
        
    def evaluate(self, obs: PommermanBoard, processed_board: ProcessedBoard,
                 items: defaultdict, dist: Dict[Tuple[int, int], float], prev: Dict[Tuple[int, int], Tuple[int, int]],
                 current_bombs: List[Dict[str, Any]], enemies_enums: List[constants.Item]) -> Optional[ActionType]:

        my_position = tuple(obs['position'])
        player_ammo = obs['ammo']
        player_blast_strength = obs['blast_strength']
        board = obs['board']
        enemies_coords = processed_board['enemies']

        evaluated_conditions = {}

        for rule in self.rules:
            current_condition_values = []
            operators_for_evaluation = []

            # Evaluate each condition in the rule
            for i, condition in enumerate(rule.conditions):
                cond_key = (condition.condition_type, condition.negation, condition.value)
                if cond_key not in evaluated_conditions:
                    evaluated_conditions[cond_key] = self.evaluate_condition(
                        condition,
                        obs,
                        processed_board,
                        my_position,
                        player_ammo,
                        player_blast_strength,
                        board,
                        enemies_coords,
                        items, # NEW: passed from _djikstra
                        dist, # NEW: passed from _djikstra
                        prev, # NEW: passed from _djikstra
                        current_bombs, # NEW: passed from act
                        enemies_enums # NEW: passed from act
                    )
                current_condition_values.append(evaluated_conditions[cond_key])

                if i < len(rule.operators):
                    operators_for_evaluation.append(rule.operators[i])

            # Evaluate rule using AND/OR logic
            if not current_condition_values:
                continue

            result = current_condition_values[0]
            for i in range(len(operators_for_evaluation)):
                operator = operators_for_evaluation[i]
                next_val = current_condition_values[i + 1]

                if operator == OperatorType.AND:
                    result = result and next_val
                elif operator == OperatorType.OR:
                    result = result or next_val

            if result:
                return rule.action

        return None # No rule satisfied


        # Iterate through the rules and evaluate conditions
        # Return the action of the first rule that is satisfied
        for rule in self.rules:
            if not rule.conditions:
                continue

            for condition in rule.conditions:
                condition_tuple = (condition.condition_type, condition.negation)
                if condition_tuple not in evaluated_conditions:
                    result = self.evaluate_condition(obs, processed_board, condition)
                    evaluated_conditions[condition_tuple] = result

            current_condition_values = [evaluated_conditions[(cond.condition_type, cond.negation)] for cond in rule.conditions]

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

    def evaluate_condition(self, condition: Condition, obs: PommermanBoard, processed_board: ProcessedBoard,
                           my_position: Tuple[int, int], player_ammo: int, player_blast_strength: int, board: np.ndarray,
                           enemies_coords: List[Tuple[int, int]],
                           items: defaultdict, dist: Dict[Tuple[int, int], float], prev: Dict[Tuple[int, int], Tuple[int, int]],
                           current_bombs: List[Dict[str, Any]], enemies_enums: List[constants.Item]) -> bool:
        value = condition.value # Get condition value if present
        result = False

        if condition.condition_type == ConditionType.IS_IN_BLAST_RADIUS:
            result = self._is_in_blast_radius(obs, my_position, current_bombs)
        elif condition.condition_type == ConditionType.IS_ADJACENT_TO_WOOD:
            result = self._is_adjacent_wood(processed_board, my_position)
        elif condition.condition_type == ConditionType.IS_ADJACENT_TO_BOMB:
            result = self._is_adjacent_bomb(processed_board, my_position)
        elif condition.condition_type == ConditionType.CAN_MOVE_UP:
            result = self._can_move(board, my_position, Direction.UP, enemies_coords)
        elif condition.condition_type == ConditionType.CAN_MOVE_DOWN:
            result = self._can_move(board, my_position, Direction.DOWN, enemies_coords)
        elif condition.condition_type == ConditionType.CAN_MOVE_LEFT:
            result = self._can_move(board, my_position, Direction.LEFT, enemies_coords)
        elif condition.condition_type == ConditionType.CAN_MOVE_RIGHT:
            result = self._can_move(board, my_position, Direction.RIGHT, enemies_coords)
        elif condition.condition_type == ConditionType.IS_TRAPPED:
            result = self._is_trapped(obs, processed_board)
        elif condition.condition_type == ConditionType.HAS_BOMB:
            result = (player_ammo > 0)
        elif condition.condition_type == ConditionType.IS_ENEMY_IN_BLAST_RANGE:
            result = self._is_enemy_in_blast_range(obs, my_position, player_blast_strength, processed_board)
        elif condition.condition_type == ConditionType.IS_BOMB_ON_PLAYER:
            result = self._is_bomb_on_player(obs, my_position)
        elif condition.condition_type == ConditionType.IS_ENEMY_UP:
            result = self._is_enemy_in_direction(my_position, enemies_coords, Direction.UP, board, player_blast_strength, processed_board['is_obstacle_with_bombs'])
        elif condition.condition_type == ConditionType.IS_ENEMY_DOWN:
            result = self._is_enemy_in_direction(my_position, enemies_coords, Direction.DOWN, board, player_blast_strength, processed_board['is_obstacle_with_bombs'])
        elif condition.condition_type == ConditionType.IS_ENEMY_LEFT:
            result = self._is_enemy_in_direction(my_position, enemies_coords, Direction.LEFT, board, player_blast_strength, processed_board['is_obstacle_with_bombs'])
        elif condition.condition_type == ConditionType.IS_ENEMY_RIGHT:
            result = self._is_enemy_in_direction(my_position, enemies_coords, Direction.RIGHT, board, player_blast_strength, processed_board['is_obstacle_with_bombs'])
        # --- NEW CONDITION EVALUATIONS BASED ON SimpleAgent ---
        elif condition.condition_type == ConditionType.IS_SAFE_TO_PLACE_BOMB:
            result = self._maybe_bomb(player_ammo, player_blast_strength, items, dist, my_position)
        elif condition.condition_type == ConditionType.IS_WOOD_IN_RANGE:
            # Assumes 'value' is the radius. If not specified, default to inf
            radius = value if value is not None else np.inf
            nearest_wood = self._nearest_position(dist, [constants.Item.Wood], items, radius)
            result = nearest_wood is not None and dist[nearest_wood] <= radius
        elif condition.condition_type == ConditionType.IS_ENEMY_IN_RANGE:
            # Assumes 'value' is the radius. If not specified, default to inf
            radius = value if value is not None else np.inf
            nearest_enemy = self._nearest_position(dist, enemies_enums, items, radius)
            result = nearest_enemy is not None and dist[nearest_enemy] <= radius
        elif condition.condition_type == ConditionType.IS_POWERUP_IN_RANGE:
            # Assumes 'value' is the radius. If not specified, default to inf
            radius = value if value is not None else np.inf
            powerups = [constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick]
            nearest_powerup = self._nearest_position(dist, powerups, items, radius)
            result = nearest_powerup is not None and dist[nearest_powerup] <= radius
        elif condition.condition_type == ConditionType.DISTANCE_TO_ENEMY_GT:
            # Assumes 'value' is the distance threshold
            if value is not None:
                nearest_enemy = self._nearest_position(dist, enemies_enums, items, np.inf)
                result = nearest_enemy is not None and dist[nearest_enemy] > value
        elif condition.condition_type == ConditionType.DISTANCE_TO_ENEMY_LT:
            # Assumes 'value' is the distance threshold
            if value is not None:
                nearest_enemy = self._nearest_position(dist, enemies_enums, items, np.inf)
                result = nearest_enemy is not None and dist[nearest_enemy] < value
        elif condition.condition_type == ConditionType.DISTANCE_TO_WOOD_GT:
            # Assumes 'value' is the distance threshold
            if value is not None:
                nearest_wood = self._nearest_position(dist, [constants.Item.Wood], items, np.inf)
                result = nearest_wood is not None and dist[nearest_wood] > value
        elif condition.condition_type == ConditionType.DISTANCE_TO_WOOD_LT:
            # Assumes 'value' is the distance threshold
            if value is not None:
                nearest_wood = self._nearest_position(dist, [constants.Item.Wood], items, np.inf)
                result = nearest_wood is not None and dist[nearest_wood] < value
        elif condition.condition_type == ConditionType.CAN_REACH_SAFE_SPOT:
            unsafe_directions = self._directions_in_range_of_bomb(board, my_position, current_bombs, dist)
            if unsafe_directions:
                safe_directions = self._find_safe_directions(board, my_position, unsafe_directions, current_bombs, enemies_enums)
                result = len(safe_directions) > 0
            else:
                result = True # Already safe if no unsafe directions

        # --- END NEW CONDITION EVALUATIONS ---

        return result if not condition.negation else not result

    # def evaluate_condition(self, obs: PommermanBoard, processed_board: ProcessedBoard, condition: Condition) -> bool:
    #     condition_type  = condition.condition_type
    #     negation = condition.negation
    #     if condition_type == ConditionType.IS_BOMB_UP:
    #         result = self._is_bomb_in_direction(obs, processed_board, Direction.UP)
    #     elif condition_type == ConditionType.IS_BOMB_DOWN:
    #         result = self._is_bomb_in_direction(obs, processed_board, Direction.DOWN)
    #     elif condition_type == ConditionType.IS_BOMB_LEFT:
    #         result = self._is_bomb_in_direction(obs, processed_board, Direction.LEFT)
    #     elif condition_type == ConditionType.IS_BOMB_RIGHT:
    #         result = self._is_bomb_in_direction(obs, processed_board, Direction.RIGHT)
    #     elif condition_type == ConditionType.IS_WOOD_IN_RANGE:
    #         result = self._is_wood_in_range(obs, processed_board)
    #     elif condition_type == ConditionType.CAN_MOVE_UP:
    #         result = self._can_move(obs, Direction.UP)
    #     elif condition_type == ConditionType.CAN_MOVE_DOWN:
    #         result = self._can_move(obs, Direction.DOWN)
    #     elif condition_type == ConditionType.CAN_MOVE_LEFT:
    #         result = self._can_move(obs, Direction.LEFT)
    #     elif condition_type == ConditionType.CAN_MOVE_RIGHT:
    #         result = self._can_move(obs, Direction.RIGHT)
    #     elif condition_type == ConditionType.IS_TRAPPED:
    #         result = self._is_trapped(obs)
    #     elif condition_type == ConditionType.HAS_BOMB:
    #         result = obs['ammo'] > 0
    #     elif condition_type == ConditionType.IS_ENEMY_IN_RANGE:
    #         result = self._is_enemy_in_range(obs, processed_board)
    #     elif condition_type == ConditionType.IS_BOMB_ON_PLAYER:
    #         result = self._is_bomb_on_player(obs)
    #     elif condition_type == ConditionType.IS_ENEMY_UP:
    #         result = self._is_enemy_in_direction(obs, processed_board, Direction.UP)
    #     elif condition_type == ConditionType.IS_ENEMY_DOWN:
    #         result = self._is_enemy_in_direction(obs, processed_board, Direction.DOWN)
    #     elif condition_type == ConditionType.IS_ENEMY_LEFT:
    #         result = self._is_enemy_in_direction(obs, processed_board, Direction.LEFT)
    #     elif condition_type == ConditionType.IS_ENEMY_RIGHT:
    #         result = self._is_enemy_in_direction(obs, processed_board, Direction.RIGHT)

    #     if negation:
    #         result = not result

    #     return result

    def _convert_bombs(self, bomb_map: np.ndarray) -> List[Dict[str, Any]]:
        '''Flatten outs the bomb array from Pommerman's bomb_blast_strength map'''
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)])
            })
        return ret

    def _djikstra(self, board: np.ndarray, my_position: Tuple[int, int], bombs: List[Dict[str, Any]], enemies: List[constants.Item], depth: int = 10, exclude: Optional[List[constants.Item]] = None) -> Tuple[defaultdict, Dict[Tuple[int, int], float], Dict[Tuple[int, int], Optional[Tuple[int, int]]]]:
        """
        Dijkstra's algorithm to find distances and paths to all reachable cells.
        Adapted from SimpleAgent.
        """
        assert (depth is not None)

        if exclude is None:
            exclude = [
                constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
            ]

        def out_of_range(p_1: Tuple[int, int], p_2: Tuple[int, int]) -> bool:
            '''Determines if two points are out of range of each other'''
            x_1, y_1 = p_1
            x_2, y_2 = p_2
            return abs(y_2 - y_1) + abs(x_2 - x_1) > depth

        items = defaultdict(list)
        dist: Dict[Tuple[int, int], float] = {}
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        Q: queue.Queue = queue.Queue()

        my_x, my_y = my_position
        # Initialize distances and predecessors for reachable cells within depth
        for r in range(max(0, my_x - depth), min(len(board), my_x + depth + 1)): # +1 to include upper bound
            for c in range(max(0, my_y - depth), min(len(board), my_y + depth + 1)): # +1 to include upper bound
                position = (r, c)
                if any([
                        out_of_range(my_position, position),
                        utility.position_in_items(board, position, exclude), # Uses pommerman.utility
                ]):
                    continue

                prev[position] = None
                item = constants.Item(board[position])
                items[item].append(position)

                if position == my_position:
                    Q.put(position)
                    dist[position] = 0
                else:
                    dist[position] = np.inf

        for bomb in bombs:
            if bomb['position'] == my_position:
                items[constants.Item.Bomb].append(my_position)

        while not Q.empty():
            position = Q.get()

            if utility.position_is_passable(board, position, enemies): # Uses pommerman.utility
                x, y = position
                val = dist[(x, y)] + 1
                for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row_delta + x, col_delta + y)
                    if new_position not in dist: # Check if new_position is within the initialized search area
                        continue

                    if val < dist[new_position]:
                        dist[new_position] = val
                        prev[new_position] = position
                        Q.put(new_position)
                    elif (val == dist[new_position] and random.random() < .5):
                        # Break ties randomly for more varied paths
                        dist[new_position] = val
                        prev[new_position] = position


        return items, dist, prev

    def _directions_in_range_of_bomb(self, board: np.ndarray, my_position: Tuple[int, int], bombs: List[Dict[str, Any]], dist: Dict[Tuple[int, int], float]) -> defaultdict:
        """
        Returns a dictionary of directions that would put the agent in a bomb's blast range.
        Adapted from SimpleAgent.
        """
        ret = defaultdict(int)

        x, y = my_position
        for bomb in bombs:
            position = bomb['position']
            distance = dist.get(position) # Get path distance from Dijkstra's
            if distance is None: # Bomb is unreachable or not on the board
                continue

            bomb_range = bomb['blast_strength']
            if distance > bomb_range:
                continue

            if my_position == position:
                # We are on a bomb. All directions are in range of bomb.
                for direction in [
                        constants.Action.Right, constants.Action.Left,
                        constants.Action.Up, constants.Action.Down,
                ]:
                    ret[direction] = max(ret[direction], bomb['blast_strength'])
            elif x == position[0]: # Bomb is on the same row
                if y < position[1]:
                    # Bomb is right.
                    ret[constants.Action.Right] = max(
                        ret[constants.Action.Right], bomb['blast_strength'])
                else:
                    # Bomb is left.
                    ret[constants.Action.Left] = max(ret[constants.Action.Left],
                                                     bomb['blast_strength'])
            elif y == position[1]: # Bomb is on the same column
                if x < position[0]:
                    # Bomb is down.
                    ret[constants.Action.Down] = max(ret[constants.Action.Down],
                                                     bomb['blast_strength'])
                else:
                    # Bomb is up.
                    ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                                   bomb['blast_strength'])
        return ret

    def _find_safe_directions(self, board: np.ndarray, my_position: Tuple[int, int], unsafe_directions: defaultdict,
                              bombs: List[Dict[str, Any]], enemies: List[constants.Item]) -> List[constants.Action]:
        """
        Finds safe directions to move given unsafe directions.
        Adapted from SimpleAgent.
        """
        def is_stuck_direction(next_position: Tuple[int, int], bomb_range: int, next_board: np.ndarray, enemies: List[constants.Item]) -> bool:
            '''Helper function to determine if the agent's next move is possible.'''
            Q: queue.PriorityQueue = queue.PriorityQueue()
            Q.put((0, next_position))
            seen = set()

            next_x, next_y = next_position
            is_stuck = True
            while not Q.empty():
                dist, position = Q.get()
                seen.add(position)

                position_x, position_y = position
                # Check if we can move orthogonally away from the blast axis
                # This condition is crucial for determining if a path away from the bomb exists
                if next_x != position_x and next_y != position_y:
                    is_stuck = False
                    break

                if dist > bomb_range: # If we can move further than the bomb's range
                    is_stuck = False
                    break

                for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row_delta + position_x, col_delta + position_y)
                    if new_position in seen:
                        continue

                    if not utility.position_on_board(next_board, new_position):
                        continue

                    if not utility.position_is_passable(next_board, new_position, enemies):
                        continue

                    dist_from_next_pos = abs(row_delta + position_x - next_x) + abs(col_delta + position_y - next_y)
                    Q.put((dist_from_next_pos, new_position)) # Store Manhattan distance for priority queue
            return is_stuck

        safe = []

        if len(unsafe_directions) == 4: # If all 4 cardinal directions are unsafe
            next_board = board.copy()
            next_board[my_position] = constants.Item.Bomb.value # Simulate placing a bomb

            for direction, bomb_range in unsafe_directions.items():
                next_position = utility.get_next_position(my_position, direction)
                if not utility.position_on_board(next_board, next_position) or \
                   not utility.position_is_passable(next_board, next_position, enemies):
                    continue

                if not is_stuck_direction(next_position, bomb_range, next_board, enemies):
                    # We found a direction that works. Return it.
                    return [direction]
            if not safe:
                safe = [constants.Action.Stop] # If no safe path, just stop
            return safe

        x, y = my_position
        disallowed = []  # Directions that would go off the board or are already unsafe

        for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            position = (x + row_delta, y + col_delta)
            direction = utility.get_direction(my_position, position) # Uses pommerman.utility

            # Don't include any direction that will go off of the board.
            if not utility.position_on_board(board, position):
                disallowed.append(direction)
                continue

            # Don't include any direction that we know is unsafe.
            if direction in unsafe_directions:
                continue

            if utility.position_is_passable(board, position, enemies) or utility.position_is_fog(board, position):
                safe.append(direction)

        if not safe:
            # We don't have any safe directions, so return something that is allowed (not off board).
            safe = [k for k in unsafe_directions if k not in disallowed]

        if not safe:
            # We don't have ANY directions. So return the stop choice.
            return [constants.Action.Stop]

        return safe

    def _is_adjacent_enemy(self, items: defaultdict, dist: Dict[Tuple[int, int], float], enemies: List[constants.Item]) -> bool:
        """
        Checks if an enemy is in an adjacent square (distance 1).
        Adapted from SimpleAgent.
        """
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] == 1:
                    return True
        return False

    def _maybe_bomb(self, ammo: int, blast_strength: int, items: defaultdict, dist: Dict[Tuple[int, int], float], my_position: Tuple[int, int]) -> bool:
        """
        Returns whether we can safely bomb right now.
        Decides this based on:
        1. Do we have ammo?
        2. If we laid a bomb right now, will we be stuck?
        Adapted from SimpleAgent.
        """
        # Do we have ammo?
        if ammo < 1:
            return False

        # Will we be stuck?
        x, y = my_position
        # Check if there is any passage reachable outside of the bomb's blast radius and axis
        for position in items.get(constants.Item.Passage, []):
            if dist[position] == np.inf: # Unreachable passage
                continue

            # We can reach a passage that's outside of the bomb strength.
            if dist[position] > blast_strength:
                return True

            # We can reach a passage that's outside of the bomb's axis.
            position_x, position_y = position
            if position_x != x and position_y != y: # Orthogonal path to escape
                return True

        return False # No safe escape found

    def _nearest_position(self, dist: Dict[Tuple[int, int], float], objs: List[constants.Item], items: defaultdict, radius: float = np.inf) -> Optional[Tuple[int, int]]:
        """
        Finds the nearest position of a given item type within a radius.
        Adapted from SimpleAgent.
        """
        nearest = None
        dist_to = np.inf # Use infinity for initial comparison

        for obj in objs:
            for position in items.get(obj, []):
                d = dist.get(position, np.inf) # Get distance from Dijkstra's results
                if d <= radius and d < dist_to: # Use < to find strictly nearest, not <=
                    nearest = position
                    dist_to = d

        return nearest

    def _get_direction_towards_position(self, my_position: Tuple[int, int], position: Optional[Tuple[int, int]], prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]]) -> Optional[constants.Action]:
        """
        Gets the first step direction towards a target position using Dijkstra's predecessors.
        Adapted from SimpleAgent.
        """
        if not position or position not in prev:
            return None

        next_position = position
        # Traverse back until the predecessor is the current position
        while prev[next_position] != my_position and prev[next_position] is not None:
            next_position = prev[next_position]
            if next_position == my_position: # Avoid infinite loop if somehow path leads back to self
                break

        if next_position == my_position: # No valid step, target is current position
            return constants.Action.Stop

        return utility.get_direction(my_position, next_position) # Uses pommerman.utility

    def _near_enemy(self, my_position: Tuple[int, int], items: defaultdict, dist: Dict[Tuple[int, int], float], prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]], enemies: List[constants.Item], radius: float) -> Optional[constants.Action]:
        """
        Finds the direction towards the nearest enemy within a radius.
        Adapted from SimpleAgent.
        """
        nearest_enemy_position = self._nearest_position(dist, enemies, items, radius)
        return self._get_direction_towards_position(my_position, nearest_enemy_position, prev)

    def _near_good_powerup(self, my_position: Tuple[int, int], items: defaultdict, dist: Dict[Tuple[int, int], float], prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]], radius: float) -> Optional[constants.Action]:
        """
        Finds the direction towards the nearest good power-up within a radius.
        Adapted from SimpleAgent.
        """
        objs = [
            constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick
        ]
        nearest_item_position = self._nearest_position(dist, objs, items, radius)
        return self._get_direction_towards_position(my_position, nearest_item_position, prev)


    def _near_wood(self, my_position: Tuple[int, int], items: defaultdict, dist: Dict[Tuple[int, int], float], prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]], radius: float) -> Optional[constants.Action]:
        """
        Finds the direction towards the nearest wood within a radius.
        Adapted from SimpleAgent.
        """
        objs = [constants.Item.Wood]
        nearest_item_position = self._nearest_position(dist, objs, items, radius)
        return self._get_direction_towards_position(my_position, nearest_item_position, prev)

    def _filter_invalid_directions(self, board: np.ndarray, my_position: Tuple[int, int], directions: List[constants.Action], enemies: List[constants.Item]) -> List[constants.Action]:
        """
        Filters out directions that would lead off the board or into an impassable tile.
        Adapted from SimpleAgent.
        """
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(board, position) and utility.position_is_passable(board, position, enemies):
                ret.append(direction)
        return ret

    def _filter_unsafe_directions(self, board: np.ndarray, my_position: Tuple[int, int], directions: List[constants.Action], bombs: List[Dict[str, Any]]) -> List[constants.Action]:
        """
        Filters out directions that would move into a bomb's blast path.
        Adapted from SimpleAgent.
        """
        ret = []
        for direction in directions:
            x, y = utility.get_next_position(my_position, direction)
            is_bad = False
            for bomb in bombs:
                bomb_x, bomb_y = bomb['position']
                blast_strength = bomb['blast_strength']
                if (x == bomb_x and abs(bomb_y - y) <= blast_strength) or \
                   (y == bomb_y and abs(bomb_x - x) <= blast_strength):
                    is_bad = True
                    break
            if not is_bad:
                ret.append(direction)
        return ret

    def _filter_recently_visited(self, directions: List[constants.Action], my_position: Tuple[int, int],
                                 recently_visited_positions: List[Tuple[int, int]]) -> List[constants.Action]:
        """
        Filters out directions that would lead to recently visited positions.
        Adapted from SimpleAgent.
        """
        ret = []
        for direction in directions:
            if not utility.get_next_position(my_position, direction) in recently_visited_positions:
                ret.append(direction)

        if not ret:
            ret = directions # If all moves lead to recently visited, allow them.
        return ret

    def _is_in_blast_radius(self, obs: PommermanBoard, my_position: tuple, current_bombs: List[Dict[str, Any]]) -> bool:
        """
        Checks if the agent is currently in the blast radius of any active bomb.
        """
        x, y = my_position
        board = obs['board']
        for bomb in current_bombs:
            bomb_x, bomb_y = bomb['position']
            blast_strength = bomb['blast_strength']
            if (x == bomb_x and abs(bomb_y - y) < blast_strength) or \
               (y == bomb_y and abs(bomb_x - x) < blast_strength):
                return True
        return False

    def _is_adjacent_wood(self, processed_board: ProcessedBoard, my_position: tuple) -> bool:
        x, y = my_position
        wood_coords = processed_board['wood']
        for wx, wy in wood_coords:
            if self.manhattan_distance((x, y), (wx, wy)) == 1:
                return True
        return False

    def _is_adjacent_bomb(self, processed_board: ProcessedBoard, my_position: tuple) -> bool:
        x, y = my_position
        bomb_coords = processed_board['bombs']
        for bx, by in bomb_coords:
            if self.manhattan_distance((x, y), (bx, by)) == 1:
                return True
        return False

    def _can_move(self, board: np.ndarray, my_position: tuple, direction: Direction, enemies_coords: List[Tuple[int, int]]) -> bool:
        next_pos = utility.get_next_position(my_position, direction.value)
        return self.position_in_bounds(board, next_pos) and utility.position_is_passable(board, next_pos, [constants.Item(e) for e in enemies_coords])

    def _is_trapped(self, obs: PommermanBoard, processed_board: ProcessedBoard) -> bool:
        # Check if all adjacent squares are obstacles
        x, y = obs['position']
        board = obs['board']
        # This implementation can be improved using _find_safe_directions and _maybe_bomb safety checks.
        # For now, it's a basic check of adjacent obstacles.
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.position_in_bounds(board, (nx, ny)) and not processed_board['is_obstacle'][ny, nx]:
                return False  # Found a non-obstacle adjacent square
        return True # All adjacent squares are obstacles

    def _is_enemy_in_blast_range(self, obs: PommermanBoard, my_position: tuple, player_blast_strength: int, processed_board: ProcessedBoard) -> bool:
        y, x = my_position
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

    def _is_bomb_on_player(self, obs: PommermanBoard, my_position: tuple) -> bool:
        return obs['board'][my_position] == constants.Item.Bomb.value

    def _is_enemy_in_direction(self, my_position: tuple, enemies_coords: List[Tuple[int, int]], direction: Direction, board: np.ndarray, player_blast_strength: int, is_obstacle_with_bombs: np.ndarray) -> bool:
        y, x = my_position
        if direction == Direction.UP:
            for enemy_y, enemy_x in enemies_coords:
                if enemy_x == x and enemy_y < y:
                    if not np.any(is_obstacle_with_bombs[enemy_y + 1:y, enemy_x]): # Check for obstacles between agent and enemy
                        return True
        elif direction == Direction.DOWN:
            for enemy_y, enemy_x in enemies_coords:
                if enemy_x == x and enemy_y > y:
                    if not np.any(is_obstacle_with_bombs[y + 1:enemy_y, enemy_x]):
                        return True
        elif direction == Direction.LEFT:
            for enemy_y, enemy_x in enemies_coords:
                if enemy_y == y and enemy_x < x:
                    if not np.any(is_obstacle_with_bombs[enemy_y, enemy_x + 1:x]):
                        return True
        elif direction == Direction.RIGHT:
            for enemy_y, enemy_x in enemies_coords:
                if enemy_y == y and enemy_x > x:
                    if not np.any(is_obstacle_with_bombs[enemy_y, x + 1:enemy_x]):
                        return True
        return False

    def position_in_bounds(self, board: np.ndarray, position: tuple) -> bool:
        y, x = position
        return 0 <= x < board.shape[1] and 0 <= y < board.shape[0]

    def manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # This method's purpose is now largely replaced by Dijkstra's for precise distance
    # It might still be useful for total distance accumulated, but not for decision making based on distance to specific enemies
    def _distance_to_enemies(self, obs: PommermanBoard, processed_board: ProcessedBoard):
        my_position = obs['position']
        enemies = processed_board['enemies']
        if enemies:
            min_dist = float('inf')
            for enemy_pos in enemies:
                dist = self.manhattan_distance(my_position, enemy_pos)
                if dist < min_dist:
                    min_dist = dist
            # Only add to total distance if there are enemies present
            self.total_distance += min_dist


    # def _can_move(self, obs: PommermanBoard, direction: Direction) -> bool:
    #     board = obs['board']
    #     y, x = obs['position']
    #     dx, dy = direction.value
    #     new_y, new_x = y + dy, x + dx
        
    #     rows, cols = board.shape

    #     if not (0 <= new_y < rows and 0 <= new_x < cols):
    #         return False

    #     if board[new_y, new_x] == constants.Item.Passage.value:
    #         return True

    #     # TODO: Add support for powerups.

    #     return False # Rigid, Wood, Bomb, Agent, etc.
    
    # def _is_bomb_on_player(self, obs: PommermanBoard):
    #     y, x = obs['position']
    #     blast_strength = obs['bomb_blast_strength']
        
    #     return blast_strength[y, x] > 0
    
    # def _is_bomb_in_direction(self, obs: PommermanBoard, processed_board: ProcessedBoard, direction: Direction):
    #     y, x = obs['position']
        
    #     bomb_blast_strength_map = obs['bomb_blast_strength']
    #     bomb_coords = processed_board['bombs']
    #     is_obstacle = processed_board['is_obstacle']

    #     is_horizontal = (direction == Direction.LEFT or direction == Direction.RIGHT)
        
    #     for bomb_y, bomb_x in bomb_coords:
    #         # Check if the bomb is in the same row or column as the player
    #         if (is_horizontal and bomb_y != y) or (not is_horizontal and bomb_x != x):
    #             continue
            
    #         # Check if the bomb is in the specified direction
    #         if (direction == Direction.UP and bomb_y >= y) or \
    #             (direction == Direction.DOWN and bomb_y <= y) or \
    #             (direction == Direction.LEFT and bomb_x >= x) or \
    #             (direction == Direction.RIGHT and bomb_x <= x):
    #             continue    

    #         bomb_strength = bomb_blast_strength_map[bomb_y, bomb_x]
    #         if is_horizontal:
    #             dist = abs(bomb_x - x)
    #             if dist < bomb_strength:
    #                 path_slice = is_obstacle[y, min(x, bomb_x) + 1 : max(x, bomb_x)]
    #                 if not np.any(path_slice):
    #                     return True
    #         else:
    #             dist = abs(bomb_y - y)
    #             if dist < bomb_strength:
    #                 path_slice = is_obstacle[min(y, bomb_y) + 1 : max(y, bomb_y), x]
    #                 if not np.any(path_slice):
    #                     return True
    #     return False

    # def _is_wood_in_range(self, obs: PommermanBoard, processed_board: ProcessedBoard):
    #     y, x = obs['position']
    #     player_blast_strength = obs['blast_strength']
    #     board = obs['board']
    #     is_rigid = processed_board['is_rigid']
    #     rows, cols = board.shape

    #     directions = [
    #         (0, 1),   # Right
    #         (0, -1),  # Left
    #         (1, 0),   # Down
    #         (-1, 0)   # Up
    #     ]

    #     for dy, dx in directions:
    #         for i in range(1, player_blast_strength + 1):
    #             new_y = y + dy * i
    #             new_x = x + dx * i

    #             if not (0 <= new_y < rows and 0 <= new_x < cols):
    #                 break
                
    #             if is_rigid[new_y, new_x]:
    #                 break

    #             # Check if the new position is a wood tile
    #             if board[new_y, new_x] == constants.Item.Wood.value:
    #                 return True

    #     return False
    
    # def _is_trapped(self, obs: PommermanBoard):
    #     board = obs['board']
    #     y, x = obs['position']
        
    #     rows, cols = board.shape
        
    #     directions_offsets = [
    #         (0, 1),   # Right
    #         (0, -1),  # Left
    #         (1, 0),   # Down
    #         (-1, 0)   # Up
    #     ]
        
    #     for dy, dx in directions_offsets:
    #         new_y, new_x = y + dy, x + dx
            
    #         if 0 <= new_y < rows and 0 <= new_x < cols:
    #             if board[new_y, new_x] == constants.Item.Passage.value:
    #                 return False
        
    #     return True
    
    # def _is_enemy_in_direction(self, obs: PommermanBoard, processed_board: ProcessedBoard, direction: Direction):
    #     y, x = obs['position']
    #     enemy_coords = processed_board['enemies']
    #     is_obstacle_with_bombs = processed_board['is_obstacle_with_bombs']

    #     # Check if there is a direct line of sight to an enemy in the specified direction
    #     is_horizontal = (direction == Direction.LEFT or direction == Direction.RIGHT)
        
    #     for enemy_y, enemy_x in enemy_coords:
    #         if (is_horizontal and enemy_y != y) or (not is_horizontal and enemy_x != x):
    #             continue

    #         # Check if the enemy is in the specified direction
    #         if (direction == Direction.UP and enemy_y >= y) or \
    #             (direction == Direction.DOWN and enemy_y <= y) or \
    #             (direction == Direction.LEFT and enemy_x >= x) or \
    #             (direction == Direction.RIGHT and enemy_x <= x):
    #             continue

    #         if is_horizontal:
    #             min_x, max_x = min(x, enemy_x), max(x, enemy_x)
    #             if not np.any(is_obstacle_with_bombs[y, min_x + 1:max_x]):
    #                 return True
    #         else:
    #             min_y, max_y = min(y, enemy_y), max(y, enemy_y)
    #             if not np.any(is_obstacle_with_bombs[min_y + 1:max_y, enemy_x]):
    #                 return True

    #     return False
    
    # # Method that checks if there an an enemy within blast range in a given direction
    # def _is_enemy_in_range(self, obs: PommermanBoard, processed_board: ProcessedBoard):
    #     y, x = obs['position']

    #     player_blast_strength = obs['blast_strength']
    #     enemy_coords = processed_board['enemies']

    #     is_obstacle_with_bombs = processed_board['is_obstacle_with_bombs']

    #     for enemy_y, enemy_x in enemy_coords:
    #         if enemy_y == y and enemy_x != x:
    #             min_x, max_x = min(x, enemy_x), max(x, enemy_x)
    #             if not np.any(is_obstacle_with_bombs[enemy_y, min_x + 1:max_x]):
    #                 if (abs(enemy_x - x) < player_blast_strength):
    #                     return True
    #         elif enemy_x == x and enemy_y != y:
    #             min_y, max_y = min(y, enemy_y), max(y, enemy_y)
    #             if not np.any(is_obstacle_with_bombs[min_y + 1:max_y, enemy_x]):
    #                 if (abs(enemy_y - y) < player_blast_strength):
    #                     return True

    #     return False
    
    # def position_in_bounds(self, obs: PommermanBoard, position: tuple):
    #     board = obs['board']
    #     y, x = position
    #     return 0 <= x < board.shape[1] and 0 <= y < board.shape[0]

    # def manhattan_distance(self, pos1: tuple, pos2: tuple):
    #     return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # def _distance_to_enemies(self, obs: PommermanBoard, processed_board: ProcessedBoard):
    #     y, x = obs['position']
    #     enemy_coords = processed_board['enemies']

    #     distances = [self.manhattan_distance((y, x), (enemy_y, enemy_x)) for enemy_y, enemy_x in enemy_coords]
    #     average_distance = np.mean(distances)
    #     self.total_distance += average_distance