import random
import queue
from collections import defaultdict
from typing import Dict, List, Set, Tuple, TypedDict, Any, Optional
from pommerman.agents.base_agent import BaseAgent
from gym.spaces import Discrete
from pommerman import characters, constants
import pommerman.utility as utility
from genetic.common_types import Condition, ActionType, Direction, OperatorType, PommermanBoard, Rule, ConditionType
import numpy as np

class ProcessedBoard(TypedDict):
    enemies: List[Tuple[int, int]]
    bombs: List[Tuple[int, int]]
    wood: List[Tuple[int, int]]
    is_rigid: np.ndarray
    is_wood: np.ndarray
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

        current_bombs = self._convert_bombs(np.array(obs['bomb_blast_strength']))

        my_position = tuple(obs['position'])
        board = np.array(obs['board'])
        enemies_enums = [constants.Item(e) for e in obs['enemies']]
        items, dist, prev = self._djikstra(
            board, my_position, current_bombs, enemies_enums, depth=10)

        action = self.evaluate(obs, processed_board, items, dist, prev, current_bombs, enemies_enums)

        if action is None:
            return constants.Action.Stop.value

        if action == ActionType.PLACE_BOMB and obs['ammo'] > 0:
            self.bombs_placed += 1
            return constants.Action.Bomb.value

        if action == ActionType.MOVE_TOWARDS_ENEMY:
            direction = self._near_enemy(my_position, items, dist, prev, enemies_enums, radius=np.inf)
            if direction is not None:
                return direction.value
            return constants.Action.Stop.value

        if action == ActionType.MOVE_TOWARDS_WOOD:
            direction = self._near_wood(my_position, items, dist, prev, radius=np.inf)
            if direction is not None:
                return direction.value
            return constants.Action.Stop.value

        if action == ActionType.MOVE_TOWARDS_POWERUP:
            direction = self._near_good_powerup(my_position, items, dist, prev, radius=np.inf)
            if direction is not None:
                return direction.value
            return constants.Action.Stop.value

        if action == ActionType.MOVE_TO_SAFE_SPOT:
            unsafe_directions = self._directions_in_range_of_bomb(board, my_position, current_bombs, dist)
            if unsafe_directions:
                safe_directions = self._find_safe_directions(
                    board, my_position, unsafe_directions, current_bombs, enemies_enums)
                if safe_directions:
                    return random.choice(safe_directions).value
            return constants.Action.Stop.value

        if action == ActionType.DO_NOTHING:
            return constants.Action.Stop.value
        elif action == ActionType.MOVE_UP:
            return constants.Action.Up.value
        elif action == ActionType.MOVE_DOWN:
            return constants.Action.Down.value
        elif action == ActionType.MOVE_LEFT:
            return constants.Action.Left.value
        elif action == ActionType.MOVE_RIGHT:
            return constants.Action.Right.value

        return constants.Action.Stop.value


    def reset_state(self):
        self.step_count = 0
        self.visited_tiles = set()
        self.bombs_placed = 0
        self.total_distance = 0
        self._recently_visited_positions = []
        self._prev_direction = None

    def episode_end(self, reward):
        self.average_distance = self.total_distance / self.step_count if self.step_count > 0 else 0
        super().episode_end(reward)


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
                        items,
                        dist,
                        prev,
                        current_bombs,
                        enemies_enums
                    )
                current_condition_values.append(evaluated_conditions[cond_key])

                if i < len(rule.operators):
                    operators_for_evaluation.append(rule.operators[i])

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

        return None

    def evaluate_condition(self, condition: Condition, obs: PommermanBoard, processed_board: ProcessedBoard,
                           my_position: Tuple[int, int], player_ammo: int, player_blast_strength: int, board: np.ndarray,
                           enemies_coords: List[Tuple[int, int]],
                           items: defaultdict, dist: Dict[Tuple[int, int], float], prev: Dict[Tuple[int, int], Tuple[int, int]],
                           current_bombs: List[Dict[str, Any]], enemies_enums: List[constants.Item]) -> bool:
        value = condition.value
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
        elif condition.condition_type == ConditionType.IS_SAFE_TO_PLACE_BOMB:
            result = self._maybe_bomb(player_ammo, player_blast_strength, items, dist, my_position)
        elif condition.condition_type == ConditionType.IS_WOOD_IN_RANGE:
            radius = value if value is not None else np.inf
            nearest_wood = self._nearest_position(dist, [constants.Item.Wood], items, radius)
            result = nearest_wood is not None and dist[nearest_wood] <= radius
        elif condition.condition_type == ConditionType.IS_ENEMY_IN_RANGE:
            radius = value if value is not None else np.inf
            nearest_enemy = self._nearest_position(dist, enemies_enums, items, radius)
            result = nearest_enemy is not None and dist[nearest_enemy] <= radius
        elif condition.condition_type == ConditionType.IS_POWERUP_IN_RANGE:
            radius = value if value is not None else np.inf
            powerups = [constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick]
            nearest_powerup = self._nearest_position(dist, powerups, items, radius)
            result = nearest_powerup is not None and dist[nearest_powerup] <= radius
        elif condition.condition_type == ConditionType.DISTANCE_TO_ENEMY_GT:
            if value is not None:
                nearest_enemy = self._nearest_position(dist, enemies_enums, items, np.inf)
                result = nearest_enemy is not None and dist[nearest_enemy] > value
        elif condition.condition_type == ConditionType.DISTANCE_TO_ENEMY_LT:
            if value is not None:
                nearest_enemy = self._nearest_position(dist, enemies_enums, items, np.inf)
                result = nearest_enemy is not None and dist[nearest_enemy] < value
        elif condition.condition_type == ConditionType.DISTANCE_TO_WOOD_GT:
            if value is not None:
                nearest_wood = self._nearest_position(dist, [constants.Item.Wood], items, np.inf)
                result = nearest_wood is not None and dist[nearest_wood] > value
        elif condition.condition_type == ConditionType.DISTANCE_TO_WOOD_LT:
            if value is not None:
                nearest_wood = self._nearest_position(dist, [constants.Item.Wood], items, np.inf)
                result = nearest_wood is not None and dist[nearest_wood] < value
        elif condition.condition_type == ConditionType.CAN_REACH_SAFE_SPOT:
            unsafe_directions = self._directions_in_range_of_bomb(board, my_position, current_bombs, dist)
            if unsafe_directions:
                safe_directions = self._find_safe_directions(board, my_position, unsafe_directions, current_bombs, enemies_enums)
                result = len(safe_directions) > 0
            else:
                result = True

        return result if not condition.negation else not result

    def _convert_bombs(self, bomb_map: np.ndarray) -> List[Dict[str, Any]]:
        ret = []
        locations = np.where(bomb_map > 0)
        for r, c in zip(locations[0], locations[1]):
            ret.append({
                'position': (r, c),
                'blast_strength': int(bomb_map[(r, c)])
            })
        return ret

    def _djikstra(self, board: np.ndarray, my_position: Tuple[int, int], bombs: List[Dict[str, Any]], enemies: List[constants.Item], depth: int = 10, exclude: Optional[List[constants.Item]] = None) -> Tuple[defaultdict, Dict[Tuple[int, int], float], Dict[Tuple[int, int], Optional[Tuple[int, int]]]]:
        assert (depth is not None)

        if exclude is None:
            exclude = [
                constants.Item.Fog, constants.Item.Rigid, constants.Item.Flames
            ]

        def out_of_range(p_1: Tuple[int, int], p_2: Tuple[int, int]) -> bool:
            x_1, y_1 = p_1
            x_2, y_2 = p_2
            return abs(y_2 - y_1) + abs(x_2 - x_1) > depth

        items = defaultdict(list)
        dist: Dict[Tuple[int, int], float] = {}
        prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        Q: queue.PriorityQueue = queue.PriorityQueue() # Changed to PriorityQueue

        my_x, my_y = my_position
        for r in range(max(0, my_x - depth), min(len(board), my_x + depth + 1)):
            for c in range(max(0, my_y - depth), min(len(board), my_y + depth + 1)):
                position = (r, c)
                if any([
                        out_of_range(my_position, position),
                        utility.position_in_items(board, position, exclude),
                ]):
                    continue

                prev[position] = None
                item = constants.Item(board[position])
                items[item].append(position)

                if position == my_position:
                    Q.put((0, position)) # (distance, position)
                    dist[position] = 0
                else:
                    dist[position] = np.inf


        for bomb in bombs:
            if bomb['position'] == my_position:
                items[constants.Item.Bomb].append(my_position)

        while not Q.empty():
            d, position = Q.get() # Get item with smallest distance

            # If we've found a shorter path to this position already, skip
            if d > dist[position]:
                continue

            if utility.position_is_passable(board, position, enemies):
                x, y = position
                for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    new_position = (row_delta + x, col_delta + y)
                    if new_position not in dist:
                        continue

                    new_dist = dist[position] + 1
                    if new_dist < dist[new_position]:
                        dist[new_position] = new_dist
                        prev[new_position] = position
                        Q.put((new_dist, new_position)) # Add with new distance

        return items, dist, prev

    def _directions_in_range_of_bomb(self, board: np.ndarray, my_position: Tuple[int, int], bombs: List[Dict[str, Any]], dist: Dict[Tuple[int, int], float]) -> defaultdict:
        ret = defaultdict(int)

        x, y = my_position
        for bomb in bombs:
            position = bomb['position']
            distance = dist.get(position)
            if distance is None:
                continue

            bomb_range = bomb['blast_strength']
            if distance > bomb_range:
                continue

            if my_position == position:
                for direction in [
                        constants.Action.Right, constants.Action.Left,
                        constants.Action.Up, constants.Action.Down,
                ]:
                    ret[direction] = max(ret[direction], bomb['blast_strength'])
            elif x == position[0]:
                if y < position[1]:
                    ret[constants.Action.Right] = max(
                        ret[constants.Action.Right], bomb['blast_strength'])
                else:
                    ret[constants.Action.Left] = max(ret[constants.Action.Left],
                                                     bomb['blast_strength'])
            elif y == position[1]:
                if x < position[0]:
                    ret[constants.Action.Down] = max(ret[constants.Action.Down],
                                                     bomb['blast_strength'])
                else:
                    ret[constants.Action.Up] = max(ret[constants.Action.Up],
                                                   bomb['blast_strength'])
        return ret

    def _find_safe_directions(self, board: np.ndarray, my_position: Tuple[int, int], unsafe_directions: defaultdict,
                              bombs: List[Dict[str, Any]], enemies: List[constants.Item]) -> List[constants.Action]:
        def is_stuck_direction(next_position: Tuple[int, int], bomb_range: int, next_board: np.ndarray, enemies: List[constants.Item]) -> bool:
            Q: queue.PriorityQueue = queue.PriorityQueue() # Changed to PriorityQueue
            Q.put((0, next_position))
            seen = set()

            next_x, next_y = next_position
            is_stuck = True
            while not Q.empty():
                dist_q, position = Q.get() # Get item with smallest distance
                seen.add(position)

                position_x, position_y = position
                if next_x != position_x and next_y != position_y:
                    is_stuck = False
                    break

                if dist_q > bomb_range:
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
                    Q.put((dist_from_next_pos, new_position))
            return is_stuck

        safe = []

        if len(unsafe_directions) == 4:
            next_board = board.copy()
            next_board[my_position] = constants.Item.Bomb.value

            for direction, bomb_range in unsafe_directions.items():
                next_position = utility.get_next_position(my_position, direction)
                if not utility.position_on_board(next_board, next_position) or \
                   not utility.position_is_passable(next_board, next_position, enemies):
                    continue

                if not is_stuck_direction(next_position, bomb_range, next_board, enemies):
                    return [direction]
            if not safe:
                safe = [constants.Action.Stop]
            return safe

        x, y = my_position
        disallowed = []

        for row_delta, col_delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            position = (x + row_delta, y + col_delta)
            direction = utility.get_direction(my_position, position)

            if not utility.position_on_board(board, position):
                disallowed.append(direction)
                continue

            if direction in unsafe_directions:
                continue

            if utility.position_is_passable(board, position, enemies) or utility.position_is_fog(board, position):
                safe.append(direction)

        if not safe:
            safe = [k for k in unsafe_directions if k not in disallowed]

        if not safe:
            return [constants.Action.Stop]

        return safe

    def _is_adjacent_enemy(self, items: defaultdict, dist: Dict[Tuple[int, int], float], enemies: List[constants.Item]) -> bool:
        for enemy in enemies:
            for position in items.get(enemy, []):
                if dist[position] == 1:
                    return True
        return False

    def _maybe_bomb(self, ammo: int, blast_strength: int, items: defaultdict, dist: Dict[Tuple[int, int], float], my_position: Tuple[int, int]) -> bool:
        if ammo < 1:
            return False

        x, y = my_position
        for position in items.get(constants.Item.Passage, []):
            if dist[position] == np.inf:
                continue

            if dist[position] > blast_strength:
                return True

            position_x, position_y = position
            if position_x != x and position_y != y:
                return True

        return False

    def _nearest_position(self, dist: Dict[Tuple[int, int], float], objs: List[constants.Item], items: defaultdict, radius: float = np.inf) -> Optional[Tuple[int, int]]:
        nearest = None
        dist_to = np.inf

        for obj in objs:
            for position in items.get(obj, []):
                d = dist.get(position, np.inf)
                if d <= radius and d < dist_to:
                    nearest = position
                    dist_to = d

        return nearest

    def _get_direction_towards_position(self, my_position: Tuple[int, int], position: Optional[Tuple[int, int]], prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]]) -> Optional[constants.Action]:
        if not position or position not in prev:
            return None

        next_position = position
        while prev[next_position] != my_position and prev[next_position] is not None:
            next_position = prev[next_position]
            if next_position == my_position:
                break

        if next_position == my_position:
            return constants.Action.Stop

        return utility.get_direction(my_position, next_position)

    def _near_enemy(self, my_position: Tuple[int, int], items: defaultdict, dist: Dict[Tuple[int, int], float], prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]], enemies: List[constants.Item], radius: float) -> Optional[constants.Action]:
        nearest_enemy_position = self._nearest_position(dist, enemies, items, radius)
        return self._get_direction_towards_position(my_position, nearest_enemy_position, prev)

    def _near_good_powerup(self, my_position: Tuple[int, int], items: defaultdict, dist: Dict[Tuple[int, int], float], prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]], radius: float) -> Optional[constants.Action]:
        objs = [
            constants.Item.ExtraBomb, constants.Item.IncrRange, constants.Item.Kick
        ]
        nearest_item_position = self._nearest_position(dist, objs, items, radius)
        return self._get_direction_towards_position(my_position, nearest_item_position, prev)

    def _near_wood(self, my_position: Tuple[int, int], items: defaultdict, dist: Dict[Tuple[int, int], float], prev: Dict[Tuple[int, int], Optional[Tuple[int, int]]], radius: float) -> Optional[constants.Action]:
        objs = [constants.Item.Wood]
        nearest_item_position = self._nearest_position(dist, objs, items, radius)
        return self._get_direction_towards_position(my_position, nearest_item_position, prev)

    def _filter_invalid_directions(self, board: np.ndarray, my_position: Tuple[int, int], directions: List[constants.Action], enemies: List[constants.Item]) -> List[constants.Action]:
        ret = []
        for direction in directions:
            position = utility.get_next_position(my_position, direction)
            if utility.position_on_board(board, position) and utility.position_is_passable(board, position, enemies):
                ret.append(direction)
        return ret

    def _filter_unsafe_directions(self, board: np.ndarray, my_position: Tuple[int, int], directions: List[constants.Action], bombs: List[Dict[str, Any]]) -> List[constants.Action]:
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
        ret = []
        for direction in directions:
            if not utility.get_next_position(my_position, direction) in recently_visited_positions:
                ret.append(direction)

        if not ret:
            ret = directions
        return ret

    def _is_in_blast_radius(self, obs: PommermanBoard, my_position: tuple, current_bombs: List[Dict[str, Any]]) -> bool:
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
        # Map the genetic Direction enum to pommerman.constants.Action enum for utility functions
        pommerman_action_direction = self._map_genetic_direction_to_pommerman_action(direction)
        next_pos = utility.get_next_position(my_position, pommerman_action_direction)
        return self.position_in_bounds(board, next_pos) and utility.position_is_passable(board, next_pos, [constants.Item(e) for e in enemies_coords])

    def _is_trapped(self, obs: PommermanBoard, processed_board: ProcessedBoard) -> bool:
        x, y = obs['position']
        board = obs['board']
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.position_in_bounds(board, (nx, ny)) and not processed_board['is_obstacle'][ny, nx]:
                return False
        return True

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
                    if not np.any(is_obstacle_with_bombs[enemy_y + 1:y, enemy_x]):
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

    def _distance_to_enemies(self, obs: PommermanBoard, processed_board: ProcessedBoard):
        my_position = obs['position']
        enemies = processed_board['enemies']
        if enemies:
            min_dist = float('inf')
            for enemy_pos in enemies:
                dist = self.manhattan_distance(my_position, enemy_pos)
                if dist < min_dist:
                    min_dist = dist
            self.total_distance += min_dist

    def _map_genetic_direction_to_pommerman_action(self, genetic_direction: Direction) -> constants.Action:
        if genetic_direction == Direction.UP:
            return constants.Action.Up
        elif genetic_direction == Direction.DOWN:
            return constants.Action.Down
        elif genetic_direction == Direction.LEFT:
            return constants.Action.Left
        elif genetic_direction == Direction.RIGHT:
            return constants.Action.Right
        elif genetic_direction == Direction.STOP:
            return constants.Action.Stop
        raise ValueError(f"Unknown genetic direction: {genetic_direction}")