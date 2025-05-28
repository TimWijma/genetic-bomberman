import pommerman
from pommerman import agents, constants
from typing import Dict, List, Set, Tuple
import numpy as np

from genetic.common_types import AgentResult, GameResult, PommermanBoard


class Game:
    def __init__(
            self,
            agent_list: List[agents.BaseAgent],
            tournament_name: str = "PommeFFACompetition-v0",
            custom_map: List[List[int]] = None,
            max_steps: int = 200,
        ):
        self.env = pommerman.make(tournament_name, agent_list)
        self.custom_map = custom_map
        self.agents = agent_list
        self.active_bombs = {}
        self.bomb_wood: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.kills: Dict[int, List[int]] = {} # Key: agent_id, Value: list of killed agents' ids
        self.wood_exploded: Dict[int, List[Tuple[int, int]]] = {} # Key: agent_id, Value: list of wood positions exploded by the agent

        if custom_map is not None:
            self._set_map(custom_map)
            self.env._max_steps = max_steps

    def _set_map(self, custom_map: List[List[int]]):
        custom_map = np.array(custom_map, dtype=np.uint8)
        assert custom_map.shape == (11, 11), "Custom map must be of shape (11, 11)"
        
        self.env._board = custom_map

    def play_game(self, num_episodes: int = 1, render_mode: str = None) -> List[GameResult]:
        results: List[GameResult] = []
        
        for i_episode in range(num_episodes):
            state = self.env.reset()

            if self.custom_map is not None:
                self._set_map(self.custom_map)

            for agent in self.agents:
                if hasattr(agent, 'reset_state'):
                    agent.reset_state()

            self.active_bombs = {}
            self.bomb_wood = {}
            self.kills = {}
            self.wood_exploded = {}

            done = False

            while not done:
                if render_mode is not None:
                    self.env.render(mode=render_mode)
                actions = self.env.act(state)

                alive_last_step = [True if agent.is_alive else False for agent in self.agents]
                self._get_active_bombs(state)
                self._get_wood_positions(state)

                state, reward, done, info = self.env.step(actions)
                
                self._calculate_kills(alive_last_step, state)
                self._calculate_wood_exploded(state)

            winners = info.get('winners', [])

            agent_results = [
                AgentResult(
                    agent_id=agent.agent_id,
                    agent_type=type(agent).__name__,
                    winner=agent.agent_id in winners,
                    step_count=getattr(agent, 'step_count', 0),
                    visited_tiles=len(getattr(agent, 'visited_tiles', set())),
                    bombs_placed=getattr(agent, 'bombs_placed', 0),
                    individual_index=getattr(agent, 'individual_index', -1),
                    kills=self.kills.get(agent.agent_id, []),
                    wood_exploded=len(self.wood_exploded.get(agent.agent_id, [])),
                    average_distance=getattr(agent, 'average_distance', 0.0),
                    is_alive=agent.is_alive,
                ) for agent in self.agents
            ]
            
            episode_result = GameResult(
                agent_results=agent_results,
                total_steps=self.env._step_count,
            )

            results.append(episode_result)

        self.env.close()
        return results
    
    def _get_active_bombs(self, state) -> Dict[tuple, int]:
        board = state[0]['board']
        bomb_blast_strength_map = state[0]['bomb_blast_strength']

        bomb_mask = (bomb_blast_strength_map > 0)
        current_bombs = np.argwhere(bomb_mask)
        current_bombs_set: Set[Tuple[int, int]] = set(map(tuple, current_bombs))

        bomb_item_values = {
            constants.Item.Agent0.value,
            constants.Item.Agent1.value,
            constants.Item.Agent2.value,
            constants.Item.Agent3.value,
            constants.Item.Bomb.value,
            constants.Item.Flames.value,
        }
        agent_item_values = {
            constants.Item.Agent0.value,
            constants.Item.Agent1.value,
            constants.Item.Agent2.value,
            constants.Item.Agent3.value,
        }
        
        rows, cols = board.shape
        
        bombs_to_remove = []
        for bomb_pos in self.active_bombs.keys():
            y, x = bomb_pos
            
            # Check if the bomb is still on the board
            if not (0 <= y < rows and 0 <= x < cols):
                bombs_to_remove.append(bomb_pos)
                continue
            
            # Check if the bomb is still a bomb or flames
            if board[y, x] not in bomb_item_values:
                bombs_to_remove.append(bomb_pos)

        # Remove bombs that are no longer on the board
        for bomb_pos in bombs_to_remove:
            del self.active_bombs[bomb_pos]
        
        for bomb_pos in current_bombs_set:
            if bomb_pos not in self.active_bombs:
                y, x = bomb_pos
                board_value = board[y, x]
                
                # Check if the bomb is owned by an agent
                if board_value in agent_item_values:
                    # Board value of agents is 10, 11, 12, or 13
                    # Subtract 10 to get the agent ID
                    agent_id = board_value - 10
                    self.active_bombs[bomb_pos] = agent_id  
                    
        return self.active_bombs
    
    def _get_wood_positions(self, state):
        self.bomb_wood = {}

        for bomb_pos, owner_id in self.active_bombs.items():
            self._get_bomb_wood(state, bomb_pos)
    
    def _get_bomb_wood(self, state, bomb_pos: Tuple[int, int]):
        board = state[0]['board']
        y, x = bomb_pos
        blast_strength = int(state[0]['bomb_blast_strength'][y, x])
        wood_positions = []
        rows, cols = board.shape

        # Check the four directions of the bomb
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            for i in range(1, blast_strength):
                new_y = y + dy * i
                new_x = x + dx * i

                if not (0 <= new_y < rows and 0 <= new_x < cols):
                    break

                # Check if the new position is a wood tile
                # and add it to the wood_positions list
                if board[new_y, new_x] == constants.Item.Wood.value:
                    wood_positions.append((new_y, new_x))

        # Store the wood positions in the bomb_wood dictionary
        if bomb_pos not in self.bomb_wood:
            self.bomb_wood[bomb_pos] = wood_positions
        else:
            self.bomb_wood[bomb_pos].extend(wood_positions)        
    
    def _calculate_wood_exploded(self, state):
        for bomb_pos, owner_id in self.active_bombs.items():
            # Check if the bomb exploded
            if state[owner_id]['bomb_blast_strength'][bomb_pos] == 0:
                # Get the wood positions that were destroyed by this bomb
                wood_positions = self.bomb_wood.get(bomb_pos, [])
                
                for wood_pos in wood_positions:
                    # Check if the wood position is still a wood tile
                    if state[owner_id]['board'][wood_pos] == constants.Item.Wood.value:
                        continue
                    
                    # If the wood position is not a wood tile, it was destroyed
                    if owner_id not in self.wood_exploded:
                        self.wood_exploded[owner_id] = []
                    self.wood_exploded[owner_id].append(wood_pos)
    
    def _calculate_kills(self, alive_before_step, state):
        for i, agents in enumerate(self.agents):
            # Check if the agent died this step
            if alive_before_step[i] and not agents.is_alive:
                death_pos = state[i]['position']
                
                # Check which bomb was responsible for the death
                for bomb_pos, owner_id in self.active_bombs.items():
                    if self._is_in_blast_path(bomb_pos, death_pos, state[owner_id]['blast_strength'], state[owner_id]['board']):
                        # Get the current agent's kills and append the new kill
                        agent_kills = self.kills.get(owner_id, [])
                        self.kills[owner_id] = agent_kills + [i]

    def _is_in_blast_path(self, bomb_pos, agent_pos, blast_strength, board):
        by, bx = bomb_pos
        ay, ax = agent_pos

        if bx == ax:
            # Vertical blast
            min_y, max_y = sorted([by, ay])
            if abs(ay - by) >= blast_strength:
                return False
            for y in range(min_y + 1, max_y):
                if board[y, bx] in [constants.Item.Rigid.value, constants.Item.Wood.value]:
                    return False
            return True
        elif by == ay:
            # Horizontal blast
            min_x, max_x = sorted([bx, ax])
            if abs(ax - bx) >= blast_strength:
                return False
            for x in range(min_x + 1, max_x):
                if board[by, x] in [constants.Item.Rigid.value, constants.Item.Wood.value]:
                    return False
            return True
        return False

    def position_in_bounds(self, state, position: Tuple[int, int]) -> bool:
        board = state[0]['board']
        y, x = position
        return 0 <= x < board.shape[1] and 0 <= y < board.shape[0]