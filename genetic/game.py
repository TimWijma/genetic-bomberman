import pommerman
from pommerman import agents, constants
from typing import Dict, List
import numpy as np

from genetic.common_types import AgentResult, GameResult, PommermanBoard


class Game:
    def __init__(
            self,
            agent_list: List[agents.BaseAgent],
            tournament_name: str = "PommeFFACompetition-v0",
            custom_map: List[List[int]] = None,
        ):
        self.env = pommerman.make(tournament_name, agent_list)
        self.custom_map = custom_map
        self.agents = agent_list
        self.active_bombs = {}
        self.kills: Dict[int, List[int]] = {} # Key: agent_id, Value: list of killed agents' ids

        if custom_map is not None:
            self._set_map(custom_map)
            self.env._max_steps = 200

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

            done = False

            while not done:
                if render_mode is not None:
                    self.env.render(mode=render_mode)
                actions = self.env.act(state)

                alive_last_step = [True if agent.is_alive else False for agent in self.agents]
                self._get_active_bombs(state)
                print(f"Active bombs: {self.active_bombs}")

                state, reward, done, info = self.env.step(actions)
                
                self._calculate_kills(alive_last_step, state)

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
                ) for agent in self.agents
            ]

            episode_result = GameResult(
                agents=agent_results,
                total_steps=self.env._step_count,
            )
            
            results.append(episode_result)

        self.env.close()
        return results
    
    def _get_active_bombs(self, state) -> Dict[tuple, int]:
        board = state[0]['board']
        bomb_map = state[0]['bomb_blast_strength']
        bomb_positions = np.where(bomb_map > 0)
        
        bombs_to_remove = []
        for bomb_pos in self.active_bombs.keys():
            y, x = bomb_pos
            
            if board[y, x] not in [
                constants.Item.Agent0.value,
                constants.Item.Agent1.value,
                constants.Item.Agent2.value,
                constants.Item.Agent3.value,
                constants.Item.Bomb.value,
                constants.Item.Flames.value,
            ]:
                bombs_to_remove.append(bomb_pos)
                
        for bomb_pos in bombs_to_remove:
            del self.active_bombs[bomb_pos]
                
        for y, x in zip(bomb_positions[0], bomb_positions[1]):
            bomb_pos = (y, x)

            if bomb_pos not in self.active_bombs:
                board_value = board[y, x]
                
                if board_value in [
                    constants.Item.Agent0.value,
                    constants.Item.Agent1.value,
                    constants.Item.Agent2.value,
                    constants.Item.Agent3.value,
                ]:
                    # Board value is 10, 11, 12, or 13
                    # Subtract 10 to get the agent ID
                    agent_id = board_value - 10
                    self.active_bombs[bomb_pos] = agent_id
                    print(f"Bomb placed by agent {agent_id} at {bomb_pos}")            

        self.active_bombs = self.active_bombs
    
    def _calculate_kills(self, alive_before_step, state):
        for i, agents in enumerate(self.agents):
            # Check if the agent died this step
            if alive_before_step[i] and not agents.is_alive:
                death_pos = state[i]['position']
                
                # Check which bomb was responsible for the death
                for bomb_pos, owner_id in self.active_bombs.items():
                    print(f"Checking bomb at {bomb_pos} placed by agent {owner_id}")
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
