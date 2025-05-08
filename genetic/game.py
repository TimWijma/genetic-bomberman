import pommerman
from pommerman import agents, constants
from typing import List
import numpy as np

from genetic.common_types import GameResult, PommermanBoard


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

        if custom_map is not None:
            self._set_map(custom_map)
            self.env._max_steps = 200

    def _set_map(self, custom_map: List[List[int]]):
        custom_map = np.array(custom_map, dtype=np.uint8)
        assert custom_map.shape == (11, 11), "Custom map must be of shape (11, 11)"
        
        self.env._board = custom_map

    def play_game(self, num_episodes: int = 1, render_mode: str = None) -> List[GameResult]:
        results = []

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
                positions_last_step = [agent.position for agent in self.agents]
                active_bombs = self._get_active_bombs(state)
                print(f"Alive agents: {alive_last_step}")
                print(f"Positions last step: {positions_last_step}")
                print(f"Active bombs: {active_bombs}")
                print("------------")

                state, reward, done, info = self.env.step(actions)
                
                self._calculate_kills(alive_last_step, positions_last_step, active_bombs, state)

                print([agent.is_alive for agent in self.agents])

            winners = info.get('winners', [])

            episode_results = {
                'agents': [{
                    'winner': agent.agent_id in winners,
                    'step_count': getattr(agent, 'step_count', 0),
                    'visited_tiles': getattr(agent, 'visited_tiles', set()),
                    'bombs_placed': getattr(agent, 'bombs_placed', 0),
                    'individual_index': getattr(agent, 'individual_index', -1),
                    'kills': getattr(agent, 'kills', []),
                } for agent in self.agents],
                'total_steps': self.env._step_count,
            }
            results.append(episode_results)

        self.env.close()
        return results
    
    def _get_active_bombs(self, state):
        active_bombs = {}
        
        for i, agent in enumerate(self.agents):
            if not agent.is_alive:
                continue
            
            agent_state: PommermanBoard = state[i]
            bomb_map = agent_state['bomb_blast_strength']
            bomb_positions = np.where(bomb_map > 0)
            
            for x, y in zip(bomb_positions[1], bomb_positions[0]):
                bomb_pos = (y, x)
                
                agent_bombs = getattr(agent, 'bomb_tracker', {})
                if bomb_pos in agent_bombs:
                    active_bombs[bomb_pos] = agent.agent_id
                    
        return active_bombs
    
    def _calculate_kills(self, alive_before_step, positions_before_step, active_bombs, state):
        for i, agents in enumerate(self.agents):
            if alive_before_step[i] and not agents.is_alive:
                death_pos = positions_before_step[i]
                print(f"Agent {i} killed at {death_pos} in step {self.env._step_count}")
                
                for bomb_pos, owner_id in active_bombs.items():
                    if self._is_in_blast_path(bomb_pos, death_pos, state[owner_id]['blast_strength'], state[owner_id]['board']):
                        owner_agent = self.agents[owner_id]
                        if hasattr(owner_agent, "kills"):
                            owner_agent.kills.append(i)
            
    
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
