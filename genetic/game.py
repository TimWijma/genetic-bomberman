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

                current_step = self.env._step_count

                # Dict of bombs about to be detonated and their owners
                detonating_bombs = {}
                for agent in self.agents:
                    if hasattr(agent, "bomb_tracker"):
                        for pos, step in agent.bomb_tracker.items():
                            if step == current_step:
                                detonating_bombs[pos] = agent.agent_id

                print(f"Detonating bombs at step {current_step}: {detonating_bombs}")

                alive_last_step = [state[i]['alive'] for i in range(len(self.agents))]
                
                state, reward, done, info = self.env.step(actions)

                for i, agent in enumerate(self.agents):
                    agent_state: PommermanBoard = state[i]
                    if alive_last_step[i] and not agent_state['alive']:
                        print(f"Agent {i} killed at step {current_step}")
                        death_pos = agent_state['position']
                        
                        for bomb_pos, owner in detonating_bombs.items():
                            if self._is_in_blast_path(bomb_pos, death_pos, agent_state['blast_strength'], agent_state['board']):
                                owner_agent = self.agents[owner]

                                if hasattr(agent, "kills"):
                                    owner_agent.kills.append(agent.agent_id)

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
