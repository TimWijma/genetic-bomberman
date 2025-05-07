import pommerman
from pommerman import agents
from typing import List
import numpy as np

from genetic.common_types import GameResult


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
                state, reward, done, info = self.env.step(actions)

            winners = info.get('winners', [])

            episode_results = {
                'agents': [{
                    'winner': agent.agent_id in winners,
                    'step_count': getattr(agent, 'step_count', 0),
                    'visited_tiles': getattr(agent, 'visited_tiles', set()),
                    'bombs_placed': getattr(agent, 'bombs_placed', 0),
                    'individual_index': getattr(agent, 'individual_index', -1),
                } for agent in self.agents],
                'total_steps': self.env._step_count,
            }
            results.append(episode_results)

        self.env.close()
        return results