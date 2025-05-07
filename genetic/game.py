import pommerman
from pommerman import agents
from typing import List
import numpy as np


class Game:
    def __init__(
            self, 
            agent_list: List[agents.BaseAgent], 
            tournament_name: str = "PommeFFACompetition-v0",
            custom_map: List[List[int]] = None,
        ):
        self.env = pommerman.make(tournament_name, agent_list)
        self.custom_map = custom_map

        if custom_map is not None:
            self._set_map(custom_map)

    def _set_map(self, custom_map: List[List[int]]):
        custom_map = np.array(custom_map, dtype=np.uint8)
        assert custom_map.shape == (11, 11), "Custom map must be of shape (11, 11)"
        
        self.env._board = custom_map

    def play_game(self, num_episodes: int = 1, render_mode: str = None):
        results = []

        for i_episode in range(num_episodes):
            state = self.env.reset()

            if self.custom_map is not None:
                self._set_map(self.custom_map)

            done = False
            step_count = 0
            agent_steps = [0] * len(self.env._agents)

            while not done:
                if render_mode is not None:
                    self.env.render(mode=render_mode)
                actions = self.env.act(state)
                state, reward, done, info = self.env.step(actions)

                step_count += 1

                current_alive = set([i for i, agent in enumerate(self.env._agents) if agent.is_alive])
                for i in current_alive:
                    agent_steps[i] += 1

            episode_results = {
                'winners': info.get('winners'),
                'survival_steps': agent_steps,
                'total_steps': step_count,
            }
            results.append(episode_results)

        self.env.close()
        return results