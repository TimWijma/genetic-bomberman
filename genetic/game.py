import pommerman
from pommerman import agents
from genetic.test_agent import TestAgent
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
        assert np.all(np.isin(custom_map, [0, 1, 2])), "Custom map must contain only values 0, 1, 2, or 3"
        
        self.env._board = custom_map

    def play_game(self, num_episodes: int = 1, render_mode: str = None):
        '''Simple function to bootstrap a game.'''
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
            print('Episode {} finished'.format(i_episode))

            if info.get('winners') is not None:
                print('Winners: {}'.format(info['winners']))

        self.env.close()