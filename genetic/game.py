import pommerman
from pommerman import agents
from genetic.test_agent import TestAgent
from typing import List


class Game:
    def __init__(self, tournament_name: str, agent_list: List[agents.BaseAgent]):
        self.env = pommerman.make(tournament_name, agent_list)

    def play_game(self, num_episodes: int = 1, render_mode: str = None):
        '''Simple function to bootstrap a game.'''
        for i_episode in range(num_episodes):
            state = self.env.reset()
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