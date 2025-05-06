'''An example to show how to set up an pommerman game programmatically'''
import sys
import os
import time

import pommerman
from pommerman import agents
from newagents.test_agent import TestAgent

def main():
    '''Simple function to bootstrap a game.'''
    
    agent_list = [
        TestAgent(),
        TestAgent(),
    ]

    env = pommerman.make('PommeFFACompetition-v0', agent_list)

    for i_episode in range(1):
        state = env.reset()
        done = False
        while not done:
            # env.render()
            actions = env.act(state)
            state, reward, done, info = env.step(actions)
        print('Episode {} finished'.format(i_episode))
        
        if info.get('winners') is not None:
            print('Winners: {}'.format(info['winners']))

    env.close()

if __name__ == '__main__':
    main()
