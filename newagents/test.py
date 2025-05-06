'''An example to show how to set up an pommerman game programmatically'''
import sys
import os
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import pommerman
from pommerman import agents
from newagents.test_agent import TestAgent

def main():
    '''Simple function to bootstrap a game.'''
    # Print all possible environments in the Pommerman registry
    print(pommerman.REGISTRY)

    starttime = time.time()
    for i in range(3000):
        # Create a set of agents (exactly four)
        agent_list = [
            # agents.SimpleAgent(),
            # agents.RandomAgent(),
            # agents.SimpleAgent(),
            TestAgent(),
            TestAgent(),
        ]

        # Make the "Free-For-All" environment using the agent list
        env = pommerman.make('PommeFFACompetition-v0', agent_list)

        # Run the episodes just like OpenAI Gym
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

    endtime = time.time()
    print('Time taken: {}'.format(endtime - starttime))


if __name__ == '__main__':
    main()
