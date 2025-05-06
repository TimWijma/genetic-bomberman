import pommerman
from pommerman import agents
from genetic.test_agent import TestAgent
from genetic.game import Game

def main():
    '''Simple function to bootstrap a game.'''

    game = Game('PommeFFACompetition-v0', [
        TestAgent(),
        TestAgent(),
    ])

    game.play_game(num_episodes=1)
    game.play_game(num_episodes=1)

if __name__ == '__main__':
    main()
