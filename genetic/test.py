import pommerman
from pommerman import agents
from test_agent import TestAgent
from game import Game

def main():
    '''Simple function to bootstrap a game.'''

    custom_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
        [1, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1],  # Player 0 starting area
        [1, 0, 1, 2, 1, 2, 1, 2, 1, 0, 1],
        [1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1],
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        [1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1],  # Middle row
        [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        [1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1],
        [1, 0, 1, 2, 1, 2, 1, 2, 1, 0, 1],
        [1, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1],  # Player 3 starting area
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    ]

    game = Game([
        TestAgent(),
        TestAgent(),
    ], 
        tournament_name="PommeFFACompetition-v0",
        custom_map=custom_map,
    )

    game.play_game(num_episodes=1, render_mode='human')
    # game.play_game(num_episodes=1)

if __name__ == '__main__':
    main()
