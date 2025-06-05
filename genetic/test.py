import time

from pommerman.agents.simple_agent import SimpleAgent
from genetic.agent import GeneticAgent
from genetic.common_types import Rule, ConditionType, OperatorType, ActionType
from game import Game
from pommerman.agents import PlayerAgent

def main():
    '''Simple function to bootstrap a game.'''

    custom_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
        [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1],  # Player 0 starting area
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1],  # Middle row
        [1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Player 3 starting area
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    ]

    rules = [
    ]

    game = Game([
        GeneticAgent(rules),
        GeneticAgent(rules),
        SimpleAgent(),
        SimpleAgent(),
        # GeneticAgent(rules),
    ], 
        custom_map=custom_map,
        max_steps=400
    )
    
    start_time = time.time()

    results = game.play_game(num_episodes=4, render_mode='human')

    print("Game Results:")
    for i, result in enumerate(results):
        print(f"Episode {i + 1}:")
        print(result)

    print(f"Game played in {time.time() - start_time:.2f} seconds")
    
if __name__ == '__main__':
    main()
