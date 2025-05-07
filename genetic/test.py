from genetic.agent import GeneticAgent
from genetic.common_types import Rule, ConditionType, OperatorType, ActionType
from game import Game
from pommerman.agents import PlayerAgent
import pickle
import os

def main():
    '''Simple function to bootstrap a game.'''

    # custom_map = [
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    #     [1, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1],  # Player 0 starting area
    #     [1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1],
    #     [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
    #     [1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1],
    #     [1, 0, 2, 2, 2, 0, 2, 2, 2, 0, 1],  # Middle row
    #     [1, 2, 1, 0, 1, 2, 1, 0, 1, 2, 1],
    #     [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1],
    #     [1, 0, 1, 0, 1, 2, 1, 0, 1, 0, 1],
    #     [1, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1],  # Player 3 starting area
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    # ]
    
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
    
    # custom_map = [
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Player 0 starting area
    #     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Middle row
    #     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #     [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    #     [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Player 3 starting area
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    # ]
    
    # print cwd
    print(os.getcwd())
    print("------")
    
    with open('./genetic/best_individual.pkl', 'rb') as f:
        best_individual = pickle.load(f)
         

    game = Game([
        GeneticAgent(rules=best_individual),
        PlayerAgent(),
    ], 
        tournament_name="PommeFFACompetition-v0",
        custom_map=custom_map,
    )

    results = game.play_game(num_episodes=1, render_mode='human')

    print("Game Results:")
    for i, result in enumerate(results):
        print(f"Episode {i + 1}:")
        print(f"  Winners: {result['winners']}")
        print(f"  Survival Steps: {result['survival_steps']}")
        print(f"  Total Steps: {result['total_steps']}")

if __name__ == '__main__':
    main()
