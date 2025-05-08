from genetic.agent import GeneticAgent
from genetic.common_types import Rule, ConditionType, OperatorType, ActionType
from game import Game
from pommerman.agents import PlayerAgent
import pickle

def main():
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
        for agent in result['agents']:
            print(f"  Agent {agent['individual_index']}:")
            print(f"    Winner: {'Yes' if agent['winner'] else 'No'}")
            print(f"    Step Count: {agent['step_count']}")
            print(f"    Visited Tiles: {len(agent['visited_tiles'])}")
            print(f"    Bombs Placed: {agent['bombs_placed']}")
        print(f"  Total Steps in Episode: {result['total_steps']}")

if __name__ == '__main__':
    main()
