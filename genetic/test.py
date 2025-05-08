from genetic.agent import GeneticAgent
from genetic.common_types import Rule, ConditionType, OperatorType, ActionType
from game import Game
from pommerman.agents import PlayerAgent
import pickle
import os

def main():
    '''Simple function to bootstrap a game.'''

    custom_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
        [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1],  # Player 0 starting area
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Middle row
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Player 3 starting area
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    ]

    rules = [
        Rule(
            conditions=[ConditionType.IS_WOOD_IN_RANGE],
            operators=[],
            action=ActionType.PLACE_BOMB
        ),
        Rule(
            conditions=[ConditionType.IS_BOMB_IN_RANGE],
            operators=[],
            action=ActionType.MOVE_DOWN
        )
    ]

    game = Game([
        GeneticAgent(rules),
        PlayerAgent(),
    ], 
        # tournament_name="PommeFFACompetition-v0",
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
