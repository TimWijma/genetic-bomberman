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
        [1, 0, 0, 0, 2, 2, 2, 0, 0, 0, 1],  # Middle row
        [1, 0, 1, 0, 1, 0, 1, 2, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 2, 0, 2, 0, 1],
        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # Player 3 starting area
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    ]

    rules = [
        Rule(
            conditions=[ConditionType.IS_WOOD_IN_RANGE],
            operators=[],
            action=ActionType.DO_NOTHING
        ),
        Rule(
            conditions=[ConditionType.IS_ENEMY_DOWN],
            operators=[],
            action=ActionType.DO_NOTHING
        )
    ]

    game = Game([
        # GeneticAgent(rules),
        # GeneticAgent(rules),
        GeneticAgent(rules),
        PlayerAgent(),
    ], 
        # tournament_name="PommeFFACompetition-v0",
        tournament_name="PommeFFACompetition-v0",
        custom_map=custom_map,
        max_steps=800
    )

    results = game.play_game(num_episodes=2, render_mode='human')

    print("Game Results:")
    for i, result in enumerate(results):
        print(f"Episode {i + 1}:")
        print(result)
if __name__ == '__main__':
    main()
