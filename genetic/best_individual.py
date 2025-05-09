from genetic.agent import GeneticAgent
from genetic.common_types import Rule, ConditionType, OperatorType, ActionType
from game import Game
from pommerman.agents import PlayerAgent
import pickle

def main():
    # custom_map = [
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    #     [1, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1],  # Player 0 starting area
    #     [1, 0, 1, 2, 1, 2, 1, 2, 1, 0, 1],
    #     [1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1],
    #     [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    #     [1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1],  # Middle row
    #     [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
    #     [1, 0, 2, 0, 2, 0, 2, 0, 2, 0, 1],
    #     [1, 0, 1, 2, 1, 2, 1, 2, 1, 0, 1],
    #     [1, 0, 0, 0, 2, 0, 2, 0, 0, 0, 1],  # Player 3 starting area
    #     [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Border walls
    # ]
    
    # with open('./genetic//best_individual.pkl', 'rb') as f:
    #     best_individual = pickle.load(f)

    game = Game([
GeneticAgent(rules=[
    Rule(
        conditions=[
            ConditionType.HAS_BOMB,
            ConditionType.CAN_MOVE_RIGHT
        ],
        operators=[
            OperatorType.OR
        ],
        action=ActionType.MOVE_RIGHT
    ),
    Rule(
        conditions=[
            ConditionType.CAN_MOVE_DOWN,
            ConditionType.CAN_MOVE_UP,
            ConditionType.IS_BOMB_IN_RANGE
        ],
        operators=[
            OperatorType.AND,
            OperatorType.AND
        ],
        action=ActionType.MOVE_DOWN
    ),
    Rule(
        conditions=[
            ConditionType.CAN_MOVE_DOWN,
            ConditionType.CAN_MOVE_UP,
            ConditionType.IS_BOMB_RIGHT
        ],
        operators=[
            OperatorType.AND,
            OperatorType.OR
        ],
        action=ActionType.MOVE_LEFT
    ),
    Rule(
        conditions=[
            ConditionType.IS_TRAPPED,
            ConditionType.IS_WOOD_IN_RANGE,
            ConditionType.CAN_MOVE_DOWN
        ],
        operators=[
            OperatorType.OR,
            OperatorType.OR
        ],
        action=ActionType.MOVE_DOWN
    ),
    Rule(
        conditions=[
            ConditionType.IS_BOMB_DOWN,
            ConditionType.CAN_MOVE_LEFT
        ],
        operators=[
            OperatorType.AND
        ],
        action=ActionType.MOVE_UP
    ),
    Rule(
        conditions=[
            ConditionType.IS_ENEMY_IN_RANGE,
            ConditionType.IS_ENEMY_IN_RANGE,
            ConditionType.IS_WOOD_IN_RANGE
        ],
        operators=[
            OperatorType.OR,
            OperatorType.OR
        ],
        action=ActionType.PLACE_BOMB
    ),
    Rule(
        conditions=[
            ConditionType.HAS_BOMB,
            ConditionType.CAN_MOVE_RIGHT
        ],
        operators=[
            OperatorType.AND
        ],
        action=ActionType.MOVE_UP
    ),
    Rule(
        conditions=[
            ConditionType.IS_BOMB_IN_RANGE,
            ConditionType.IS_BOMB_LEFT,
            ConditionType.CAN_MOVE_LEFT
        ],
        operators=[
            OperatorType.OR,
            OperatorType.AND
        ],
        action=ActionType.MOVE_UP
    ),
    Rule(
        conditions=[
            ConditionType.IS_BOMB_LEFT,
            ConditionType.CAN_MOVE_RIGHT,
            ConditionType.IS_ENEMY_IN_RANGE
        ],
        operators=[
            OperatorType.OR,
            OperatorType.AND
        ],
        action=ActionType.MOVE_LEFT
    ),
    Rule(
        conditions=[
            ConditionType.IS_BOMB_IN_RANGE,
            ConditionType.CAN_MOVE_DOWN,
            ConditionType.IS_BOMB_DOWN
        ],
        operators=[
            OperatorType.AND,
            OperatorType.AND
        ],
        action=ActionType.MOVE_LEFT
    )
]),
        PlayerAgent(),
    ], 
        tournament_name="PommeFFACompetition-v0",
        # custom_map=custom_map,
    )
    print("Best Individual Rules:")

    results = game.play_game(num_episodes=1, render_mode='human')

    print("Game Results:")
    for i, result in enumerate(results):
        print(f"Episode {i + 1}:")
        print(result)

if __name__ == '__main__':
    main()
