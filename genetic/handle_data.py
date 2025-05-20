import pickle
import os
from collections import defaultdict
from typing import List

def summarize_tournament_results(filename: str):
    all_aggregated = defaultdict(lambda: defaultdict(list))  # individual_index -> metric -> list of values

    with open(filename, 'rb') as f:
        all_results = pickle.load(f)  # List[List[GameResult]]

        for tournament in all_results:
            for game_result in tournament:
                for agent_result in game_result.agent_results:
                    idx = agent_result.individual_index
                    all_aggregated[idx]['wins'].append(int(agent_result.winner))
                    all_aggregated[idx]['step_count'].append(agent_result.step_count)
                    all_aggregated[idx]['visited_tiles'].append(agent_result.visited_tiles)
                    all_aggregated[idx]['bombs_placed'].append(agent_result.bombs_placed)
                    all_aggregated[idx]['kills'].append(len(agent_result.kills))
                    all_aggregated[idx]['wood_exploded'].append(agent_result.wood_exploded)

    print("=== Agent Performance Summary ===")
    for idx in sorted(all_aggregated.keys()):
        data = all_aggregated[idx]
        games_played = len(data['wins'])
        print(f"\nAgent Individual Index: {idx}")
        print(f"  Games Played:       {games_played}")
        print(f"  Win Rate:           {sum(data['wins']) / games_played:.2f}")
        print(f"  Avg Step Count:     {sum(data['step_count']) / games_played:.2f}")
        print(f"  Avg Visited Tiles:  {sum(data['visited_tiles']) / games_played:.2f}")
        print(f"  Avg Bombs Placed:   {sum(data['bombs_placed']) / games_played:.2f}")
        print(f"  Avg Kills:          {sum(data['kills']) / games_played:.2f}")
        print(f"  Avg Wood Exploded:  {sum(data['wood_exploded']) / games_played:.2f}")


summarize_tournament_results("./genetic/results/18004/tournament/189.pkl")