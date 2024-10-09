import json
import argparse
from gridworld.gridworld_agent.envs.GridworldTextEnv import Gridworld
import os


def main(args):
    data = {}
    with open(args.data_path, 'r') as f:
        for idx, line in enumerate(f):
            print(f"\n\n\nidx: {idx}")
            world = json.loads(line)
            world_size = f"{world['question']['world_size_x']}x{world['question']['world_size_y']}"
            game = Gridworld()
            game.init_from_sample(world)
            #returns the optimal path for the agent
            if game.q_table == dict():
                game.getOptimalQValueTable()
            if game.l_table == dict():
                game.getOptimalLengthTable()
            
            prob = 1
            path = []
            actions = []
            pos = game.board.findPiecebyName('Player')[0].pos
            
            while pos != game.board.findPiecebyName('Goal')[0].pos:
                print(f"current pos: {pos}")
                print(f"current q_table: {game.q_table[pos]}")
                # This is crossroad
                if pos == (game.xstart, game.ystart):
                    crossroad_num = len([key for key in game.q_table[pos].keys() if game.q_table[pos][key] > 0])
                else:
                    crossroad_num = len([key for key in game.q_table[pos].keys() if game.q_table[pos][key] > 0]) - 1
                if crossroad_num > 1:
                    prob = prob / crossroad_num
                    print(f"There is crossroad! Now prob is {prob}")
                    
                action = max(game.q_table[pos], key=game.q_table[pos].get)
                path.append(pos)
                actions.append(action)
                pos = game.checkMove(pos, action)
            if world_size in data.keys():
                data[world_size]["prob_list"].append(prob)
                data[world_size]["idx"].append(idx)
            else:
                data[world_size] = {"prob_list": [prob], "idx": [idx]}
    
    for key in data.keys():
        prob_sum = 0
        for prob in data[key]["prob_list"]:
            prob_sum += prob
        data[key]["avg_prob"] = prob_sum / len(data[key]["prob_list"])
    with open(os.path.join(args.output_path, "avg_prob_" + os.path.splitext(os.path.basename(args.data_path))[0]) + ".json", "w") as f:
        json.dump(data, f)


if __name__=="__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./utils/",
    )
    args = parser.parse_args()
    main(args)