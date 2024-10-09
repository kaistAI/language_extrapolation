import argparse
import json
from eval_agent.utils.utils_gridworld.env import BaseEnv
from eval_agent.utils.utils_gridworld.events import SquareGold, RectangleBomb, SquareStep
from matplotlib import pyplot as plt

def main(args):
    with open(args.data_path, 'r') as f:
        idx = 0
        for line in f:
            print(idx)
            world = json.loads(line)

            xsize = world['question']['world_size_x']
            ysize = world['question']['world_size_y']
            walls = [tuple(wall) for wall in world['question']['wall']]
            pits = world['question']['pit']
            start = tuple(world['question']['start'])
            goal = tuple(world['question']['goal'])
            actions, states = [], []

            events = []
            for p in pits:
                pit = RectangleBomb(loc = p)
                events.append(pit)
            for state in states:
                step = SquareStep(loc = state)
                events.append(step)
            gold   = SquareGold(loc = goal)
            events.append(gold)
            print(events)

            env = BaseEnv(walls, start[0]+xsize, start[1]+ysize, agent_loc=start, events = events)
            env.render()
            
            plt.savefig(f"./{idx}.png")
            plt.close()
            idx += 1

if __name__=="__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="",
    )
    args = parser.parse_args()
    main(args)