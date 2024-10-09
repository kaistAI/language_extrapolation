import os
import json
import logging

import argparse
from typing import List, Dict, Any

from httpx import delete

from eval_agent.utils.utils_gridworld.env import BaseEnv
from eval_agent.utils.utils_gridworld.events import SquareGold, RectangleBomb, SquareStep
from matplotlib import pyplot as plt


from eval_agent.utils.gridworld_visualize import *
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import pathlib
logger = logging.getLogger("agent_frame")

def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    
    model_name = args.model_path.split("/")[-1]
    output_path = os.path.join(args.output_dir, model_name.replace('/', '_'), args.exp_config+args.exp_name, args.split)

    for local_file in os.listdir(output_path):

        file = os.path.join(output_path, local_file)
        idx = file.split("/")[-1].split(".")[0]
        fig_dir = os.path.join(output_path, 'visualize' ,f'{idx}')
        q_path = os.path.join(output_path, 'q_table', f'{idx}.json')
        
        if os.path.isfile(file) and file.endswith(".json") and os.path.isfile(q_path):
            pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
            fig_path = os.path.join(fig_dir,'estimated_q_value.png')
            if os.path.exists(fig_path):
                continue
            else:
                history = json.load(open(file))
                xsize, ysize, goal, pit, wall, start, actions, states = start_env(history)
                tables = json.load(open(q_path))
                evaluated = tables["evaluated"]
                fig, ax = plt.subplots()
                left = np.zeros((ysize, xsize))
                up = np.zeros((ysize, xsize))
                right = np.zeros((ysize, xsize))
                down = np.zeros((ysize, xsize))

                for key, value in evaluated.items():
                    i = key.split(',')[0]
                    j = key.split(',')[1]
                    if (int(i), int(j))in wall or (int(i), int(j)) in pit or (int(i), int(j))== goal:
                        continue
                    else:
                        up[int(j), int(i)] = value["up"]
                        right[int(j), int(i)] = value["right"]
                        down[int(j), int(i)] = value["down"]
                        left[int(j), int(i)] = value["left"]
                    # ax.text(j+0.5, i+0.5, value, ha='center', va='center', color='w', fontsize=20)
                ax.set_xlim((0, xsize))
                ax.set_xticks([i for i in range(xsize)])
                ax.set_ylim((0, ysize))
                ax.set_yticks([i for i in range(ysize)])
                tripcolor = quatromatrix(left, down, right, up, ax=ax,
                                triplotkw={"color":"k", "lw":1},
                                tripcolorkw={"cmap": "coolwarm"}) 

                ax.margins(0)
                ax.set_aspect("equal")
                fig.colorbar(tripcolor)
                plt.title(f'Instance {idx}')
                plt.savefig(fig_path)
                plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    parser.add_argument(
        "--exp_path",
        type=str,
        default="./eval_agent/configs/task",
        help="Config path of experiment.",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="gridworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory.",
    )
    parser.add_argument(
        "--part_num",
        type=int,
        default=1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--part_idx",
        type=int,
        default=-1,
        help="Evaluation part.",
    )
    parser.add_argument(
        "--agent_path",
        type=str,
        default="./eval_agent/configs/model",
        help="Config path of model.",
    )
    parser.add_argument(
        "--agent_config",
        type=str,
        default="fastchat",
        help="Config of model.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="path of model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Whether to run in debug mode (10 ex per task).",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Whether to ignore done tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Whether to run in interactive mode for demo purpose.",
    )
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)

