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
        if os.path.isfile(file) and file.endswith(".json"):
            idx = file.split("/")[-1].split(".")[0]
            fig_dir = os.path.join(output_path, 'visualize',f'{idx}')
            pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
            fig_path = os.path.join(fig_dir,'path.png')
                
            if os.path.exists(fig_path):
                continue
            else:
                history = json.load(open(file))
                xsize, ysize, goal, pit, wall, start, actions, states = start_env(history)


                events = []
                for p in pit:
                    pit = RectangleBomb(loc = p)
                    events.append(pit)
                for state in states:
                    step = SquareStep(loc = state)
                    events.append(step)
                gold   = SquareGold(loc = goal)
                events.append(gold)
                    

                env = BaseEnv(wall, xsize, ysize, agent_loc=start, events = events)

                fig, _, _, _ = env.render()
                success = history["meta"]["reward"]
                if success== 0:
                    success = "Failure: max_steps"
                elif success == -1:
                    success = "Failure : pit"
                else:
                    success = "Success"
                path = ""
                for seq, actions in enumerate(actions):
                    path += f'{actions}, '
                    if seq >0 and seq%25 == 0:
                        path += "\n"
                plt.title(f'Instance {idx}')
                plt.figtext(0.1, 0.02, f'Success : {success}\nactions: {path}', fontsize=5)
                
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

