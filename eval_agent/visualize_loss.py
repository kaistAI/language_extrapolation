import os
import json
import logging

import argparse
from typing import List, Dict, Any

from eval_agent.utils.utils_gridworld.env import BaseEnv
from eval_agent.utils.utils_gridworld.events import SquareGold, RectangleBomb, SquareStep
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.patches as patches


from eval_agent.utils.gridworld_visualize import *
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
logger = logging.getLogger("agent_frame")

def main(args: argparse.Namespace):
    
    # Read `trainer_state.json` for target `
    trainer_json_path = os.path.join(args.model_path, "trainer_state.json")
    with open(trainer_json_path, "r", encoding="utf-8") as f_in:
        trainer_state = json.load(f_in)
        log_history = trainer_state["log_history"]

    # Set early stop step if necessary
    early_stop_step = log_history[-1]["step"]
    if args.early_stop_step != -1:
        early_stop_step = min(early_stop_step, args.early_stop_step)

    # List up `step` and `loss`, except the very last one
    steps = []
    losses = []
    for item in log_history[:-1]:
        try:
            step = item["step"]
            if step > early_stop_step:
                break
            steps.append(step)
            losses.append(item["loss"])
        except:
            raise ValueError(f"Neither step nor loss: {item}")
    
    plt.figure(figsize=(args.xsize, args.ysize))
    
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.plot(steps, losses)

    # Set output path for plotting
    model_name = args.model_path.split("/")[-1]
    output_path = os.path.join(
        args.output_dir, 
        model_name.replace('/', '_'), 
        args.exp_config + args.exp_name, 
        args.split, 
        f"shot-{args.n_icl}", "inference_output"
    )
    loss_fig_path = os.path.join(output_path, f'_loss.png')
    plt.savefig(loss_fig_path)
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
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory.",
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
        "--model_path",
        type=str,
        default="",
        help="path of model.",
    )
    parser.add_argument(
        "--n_icl",
        type=int,
        default=0
    )
    parser.add_argument(
        "--early_stop_step",
        type=int,
        default=-1
    )
    parser.add_argument(
        "--xsize",
        type=int,
        default=10
    )
    parser.add_argument(
        "--ysize",
        type=int,
        default=10
    )
    args = parser.parse_args()
    
    main(args)

