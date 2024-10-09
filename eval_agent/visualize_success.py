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
    
    print(f"Current case \n model: {args.model_path} config : {args.exp_config} shot: {args.n_icl}")
    
    model_name = args.model_path.split("/")[-1]
    output_path = os.path.join(args.output_dir, model_name.replace('/', '_'), args.exp_config+args.exp_name, args.split, f"shot-{args.n_icl}", "inference_output")
    max_xsize, max_ysize = 0, 0
    
    result_dict = {}
    total_cnt = 0
    success_cnt = 0
    pit_cnt = 0
    invalid_cnt = 0
    max_step_cnt = 0
    for local_file in os.listdir(output_path):
        file = os.path.join(output_path, local_file)
        if os.path.isfile(file) and file.endswith(".json"):
            total_cnt+=1
            idx = file.split("/")[-1].split(".")[0]
            try:
                history = json.load(open(file))
            except:
                raise Exception(f"{idx} file has problems!!!!!!!")
            if args.n_icl > 0:
                if "llama" in model_name.lower():
                    del history["conversations"][2:2+2*args.n_icl]
                else:   # GPT
                    del history["conversations"][0:2*args.n_icl]
            xsize, ysize, goal, pit, wall, start, actions, states = start_env(history, model_name)
            if xsize > max_xsize:
                max_xsize = xsize
            if ysize > max_ysize:
                max_ysize = ysize
        else:
            continue
        
        if (xsize, ysize) not in result_dict.keys():   
            result_dict[(xsize, ysize)] = {"success": 0, "pit": 0, "max_step": 0, "invalid_action": 0, "total": 0}

        if history["meta"]["reward"] == 1:
            success_cnt +=1
            result_dict[(xsize, ysize)]["success"]+=1
            
        elif history["meta"]["reward"] == -1:
            if history["meta"]["terminate_reason"] == "pit":
                pit_cnt+=1
                result_dict[(xsize, ysize)]["pit"]+=1
            elif history["meta"]["terminate_reason"] == "invalid_action":
                invalid_cnt+=1
                result_dict[(xsize, ysize)]["invalid_action"]+=1
        elif history["meta"]["reward"] == 0:
            max_step_cnt+=1
            result_dict[(xsize, ysize)]["max_step"]+=1
        result_dict[(xsize, ysize)]["total"]+=1
    
    success_data = np.zeros((max_ysize+1, max_xsize+1))
    for key in result_dict.keys():
        success_data[max_ysize-key[1],key[0]] = result_dict[key]["success"]/result_dict[key]["total"]
    max_data = np.zeros((max_ysize+1, max_xsize+1))
    for key in result_dict.keys():
        max_data[max_ysize-key[1],key[0]] = result_dict[key]["max_step"]/result_dict[key]["total"]
    invalid_data = np.zeros((max_ysize+1, max_xsize+1))
    for key in result_dict.keys():
        invalid_data[max_ysize-key[1],key[0]] = result_dict[key]["invalid_action"]/result_dict[key]["total"]
    pit_data = np.zeros((max_ysize+1, max_xsize+1))
    for key in result_dict.keys():
        pit_data[max_ysize-key[1],key[0]] = result_dict[key]["pit"]/result_dict[key]["total"]
    # env = BaseEnv([], max_xsize, max_ysize)
    # fig, ax, _, _ = env.render()
    fig, ax = plt.subplots(figsize=(18,18))
    sns.heatmap(success_data, annot=True, fmt='.2f', ax = ax, cmap='Blues', annot_kws={"size": 12}, vmin=0, vmax=1)
    fig_path = os.path.join(output_path, f'_success_rate.png')
    ax.add_patch(
     patches.Rectangle(
         (2, args.test_ysize-args.ysize),
         args.xsize-1,
         args.ysize-1,
         edgecolor='red',
         fill=False,
         lw=2
     ) )
    success_rate = success_cnt / total_cnt
    plt.figtext(0.1, 0.02, f'{success_cnt}/{total_cnt} = {success_rate}', fontsize=15)
    plt.savefig(fig_path)
    plt.close()
    fig, ax = plt.subplots(figsize=(18,18))
    sns.heatmap(max_data, annot=True, fmt='.2f', ax = ax, cmap='YlOrRd', annot_kws={"size": 12}, vmin=0, vmax=1)
    fig_path = os.path.join(output_path, f'_max_steps_rate.png')
    ax.add_patch(
     patches.Rectangle(
         (2, args.test_ysize-args.ysize),
         args.xsize-1,
         args.ysize-1,
         edgecolor='red',
         fill=False,
         lw=2
     ) )
    max_rate = max_step_cnt/total_cnt
    plt.figtext(0.1, 0.02, f'{max_step_cnt}/{total_cnt} = {max_rate}', fontsize=15)
    plt.savefig(fig_path)
    plt.close()
    fig, ax = plt.subplots(figsize=(18,18))
    sns.heatmap(invalid_data, annot=True, fmt='.2f', ax = ax, cmap='Oranges', annot_kws={"size": 12}, vmin=0, vmax=1)
    fig_path = os.path.join(output_path, f'_invalid_rate.png')
    ax.add_patch(
     patches.Rectangle(
         (2, args.test_ysize-args.ysize),
         args.xsize-1,
         args.ysize-1,
         edgecolor='red',
         fill=False,
         lw=2
     ) )
    invalid_rate = invalid_cnt/total_cnt
    plt.figtext(0.1, 0.02, f'{invalid_cnt}/{total_cnt} = {invalid_rate}', fontsize=15)
    plt.savefig(fig_path)
    plt.close()
    fig, ax = plt.subplots(figsize=(18,18))
    sns.heatmap(pit_data, annot=True, fmt='.2f', ax = ax, cmap='Reds', annot_kws={"size": 12}, vmin=0, vmax=1)
    fig_path = os.path.join(output_path, f'_pit_rate.png')
    ax.add_patch(
     patches.Rectangle(
         (2, args.test_ysize-args.ysize),
         args.xsize-1,
         args.ysize-1,
         edgecolor='red',
         fill=False,
         lw=2
     ) )
    pit_rate = pit_cnt/total_cnt
    plt.figtext(0.1, 0.02, f'{pit_cnt}/{total_cnt} = {pit_rate}', fontsize=15)
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
        "--output_dir",
        type=str,
        default="outputs",
        help="Output directory.",
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
    parser.add_argument(
        "--xsize",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--ysize",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--test_ysize",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--n_icl",
        type=int,
        default=0
    )
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)

