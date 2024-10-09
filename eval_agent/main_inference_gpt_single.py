import os
from os.path import dirname, abspath, join
import json
import logging
import pathlib
import argparse
import time
import openai
import copy
import re
import math
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State


logger = logging.getLogger("agent_frame")


re_position = re.compile(r"You are at \((\d)+, (\d)+\)")


def multi_request_gpt(messages, model_name, stop=None):
    
    _MAX_API_RETRY=5    
    req_start = time.time()    
    completion = None
    
    for _ in range(_MAX_API_RETRY):
        try:
            if "o1" in model_name:
                completion = openai.ChatCompletion.create(
                            model=model_name,                            
                            messages=messages,
                            top_p=1.)
            else:
                completion = openai.ChatCompletion.create(
                            model=model_name,                            
                            messages=messages,
                            temperature=0.0,
                            top_p=1.,
                            top_logprobs=5,     # get log_prob of upto 5 tokens (5 is max)
                            stop=stop)
        except Exception as e:
            print(e)
            if "limit" in str(e):   # Let it have idle time.
                time.sleep(5)
    
    print(f"completion: {completion}")
    
    req_duration = time.time() - req_start

    return completion, req_duration



def interactive_loop_by_gpt(
    task: tasks.Task,
    env_config: Dict[str, Any],
    model_name: str,
    add_question_sent: bool = True,
) -> State:
    print(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    # reset the environment and set the prompt
    observation, state = env.reset(n_icl = env_config["n_icl"])
    
    # For tracking position and logprobs of each step,
    state.from_step_to_position = {}
    state.from_step_to_log_prob = {}
    
    init_msg = observation
    # TODO-jw: for fast inference skip printing
    print(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")

    # cur_step = 1
    
    # To get more accurate generation and its prob'
    #   Concat " The next action is '"
    question_sent = "What is the next action?"
    replaced_sent = " What is the next action? The next action is '"
    
    # Set the system prompt for GridWorld
    instruction = state.history.pop(0)["content"]
    if "o1" in model_name:
        # o1 model (o1-mini or o1-preview) doesn't support system role on 2024/09/30
        init_message = {"role": "user", "content": instruction}
    else:
        init_message = {"role": "system", "content": instruction}
    
    # Skip the unnecessary turn of Assistant (e.g., 'OK')
    if state.history[0]["content"].lower() == "ok" and state.history[0]["role"].lower() in ["assistant", "system"]:
        state.history.pop(0)

    # TODO-jw: for fast inference skip printing
    # print(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
    # cur_step += 1
    
    if state.history[0]["role"].lower() != "user":
        # Skip the first one if it is not from human
        state.history = state.history[1:]

    # Initialize msg with system prompt    
    messages = [init_message]
    
    # Concat history of moves in GridWorld
    for curr_turn in state.history:
        role = curr_turn["role"].lower()
        if role in ["user"]:
            messages += [{"role": "user", "content": curr_turn["content"]}]
        elif role in ["assistant", "system"]:
            messages += [{"role": "assistant", "content": curr_turn["content"]}]
        else:
            raise ValueError(f"[{role}] is not expected.")
        
    # To extract more clear signal, add "The next action is '"
    if add_question_sent and messages[-1]["role"] == "user" \
        and messages[-1]["content"].rfind(question_sent) + len(question_sent) == len(messages[-1]["content"]):
        messages[-1]["content"] = messages[-1]["content"].replace(question_sent, replaced_sent)
    
    # OpenAI Completion
    completion, _ = multi_request_gpt(messages=messages, model_name=model_name, stop=["'."])
    print(f"GPT_output : {completion['choices'][0]}")
    
    # Get the next move
    llm_output = completion["choices"][0]["message"]["content"]
    
    # Get the log_prob for possible moves
    # action_to_prob = {action: float("-inf") for action in env.env.game.getActions()}
    # for curr_item in completion["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
    #     token = curr_item.get("token", "").lower()
    #     if token and token in action_to_prob and action_to_prob[token] == float("-inf"):
    #         action_to_prob[token] = math.exp(curr_item.get("logprob"))
            
    # Get the current position info
    # m = re.search(re_position, messages[-1]["content"])
    # state.from_step_to_position[cur_step] = (m.group(1), m.group(2))
    # state.from_step_to_log_prob[cur_step] = action_to_prob
    
    # TODO-jw: for fast inference skip printing
    # print(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n")
    
    # environment step
    observation, state = env.step(llm_output)
    # color the state in blue
    if not state.finished:
    #     # color the observation in blue
        print(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n")

    if state.reward is not None:
        print(f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}")
    else:
        print(f"Task finished in {state.steps} steps. Success: {state.success}")

    return state


def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    
    model_name = args.model_path.split("/")[-1]
    output_path = os.path.join(args.output_dir, model_name, args.exp_config+args.exp_name, args.split, f"shot-{args.n_icl}", "inference_output")
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "inference_log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]

    print(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config['env_class'] == 'GridworldEnv':
        if "dark" in args.exp_config:
            from gridworld.gridworld_agent.envs import GridworldTextEnvDark as GridworldTextEnv
        else:
            from gridworld.gridworld_agent.envs import GridworldTextEnvSingle
        if args.data_path != None:
            print(f"args.data_path: {args.data_path}")
            env_config['env'] = GridworldTextEnvSingle(file_path = args.data_path)
        elif 'train' in args.split:
            env_config['env'] = GridworldTextEnvSingle(file_path = 'envs/gridworld/data/train_50000_hard_10x10.jsonl')
        elif 'test' in args.split:
            env_config['env'] = GridworldTextEnvSingle(file_path = 'envs/gridworld/data/inference_3000_hard_20x20.jsonl')

    # Set observation mode based on the given task
    if "dark" in args.exp_config:
        env_config["dark"] = True
        env_config["mode"] = args.dark_mode
    else:
        # basic and vanilla
        env_config["dark"] = False
        env_config["mode"] = args.basic_mode

    env_config["n_icl"] = args.n_icl

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)

    # initialize the agent
    state_list = []

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.split('.')[0])
        print(f"Existing output file found. {len(done_task_id)} tasks done.")


    if len(done_task_id) == n_tasks:
        print("All tasks done. Exiting.")
        return

    # run the loop for all tasks
    logging.info(f"Running interactive loop for {n_tasks} tasks.")
    n_todo_tasks = n_tasks - len(done_task_id)  # only run the remaining tasks

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)
        for i, task in enumerate(all_tasks):
            # Only test 10 tasks in debug mode
            if args.debug and i == 5:
                break

            # skip done tasks
            if task.task_id in done_task_id or str(task.task_id) in done_task_id:
                continue

            state = interactive_loop_by_gpt(task, env_config, args.model_path)

            state_list.append(state)
            json.dump(state.to_dict(), open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

            pbar.update(1)
        pbar.close()
    
    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")

    # calculate metrics
    reward_list = []
    success_list = []
    for state in state_list:
        if state.reward is not None:
            reward_list.append(state.reward)
        success_list.append(state.success)

    if len(reward_list) != 0:
        logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
    logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")


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
        help="output directory path",
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
        "--data_path",
        type=str,
        default=None,
        help="Evaluation data path.",
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
        default="gpt-4",
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
        "--basic_mode",
        type=int,
        default=-100,
        choices=[-1, 0, 1, 2, 3, 4, 5],
        help="Select a specific mode for basic DFS " \
        "(-100: Not Selected, " \
        "0: No info, 1: Possible Action, 2: Current Coord, " \
        "3: Current Coord + Possible Action, 4: Current Coord + Possible Coord, " \
        "5: Current Coord + Possible Coord + Possible Action)"
    )
    parser.add_argument(
        "--dark_mode",
        type=int,
        default=-100,
        choices=[-1, 0, 1, 2],
        help="Select a specific mode for basic DFS " \
        "(-100: Not Selected, " \
        "0: Possible Action, 1: Possible ACtion + Current Coord, " \
        "3: Possible Action + Current Coord + Possible Coord)"
    )
    parser.add_argument(
        "--n_icl",
        type=int,
        default=0,
    )
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)

