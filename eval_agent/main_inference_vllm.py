import os
from os.path import dirname, abspath, join
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from cv2 import log
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter
from vllm import LLM, SamplingParams

from transformers import AutoModelForCausalLM, AutoTokenizer
logger = logging.getLogger("agent_frame")


def interactive_loop(
    task: tasks.Task,
    agent: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    env_config: Dict[str, Any],
    model_path: str
) -> State:
    print(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    # reset the environment and set the prompt
    observation, state = env.reset(n_icl = env_config["n_icl"])
    
    init_msg = observation
    # TODO-jw: for fast inference skip printing
    # print(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")

    cur_step = 1
    while not state.finished:
        # TODO-jw: for fast inference skip printing
        # print(f"\n{Fore.RED}Step {cur_step}{Fore.RESET}\n")
        cur_step += 1
        # print(f"state: {vars(state)}")
        # agent act
        # # try:
        conv = get_model_adapter(model_path).get_default_conv_template(model_path)
        # print("conv: ", conv)
        roles = {"user": conv.roles[0], "assistant": conv.roles[1]}
        # print(roles)
        # break
        if roles[state.history[0]["role"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            state.history = state.history[1:]
        for j, sentence in enumerate(state.history):
            role = roles[sentence["role"]]
            assert role == conv.roles[j % 2], f"{role}{conv.roles[j % 2]}"
            conv.append_message(role, sentence["content"])
            # print("conv append_message: ", conv.get_prompt())
        prompts = conv.get_prompt()
        # prompts = f"<s> [INST] {state.history[0]["content"]} [/INST]\nOK</s><s>[INST] {state.history[1]["content"]} [/INST]\n"
        prompts +=f"{conv.roles[1]}"    # llama-3: "[/INST]\n"

        # TODO-jw: for fast inference skip printing
        # print("prompts: ", prompts)

        inputs = tokenizer(prompts, return_tensors="pt")
        input_len = inputs["input_ids"].shape[1]
        sampling_params = SamplingParams(
            best_of=1,
            temperature=0.0,
            top_p=1,
            top_k=1,
            use_beam_search=False,
            max_tokens=args.max_tokens_to_generate,
            presence_penalty=0,
            frequency_penalty=0,
            logprobs=5,
        )
        output = agent.generate([prompts], sampling_params)[0]
        
        # TODO-jw: for fast inference skip printing
        # print(output.outputs[0].text)
        
        # print(tokenizer.decode(output[0], skip_special_tokens=True))
        llm_output: str = output.outputs[0].text.rstrip()
        prob: str = str(output.outputs[0].logprobs)
        
        # TODO-jw: for fast inference skip printing
        # print("llm_output: ", llm_output)
        
        # color the action in green
        # print(f"\nLM Agent Action:\n\033[92m{action.value}\033[0m")

        # TODO-jw: for fast inference skip printing
        # print(f"\n{Fore.GREEN}{llm_output}\n{prob}{Fore.RESET}\n")
        
        # except Exception as e:
        #     print(f"Agent failed with error: {e}")
        #     state.success = False
        #     state.finished = True
        #     state.terminate_reason = "exceeding maximum input length"
        #     break
        # environment step
        observation, state = env.step(llm_output, prob)
        # color the state in blue
        if not state.finished:
            # color the observation in blue
            print(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n")

        if state.finished:
            break

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
            from gridworld.gridworld_agent.envs import GridworldTextEnv
        if args.data_path != None:
            print(f"args.data_path: {args.data_path}")
            env_config['env'] = GridworldTextEnv(file_path = args.data_path)
        elif 'train' in args.split:
            env_config['env'] = GridworldTextEnv(file_path = 'envs/gridworld/data/train_50000_hard_10x10.jsonl')
        elif 'test' in args.split:
            env_config['env'] = GridworldTextEnv(file_path = 'envs/gridworld/data/inference_3000_hard_20x20.jsonl')

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


    agent = LLM(model = args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
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

            state = interactive_loop(
                task, agent, tokenizer, env_config, args.model_path
            )

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
        "--data_path",
        type=str,
        default=None,
        help="Evaluation data path.",
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
        "--max_tokens_to_generate",
        type=int,
        default=8192,
        help="NUmber of tokens to generate via VLLM"
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

