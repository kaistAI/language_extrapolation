import os
import json
import logging
import pathlib
import argparse
from typing import List, Dict, Any
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from colorama import Fore

import eval_agent.tasks as tasks
import eval_agent.agents as agents
import eval_agent.envs as envs
from eval_agent.utils.datatypes import State
from fastchat.model.model_adapter import get_conversation_template, get_model_adapter

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
    observation, state = env.reset(n_icl = 0)
    x_size = env.env.game.xsize
    y_size = env.env.game.ysize
    optimal_q_table = env.env.game.getOptimalQValueTable()
    print("optimal_q_table: ", optimal_q_table)
    evaluated_q_table = {}
    for x_index in range(x_size):
        for y_index in range(y_size):
            player = env.env.game.board.findPiecebyName('Player')[0]
            env.env.game.board.movePiece(player.id, (x_index, y_index))
            env.env.samples[env.task.session_id]["question"]["start"] = [x_index, y_index]
            observation, state = env.reset(n_icl = 0)
            evaluated_q_table[f"{x_index},{y_index}"] = {}
            
            
            init_msg = observation

            print(f"\n{Fore.YELLOW}{init_msg}{Fore.RESET}")

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
            prompts +=f"{conv.roles[1]}"
            print("prompts: ", prompts)
            inputs = tokenizer(prompts, return_tensors="pt").to(agent.device)
            actions = env.env.game.getActions()
            outputs = agent.generate(**inputs,
                return_dict_in_generate=True,
                output_scores=True,
                output_logits=True,
                max_new_tokens=30,
                num_beams = 1,
                temperature = 0.0,
                top_k = 1,
                top_p = 1.0,
                do_sample = False
            )
            transition_scores = agent.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )    
            input_length = 1 if agent.config.is_encoder_decoder else inputs.input_ids.shape[1]
            print("output decode: ", tokenizer.decode(outputs.sequences[0][input_length:]))
            output_tensor = outputs.sequences[0][input_length:]
            actions_to_tokens = [tokenizer.convert_tokens_to_ids(action) for action in actions]
            print(actions_to_tokens)
            print(output_tensor)
            text, logits, probabilities = [], [], []
            cnt =0
            tok = output_tensor[0]
            score = transition_scores[0][0]
            tok = tok.item()
            for action, token in zip(actions, actions_to_tokens):
                probability = float(outputs.scores[cnt][0][token].cpu().numpy())
                logit = outputs.logits[cnt][0][token].cpu().float()
                print(f"{tokenizer.convert_ids_to_tokens(token)} has q value {probability} and logit {logit}")
                evaluated_q_table[f"{x_index},{y_index}"][action] = probability
                
    return {'optimal':optimal_q_table,
            'evaluated':evaluated_q_table
        }


def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    
    model_name = args.model_path.split("/")[-1]
    output_path = os.path.join(args.output_dir, model_name, args.exp_config+args.exp_name, args.split, 'q_table')
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "q_table_log.txt"), mode='w')
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]

    print(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")

    if env_config['env_class'] == 'GridworldEnv':
        from gridworld.gridworld_agent.envs import GridworldTextEnv    
        if args.split == 'train':
            env_config['env'] = GridworldTextEnv(file_path = 'envs/gridworld/data/train_10000_hard.jsonl')
        elif args.split=='test':
            env_config['env'] = GridworldTextEnv(file_path = 'envs/gridworld/data/inference_3000_hard_20x20.jsonl')

    # initialize all the tasks
    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    all_tasks, n_tasks = task_class.load_tasks(args.split, args.part_num, args.part_idx)
    
    # initialize the agent
    agent = AutoModelForCausalLM.from_pretrained(args.model_path).to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith('json'):
                continue
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

            # state_list.append(state)
            json.dump(state, open(os.path.join(output_path, f"{task.task_id}.json"), 'w'), indent=4)

            pbar.update(1)
        pbar.close()
    
    logger.warning("All tasks done.")
    logger.warning(f"Output saved to {output_path}")

    # calculate metrics
    # reward_list = []
    # success_list = []
    # for state in state_list:
    #     if state.reward is not None:
    #         reward_list.append(state.reward)
    #     success_list.append(state.success)

    # if len(reward_list) != 0:
    #     logger.warning(f"Average reward: {sum(reward_list)/len(success_list):.4f}")
    # logger.warning(f"Success rate: {sum(success_list)/len(success_list):.4f}")


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
        help="output directory path",
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
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.INFO)
    elif args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.WARNING)

    main(args)

