import os
import json
import random
import argparse
from gridworld.gridworld_agent.envs.GridworldTextEnv import Gridworld
import numpy as np
from transformers import AutoTokenizer


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def concat_string_list(l):
    if len(l) == 1:
        return l[0]
    elif len(l) > 1:
        return f"{', '.join([element for element in l[:-1]])}, and {l[-1]}"

def cal_previous_state(state, action):
    if action == "up":
        previous_state = (state[0], state[1]-1)
    elif action == "down":
        previous_state = (state[0], state[1]+1)
    elif action == "left":
        previous_state = (state[0]+1, state[1])
    else:
        previous_state = (state[0]-1, state[1])
    return previous_state

# Deprecated
def make_trivial_thought(state, action, world, when_thought):
    prompt = "Thought:\n"
    previous_state = cal_previous_state(state, action)
    obstacle_num = 0
    out_of_world = []
    walls = []
    pits = []
    available = []
    for direction, move in zip([(previous_state[0], previous_state[1]+1), 
                      (previous_state[0], previous_state[1]-1), 
                      (previous_state[0]-1, previous_state[1]), 
                      (previous_state[0]+1, previous_state[1])],
                         ["'up'", "'down'", "'left'", "'right'"]):
        if (direction[0]< world["question"]['start'][0] or direction[0] > world["question"]['goal'][0] or 
            direction[1]< world["question"]['start'][1] or direction[1] > world["question"]['goal'][1]):
                out_of_world.append(move)
        elif list(direction) in [item.pos for item in game.board.findPiecebyName('Wall')]:
            walls.append(move)
            # prompt += f"wall at {direction}, "
        elif list(direction) in [item.pos for item in game.board.findPiecebyName('Pit')]:
            pits.append(move)
            # prompt += f"pit at {direction}, "
        else:   # available path
            available.append(move)
    
    # This state is not crossroad
    if (when_thought == "each-crossroad" and 
        ((previous_state == ({world["question"]["start"][0]}, {world["question"]["start"][1]}) and (len(walls) + len(pits) + len(out_of_world)) > 2) or 
         (previous_state != ({world["question"]["start"][0]}, {world["question"]["start"][1]}) and (len(walls) + len(pits) + len(out_of_world)) > 1))
    ):
        return ""
    else:
        if len(walls) == 1:
            prompt += f"There is a pit at {concat_string_list(walls)}, "
        elif len(walls) > 1:
            prompt += f"There are pits at {concat_string_list(walls)}, "
        
        if len(pits) == 0:
            if len(walls) != 0:
                prompt = prompt[:-2] + f" from out current position {previous_state}"
            else:
                prompt += f"There is nothing at {concat_string_list(available)} direction"
        elif len(pits) == 1:
            prompt += f"and there is a wall at {concat_string_list(pits)} from our current position {previous_state}"
        else:
            prompt += f"and there are walls at {concat_string_list(pits)} from our current position {previous_state}"
        
        if len(out_of_world) == 1:
            prompt += f". We can't move {concat_string_list(out_of_world)} because it would take us out of the gridworld"
        elif len(out_of_world) > 1:
            prompt += f". We can't move {concat_string_list(out_of_world)} because it would take us out of the gridworld"
        prompt += "\nAction:\n"
        return prompt

def make_look_ahead_thought(state, action, game, when_thought):
    available = []
    for direction, action in zip([(state[0], state[1]+1), 
                                (state[0], state[1]-1), 
                                (state[0]-1, state[1]), 
                                (state[0]+1, state[1])],
                                ["up", "down", "left", "right"]):
        if not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
            available.append((direction, action))
    if when_thought == "each-crossroad":
        if state == game.board.findPiecebyName('Player')[0].pos:
            if len(available) < 2:
                return "" 
        elif len(available) < 3:    # up to 2 ways (1 for backward, 1 for new way) -> Not met the condition
            return ""
        cnt = 0
        prompt = "Thought:\n"
        visited = [state]
        curr = [state]
        next = []
        
        goal = game.board.findPiecebyName('Goal')[0].pos
        while goal not in visited:
            cnt +=1
            prompt += f"Step {cnt}:\n"
            for key in curr:
                for direction, action in zip([(key[0], key[1]+1), 
                                            (key[0], key[1]-1), 
                                            (key[0]-1, key[1]), 
                                            (key[0]+1, key[1])],
                                            ["up", "down", "left", "right"]):
                    if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
                        next.append((direction, action))
                    else:
                        next.append(("", "<think>"))
            for key, action in next:
                if key != "":
                    prompt += f"{key}\n{action}\n"
                else:
                    prompt += f"<think>\n"
        # print(prompt)
            visited.extend([key for key, action in next if key != ""])
            curr = [key for key, action in next if key != ""]
            next = []
            
        prompt += "Action:\n"
        return prompt
    elif when_thought == "each-crossroad-reversed":
        goal = game.board.findPiecebyName('Goal')[0].pos
        if state == game.board.findPiecebyName('Player')[0].pos:
            if len(available) < 2:
                return "" 
        elif len(available) < 3:    # up to 2 ways (1 for backward, 1 for new way) -> Not met the condition
            return ""
        cnt = 0
        prompt = "Thought:\n"
        visited = [goal]
        curr = [goal]
        next = []
        
        
        while game.board.findPiecebyName('Player')[0].pos not in visited:
            cnt +=1
            prompt += f"Step {cnt}:\n"
            for key in curr:
                for direction, action in zip([(key[0], key[1]+1), 
                                            (key[0], key[1]-1), 
                                            (key[0]-1, key[1]), 
                                            (key[0]+1, key[1])],
                                            ["down", "up", "right", "left"]):
                    if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
                        next.append((direction, action))
                    else:
                        next.append(("", "<think>"))
            for key, action in next:
                if key != "":
                    prompt += f"{key}\n{action}\n"
                else:
                    prompt += f"<think>\n"
        # print(prompt)
            visited.extend([key for key, action in next if key != ""])
            curr = [key for key, action in next if key != ""]
            next = []
            
        prompt += "Action:\n"
        return prompt


def make_look_ahead_thought_implicit(state, action, game, when_thought, args):
    available = []
    for direction, action in zip([(state[0], state[1]+1), 
                                (state[0], state[1]-1), 
                                (state[0]-1, state[1]), 
                                (state[0]+1, state[1])],
                                ["up", "down", "left", "right"]):
        if not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
            available.append((direction, action))
    if when_thought == "each-crossroad":
        if state == game.board.findPiecebyName('Player')[0].pos:
            if len(available) < 2:
                return "" 
        elif len(available) < 3:
            return ""
        cnt = 0

        if args.length_think == "len_explicit":
            explicit_prompt = make_look_ahead_thought(state, action, game, when_thought)
            num_think_tokens = len(args.tokenizer.tokenize(explicit_prompt))
            prompt = "<think>" * num_think_tokens
        else:
            prompt = "<think>"*game.l_table[state][min(game.l_table[state], key=game.l_table[state].get)]
            
        return prompt


def make_all_thought_at_start_reverse(game, args):
    start_state = game.board.findPiecebyName('Player')[0].pos
    goal = game.board.findPiecebyName('Goal')[0].pos
    cnt = 0
    prompt = "Thought:\n"
    flag = False
    visited = [goal]
    curr = [goal]
    next = []
    
    if args.cot_type == "none":
        prompt = ""
    else:
        while start_state not in visited:
            cnt +=1
            prompt += f"Step {cnt}:\n"
            for key in curr:
                for direction, action in zip([(key[0], key[1]+1), 
                                            (key[0], key[1]-1), 
                                            (key[0]-1, key[1]), 
                                            (key[0]+1, key[1])],
                                            ["down", "up", "right", "left"]):
                    if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
                        next.append((direction, action))
                    if "all_possible" in args.cot_type:
                        if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
                            prompt += f"{direction}\n{action}\n"
                        else:
                            prompt += f"{direction}\ncut\n"
                    elif "all" in args.cot_type:
                        prompt += f"{direction}\n{action}\n"

                    elif "possible" in args.cot_type:
                        if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
                            prompt += f"{direction}\n{action}\n"
                    
        # print(prompt)
            visited.extend([key for key, action in next if key != ""])
            curr = [key for key, action in next if key != ""]
            next = []
    states, actions = game.getOptimalPath()
    if "backtrack" in args.cot_type:
        prompt += f"Backtrack:\n"
        for state, action in zip(states, actions):
            prompt += f"{state}\n{action}\n"
        prompt += f"{goal}\n"
    if not args.first_only:
        prompt += f"Action:\n{actions[0]}"
        
    else:
        prompt +="Plan:\n"
        for state, action in zip(states, actions):
            prompt += f"{action}\n"
    return prompt

def make_all_thought_at_start(game, args):
    start_state = game.board.findPiecebyName('Player')[0].pos
    goal = game.board.findPiecebyName('Goal')[0].pos
    cnt = 0
    prompt = "Thought:\n"
    flag = False
    max_len = min(game.l_table[start_state].values())
    visited = [start_state]
    curr = [start_state]
    next = []
    if args.cot_type == "none":
        prompt = ""
    else:
        while goal not in visited:
            cnt +=1
            prompt += f"Step {cnt}:\n"
            for key in curr:
                for direction, action in zip([(key[0], key[1]+1), 
                                            (key[0], key[1]-1), 
                                            (key[0]-1, key[1]), 
                                            (key[0]+1, key[1])],
                                            ["up", "down", "left", "right"]):
                    if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
                        next.append((direction, action))
                    if "all_possible" in args.cot_type:
                        if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
                            prompt += f"{direction}\n{action}\n"
                        else:
                            prompt += f"{direction}\ncut\n"
                    elif "all" in args.cot_type:
                        prompt += f"{direction}\n{action}\n"

                    elif "possible" in args.cot_type:
                        if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
                            prompt += f"{direction}\n{action}\n"
                    
        # print(prompt)
            visited.extend([key for key, action in next if key != ""])
            curr = [key for key, action in next if key != ""]
            next = []
    states, actions = game.getOptimalPath()
    if "backtrack" in args.cot_type:
        prompt += f"Backtrack:\n{goal}"
        reversed_states = states[::-1]
        reversed_actions = actions[::-1]
        for state, action in zip(reversed_states, reversed_actions):
            prompt += f"{action}\n{state}\n"

    if not args.first_only:
        prompt += f"Action:\n{actions[0]}"
        
    else:
        prompt +="Plan:\n"
        for state, action in zip(states, actions):
            prompt += f"{action}\n"
    return prompt

def make_all_thought_at_start_reverse_implicit(game, args):
    cnt = 0
    visited = [game.board.findPiecebyName('Player')[0].pos]
    curr = [game.board.findPiecebyName('Player')[0].pos]
    next = []
    prompt = ""
    curr = game.board.findPiecebyName('Player')[0].pos
    if args.length_think == "len_explicit":
        explicit_prompt = make_all_thought_at_start_reverse(game, args)        
        num_think_tokens = len(args.tokenizer.tokenize(explicit_prompt))
        prompt = "<think>" * num_think_tokens -1
    else:
        prompt = "<think>"*game.l_table[curr][min(game.l_table[curr], key=game.l_table[curr].get)] -1 
        
    return prompt


def make_all_thought_at_start_implicit(start_state, game, args):
    cnt = 0
    visited = [game.board.findPiecebyName('Player')[0].pos]
    curr = [game.board.findPiecebyName('Player')[0].pos]
    next = []
    prompt = ""
    curr = game.board.findPiecebyName('Player')[0].pos
    if args.length_think == "len_explicit":
        explicit_prompt = make_all_thought_at_start(game, args)        
        num_think_tokens = len(args.tokenizer.tokenize(explicit_prompt))
        prompt = "<think>" * num_think_tokens -1
    else:
        prompt = "<think>"*game.l_table[curr][min(game.l_table[curr], key=game.l_table[curr].get)] -1
        
    return prompt


def make_kld_instance(idx, world, game, states, actions, args):
    data = {
        "id": idx,
        "conversations": [
            {
                "from": "human",
                "value": "You are given a rectangular gridworld, where you can move up, down, left, or right as long as each of your x, y coordinate is within 0 to the x, y size of the grid. If you move up, your y coordinate increases by 1. If you move down, your y coordinate decreases by 1. If you move left, your x coordinate decreases by 1. If you move right, your x coordinate increases by 1.\n\nYou will interact with the girdworld environment to reach the goal state, while avoiding the pit and the wall. You cannot move through the wall or move outside the grid. If you fall into the pit, you lose. If you reach the goal, you win. For each of your turn, you will be given the possible moves.\n\nYou should respond your move with either one of 'up', 'down', 'left', or 'right'."
            },
            {
                "from": "gpt",
                "value": "OK"
            }
        ]
    }
    world_size_x = world["question"]['world_size_x']-1
    world_size_y = world["question"]['world_size_y']-1
    goal = tuple(world["question"]['goal'])
    start = tuple(world["question"]['start'])
    wall_string = ""
    if len([item.pos for item in game.board.findPiecebyName('Wall')]) > 0:
        wall_string += "The wall is at "
        for i in [item.pos for item in game.board.findPiecebyName('Wall')]:
            if i != [item.pos for item in game.board.findPiecebyName('Wall')][-1]:
                wall_string += f"{tuple(i)}, " 
            else:
                if len([item.pos for item in game.board.findPiecebyName('Wall')]) > 1:
                    wall_string += "and "
                wall_string += f"{tuple(i)}"
    else:
        wall_string = "There is no wall"
    pit_string = ""
    if len([item.pos for item in game.board.findPiecebyName('Pit')]) > 0:
        pit_string += "The pit is at "
        for i in [item.pos for item in game.board.findPiecebyName('Pit')]:
            if i != [item.pos for item in game.board.findPiecebyName('Pit')][-1]:
                pit_string += f"{tuple(i)}, "
            else:
                if len([item.pos for item in game.board.findPiecebyName('Pit')]) > 1:
                    pit_string += "and "
                pit_string += f"{tuple(i)}"
    else:
        pit_string = "There is no pit"
    question = f"Grid is from {start} to {goal}. Goal: {goal}\nCurrent: {start}\n{pit_string}. {wall_string}.\n"

    deltas = [(0, 1, "up"), (0, -1, "down"), (-1, 0, "left"), (1, 0, "right")]

    prev_state = start
    for ord, (state, action) in enumerate(zip(states, actions)):
        beta = 0.9
        state_q_values = game.l_table[prev_state]
        actions = list(state_q_values.keys())
        q_values = list(beta ** item if item<float('inf') else  -float('inf') for item in state_q_values.values())
        q_values = softmax(q_values)
        state_q_values = {a: p for a, p in zip(actions, q_values)}
        string = ""
        if ord == 0:
            string = question

        curr_coord_str = ""
        if args.add_current_coord_for_basic:
            curr_coord_str = f"\nCurrent:\n{state}"

        # Handling possible moves 
        possible_str = ""
        if args.add_possible_action_for_basic or args.add_possible_coord_for_basic:
            for delta in deltas:
                if is_safe_move(state, delta, game) and is_effective_move(state, delta, goal, start):
                    if args.add_possible_coord_for_basic:
                        possible_str += f"\n({state[0] + delta[0]}, {state[1] + delta[1]})"
                    if args.add_possible_action_for_basic:
                        possible_str += f"\n{delta[-1]}"

            if possible_str == "":
                possible_str = "\nThere is no possible move."
            else:
                possible_str = "\nPossible:" + possible_str
        
        value_str = f"{curr_coord_str}{possible_str}".strip()
        data["conversations"].append({"from": "human", "value": f"{string}{value_str}"})

        # gpt
        if args.thought == "none":
            data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": state_q_values})
        # Thought generation
        elif args.thought == "implicit":
            if args.when_thought == "first-step":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start_implicit(state, game, args)}{action}", "q_value": state_q_values})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": state_q_values})
            if args.when_thought == "first-step-reversed":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start_reverse_implicit(game, args)}{action}", "q_value": state_q_values})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": state_q_values})
            elif args.when_thought == "each-step":
                data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought_implicit(state, action, game, args.when_thought, args)}{action}", "q_value": state_q_values})
            else:   # each_crossroad
                data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought_implicit(state, action, game, args.when_thought, args)}{action}", "q_value": state_q_values})

        else:
            if args.when_thought == "first-step":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start(game, args)}", "q_value": state_q_values})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": state_q_values})
            elif args.when_thought == "first-step-reversed":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start_reverse(game, args)}", "q_value": state_q_values})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": state_q_values})
            elif args.when_thought == "each-step":
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}", "q_value": state_q_values})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}", "q_value": state_q_values})
            else:   # each-crossroad
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}", "q_value": state_q_values})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}", "q_value": state_q_values})
            
        prev_state = state
        if ord == 0 and args.first_only:
            return data

    # print(data)
    return data

def make_kld_dark_instance(idx, world, game, states, actions, args):
    data = {
        "id": idx,
        "conversations": [
            {
                "from": "human",
                "value": "You are given a rectangular gridworld, where you can move up, down, left, or right.\n\nYou should respond your move with either one of 'up', 'down', 'left', or 'right' until you find the goal."
            },
            {
                "from": "gpt",
                "value": "OK"
            }
        ]
    }
    
    goal = tuple(world["question"]['goal'])
    start = tuple(world["question"]['start'])
    # Making World Description
    
    question = f"Grid is from {start} to {goal}. Goal: {goal}\nCurrent: {start}\n"
    
    prev_state = start
    for ord, (state, action) in enumerate(zip(states, actions)):
        beta = 0.9
        state_q_values = game.l_table[prev_state]
        actions = list(state_q_values.keys())
        q_values = list(beta ** item if item<float('inf') else  -float('inf') for item in state_q_values.values())
        q_values = softmax(q_values)
        state_q_values = {a: p for a, p in zip(actions, q_values)}
        string = ""
        if ord ==0:
            string = question
        possible : str = ""
        if (state[0], state[1]+1) not in [item.pos for item in game.board.findPiecebyName('Wall')] and (state[0], state[1]+1) not in [item.pos for item in game.board.findPiecebyName('Pit')] and state[1]+1 <= goal[1]:
            possible += '\nup'
        if (state[0], state[1]-1) not in [item.pos for item in game.board.findPiecebyName('Wall')] and (state[0], state[1]-1) not in [item.pos for item in game.board.findPiecebyName('Pit')] and state[1]-1 >= start[1]:
            possible += '\ndown'
        if (state[0]-1, state[1]) not in [item.pos for item in game.board.findPiecebyName('Wall')] and (state[0]-1, state[1]) not in [item.pos for item in game.board.findPiecebyName('Pit')] and state[0]-1 >= start[0]:
            possible += '\nleft'
        if (state[0]+1, state[1]) not in [item.pos for item in game.board.findPiecebyName('Wall')] and (state[0]+1, state[1]) not in [item.pos for item in game.board.findPiecebyName('Pit')] and state[0]+1 <= goal[0]:
            possible += '\nright'
        if possible == "":
            possible = "There is no possible move."
        data["conversations"].append({"from": "human", "value": f"{string}Possible:{possible}"})

        # gpt
        if args.thought == "none":
            data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": state_q_values})
        # Thought generation
        else:
            if args.when_thought == "first-step":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start(game, args)}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": state_q_values})
            elif args.when_thought == "first-step-reversed":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start_reverse(game, args)}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": state_q_values})
            elif args.when_thought == "each-step":
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}", "q_value": state_q_values})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}", "q_value": state_q_values})
            else:   # each-crossroad
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}", "q_value": state_q_values})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}", "q_value": state_q_values})


            
        prev_state = state
    print(data)
    return data


def make_dfs_instance(idx, world, game, states, actions, values, args):
    # Task Description
    data = {
        "id": idx,
        "conversations": [
            {
                "from": "human",
                "value": "You are given a rectangular gridworld, where you can move up, down, left, or right as long as each of your x, y coordinate is within 0 to the x, y size of the grid. If you move up, your y coordinate increases by 1. If you move down, your y coordinate decreases by 1. If you move left, your x coordinate decreases by 1. If you move right, your x coordinate increases by 1.\n\nYou will interact with the girdworld environment to reach the goal state, while avoiding the pit and the wall. You cannot move through the wall or move outside the grid. If you fall into the pit, you lose. If you reach the goal, you win. For each of your turn, you will be given the possible moves.\n\nYou should respond your move with either one of 'up', 'down', 'left', or 'right'."
            },
            {
                "from": "gpt",
                "value": "OK"
            }
        ]
    }
    
    goal = tuple(world["question"]['goal'])
    start = tuple(world["question"]['start'])
    wall_string = ""
    if len([item.pos for item in game.board.findPiecebyName('Wall')]) > 0:
        wall_string += "The wall is at "
        for i in [item.pos for item in game.board.findPiecebyName('Wall')]:
            if i != [item.pos for item in game.board.findPiecebyName('Wall')][-1]:
                wall_string += f"{tuple(i)}, " 
            else:
                if len([item.pos for item in game.board.findPiecebyName('Wall')]) > 1:
                    wall_string += "and "
                wall_string += f"{tuple(i)}"
    else:
        wall_string = "There is no wall"
    pit_string = ""
    if len([item.pos for item in game.board.findPiecebyName('Pit')]) > 0:
        pit_string += "The pit is at "
        for i in [item.pos for item in game.board.findPiecebyName('Pit')]:
            if i != [item.pos for item in game.board.findPiecebyName('Pit')][-1]:
                pit_string += f"{tuple(i)}, "
            else:
                if len([item.pos for item in game.board.findPiecebyName('Pit')]) > 1:
                    pit_string += "and "
                pit_string += f"{tuple(i)}"
    else:
        pit_string = "There is no pit"
    # Making World Description
    question = f"Grid is from {start} to {goal}. Goal: {goal}\nCurrent: {start}\n{pit_string}. {wall_string}.\n"
    
    deltas = [(0, 1, "up"), (0, -1, "down"), (-1, 0, "left"), (1, 0, "right")]

    for ord, (state, action) in enumerate(zip(states, actions)):
        # human
        string = ""
        if ord == 0:
            string = question

        # Handling current coordinate information
        curr_coord_str = ""
        if args.add_current_coord_for_basic:
            curr_coord_str = f"\nCurrent:\n{state}"

        # Handling possible moves 
        possible_str = ""
        if args.add_possible_action_for_basic or args.add_possible_coord_for_basic:
            for delta in deltas:
                if is_safe_move(state, delta, game) and is_effective_move(state, delta, goal, start):
                    if args.add_possible_coord_for_basic:
                        possible_str += f"\n({state[0] + delta[0]}, {state[1] + delta[1]})"
                    if args.add_possible_action_for_basic:
                        possible_str += f"\n{delta[-1]}"

            if possible_str == "":
                possible_str = "\nThere is no possible move."
            else:
                possible_str = "\nPossible:" + possible_str
        
        value_str = f"{curr_coord_str}{possible_str}".strip()
        data["conversations"].append({"from": "human", "value": f"{string}{value_str}"})

        # gpt
        if args.thought == "none":
            data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": values[ord]})
        # Thought generation
        else:
            if args.when_thought == "first-step":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start(game, args)}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": values[ord]})
            elif args.when_thought == "first-step-reversed":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start_reverse(game, args)}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": values[ord]})
            elif args.when_thought == "each-step":
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}", "q_value": values[ord]})
            else:   # each-crossroad
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}", "q_value": values[ord]})

    return data


def is_safe_move(state, delta, game):
    if (state[0] + delta[0], state[1] + delta[1]) in [item.pos for item in game.board.findPiecebyName('Wall')]:
        return False
    if (state[0] + delta[0], state[1] + delta[1]) in [item.pos for item in game.board.findPiecebyName('Pit')]:
        return False
    return True


def is_effective_move(state, delta, goal, start):
    next_x, next_y = (state[0] + delta[0], state[1] + delta[1])
    if start[0] <= next_x <= goal[0] and start[1] <= next_y <= goal[1]:
        return True
    else:
        return False


def make_dfs_dark_instance(idx, world, game, states, actions, values, args):
        # Task Description
    data = {
        "id": idx,
        "conversations": [
            {
                "from": "human",
                "value": "You are given a rectangular gridworld, where you can move up, down, left, or right.\n\nYou should respond your move with either one of 'up', 'down', 'left', or 'right' until you find the goal."
            },
            {
                "from": "gpt",
                "value": "OK"
            }
        ]
    }

    goal = tuple(world["question"]['goal'])
    start = tuple(world["question"]['start'])
    # Making World Description  
    question = f"Grid is from {start} to {goal}. Goal: {goal}\nCurrent: {start}\n"
    
    deltas = [(0, 1, "up"), (0, -1, "down"), (-1, 0, "left"), (1, 0, "right")]

    for ord, (state, action) in enumerate(zip(states, actions)):
        # human
        string = ""
        if ord == 0:
            string = question
        else:
            # Question includes the coordinate info.
            if args.add_current_coord_for_dark:
                string += f"Current:\n({state[0]}, {state[1]})\n"

        possible : str = ""
        for delta in deltas:
            if is_safe_move(state, delta, game) and is_effective_move(state, delta, goal, start):
                possible += (f"\n({state[0] + delta[0]}, {state[1] + delta[1]})\n{delta[-1]}" if args.add_possible_coord_for_dark else f"\n{delta[-1]}")

        if not possible:
            possible = "\nThere is no possible move."

        data["conversations"].append({"from": "human", "value": f"{string}Possible:{possible}"})

        # gpt
        if args.thought == "none":
            data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": values[ord]})
        # Thought generation
        else:
            if args.when_thought == "first-step":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start(game, args)}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": values[ord]})
            elif args.when_thought == "first-step-reversed":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start_reverse(game, args)}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}", "q_value": values[ord]})
            elif args.when_thought == "each-step":
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}", "q_value": values[ord]})
            else:   # each-crossroad
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}", "q_value": values[ord]})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}", "q_value": values[ord]})

    return data

# Derprecated
def make_instance(idx, world, states, actions, args):
    # Task Description
    data = {
        "id": idx,
        "conversations": [
            {
                "from": "human",
                "value": "You are given a rectangular gridworld, where you can move up, down, left, or right as long as each of your x, y coordinate is within 0 to the x, y size of the grid. If you move up, your y coordinate increases by 1. If you move down, your y coordinate decreases by 1. If you move left, your x coordinate decreases by 1. If you move right, your x coordinate increases by 1.\n\nYou will interact with the girdworld environment to reach the goal state, while avoiding the pit and the wall. You cannot move through the wall or move outside the grid. If you fall into the pit, you lose. If you reach the goal, you win. For each of your turn, you will be given the possible moves.\n\nYou should respond your move with either one of 'up', 'down', 'left', or 'right'."
            },
            {
                "from": "gpt",
                "value": "OK"
            }
        ]
    }
    
    world_size_x = world["question"]['world_size_x']-1
    world_size_y = world["question"]['world_size_y']-1
    goal = tuple(world["question"]['goal'])
    start = tuple(world["question"]['start'])
    wall_string = ""
    if len([item.pos for item in game.board.findPiecebyName('Wall')]) > 0:
        wall_string += "The wall is at "
        for i in [item.pos for item in game.board.findPiecebyName('Wall')]:
            if i != [item.pos for item in game.board.findPiecebyName('Wall')][-1]:
                wall_string += f"{tuple(i)}, " 
            else:
                if len([item.pos for item in game.board.findPiecebyName('Wall')]) > 1:
                    wall_string += "and "
                wall_string += f"{tuple(i)}"
    else:
        wall_string = "There is no wall"
    pit_string = ""
    if len([item.pos for item in game.board.findPiecebyName('Pit')]) > 0:
        pit_string += "The pit is at "
        for i in [item.pos for item in game.board.findPiecebyName('Pit')]:
            if i != [item.pos for item in game.board.findPiecebyName('Pit')][-1]:
                pit_string += f"{tuple(i)}, "
            else:
                if len([item.pos for item in game.board.findPiecebyName('Pit')]) > 1:
                    pit_string += "and "
                pit_string += f"{tuple(i)}"
    else:
        pit_string = "There is no pit"
    # Making World Description
    question = f"Grid is from {start} to {goal}. Goal: {goal}\nCurrent: {start}\n{pit_string}. {wall_string}.\n"
    data["conversations"].append({"from": "human", "value": question})
    
    for ord, (state, action) in enumerate(zip(states, actions)):
        # gpt
        if args.thought == "none":
            data["conversations"].append({"from": "gpt", "value": f"{action}"})
        # Thought generation
        else:
            if args.when_thought == "first-step":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start(game, args)}"})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}"})
            elif args.when_thought == "first-step-reversed":
                if ord == 0:
                    data["conversations"].append({"from": "gpt", "value": f"{make_all_thought_at_start_reverse(game, args)}"})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{action}"}) 
            elif args.when_thought == "each-step":
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}"})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}"})
            else:   # each-crossroad
                if args.thought == "trivial":
                    data["conversations"].append({"from": "gpt", "value": f"{make_trivial_thought(state, action, world, args.when_thought)}{action}"})
                else:
                    data["conversations"].append({"from": "gpt", "value": f"{make_look_ahead_thought(state, action, game, args.when_thought)}{action}"})
        # human
        if ord < len(states)-1:
            data["conversations"].append({"from": "human", "value": f"Current: {state}\nup: ({state[0]}, {state[1]+1})\ndown: ({state[0]}, {state[1]-1})\nleft: ({state[0]-1}, {state[1]})\nright: ({state[0]+1}, {state[1]})."})

    return data


def main(args):
    value_adding_options = ""
    # Check whether task is basic or dark
    if args.task.endswith("dark"):

        if args.dark_mode == -1:
            value_adding_options = [arg for arg in vars(args) if "add_" in arg and getattr(args, arg)]
            if value_adding_options:
                value_adding_options = "-" + "-".join(value_adding_options)
            else:
                value_adding_options = ""
        
        elif args.dark_mode != -1:
            if args.dark_mode == 0:
                args.add_current_coord_for_dark = False
                args.add_possible_coord_for_dark = False
            elif args.dark_mode == 1:
                args.add_current_coord_for_dark = True
                args.add_possible_coord_for_dark = False
            elif args.dark_mode == 2:
                args.add_current_coord_for_dark = True
                args.add_possible_coord_for_dark = True
            else:
                raise ValueError(f"{args.dark_mode} NOT supported")
            
            value_adding_options = f"-dark_{str(args.dark_mode)}"

    elif "vanilla" in args.method.lower():
        if args.basic_mode == -1:
            value_adding_options = [arg for arg in vars(args) if "add_" in arg and getattr(args, arg)]
            if value_adding_options:
                value_adding_options = "-" + "-".join(value_adding_options)
            else:
                value_adding_options = ""

        elif args.basic_mode != -1:    
            if args.basic_mode == 0:
                args.add_current_coord_for_basic = False
                args.add_possible_action_for_basic = False
                args.add_possible_coord_for_basic = False
            elif args.basic_mode == 1:
                args.add_current_coord_for_basic = False
                args.add_possible_action_for_basic = True
                args.add_possible_coord_for_basic = False
            elif args.basic_mode == 2:
                args.add_current_coord_for_basic = True
                args.add_possible_action_for_basic = False
                args.add_possible_coord_for_basic = False
            elif args.basic_mode == 3:
                args.add_current_coord_for_basic = True
                args.add_possible_action_for_basic = True
                args.add_possible_coord_for_basic = False
            elif args.basic_mode == 4:
                args.add_current_coord_for_basic = True
                args.add_possible_action_for_basic = False
                args.add_possible_coord_for_basic = True
            elif args.basic_mode == 5:
                args.add_current_coord_for_basic = True
                args.add_possible_action_for_basic = True
                args.add_possible_coord_for_basic = True
            else:
                raise ValueError(f"{args.add_basic_mode} NOT supported")
            value_adding_options = f"-basic_{str(args.basic_mode)}"
    else:
        if args.basic_mode == -1:
            value_adding_options = [arg for arg in vars(args) if "add_" in arg and getattr(args, arg)]
            if value_adding_options:
                value_adding_options = "-" + "-".join(value_adding_options)
            else:
                value_adding_options = ""

        elif args.basic_mode != -1:        
            if args.basic_mode == 0:
                args.add_current_coord_for_basic = False
                args.add_possible_action_for_basic = False
                args.add_possible_coord_for_basic = False
            elif args.basic_mode == 1:
                args.add_current_coord_for_basic = False
                args.add_possible_action_for_basic = True
                args.add_possible_coord_for_basic = False
            elif args.basic_mode == 2:
                args.add_current_coord_for_basic = True
                args.add_possible_action_for_basic = False
                args.add_possible_coord_for_basic = False
            elif args.basic_mode == 3:
                args.add_current_coord_for_basic = True
                args.add_possible_action_for_basic = True
                args.add_possible_coord_for_basic = False
            elif args.basic_mode == 4:
                args.add_current_coord_for_basic = True
                args.add_possible_action_for_basic = False
                args.add_possible_coord_for_basic = True
            elif args.basic_mode == 5:
                args.add_current_coord_for_basic = True
                args.add_possible_action_for_basic = True
                args.add_possible_coord_for_basic = True
            else:
                raise ValueError(f"{args.add_basic_mode} NOT supported")
            value_adding_options = f"-basic_{str(args.basic_mode)}"
        
    length_think = ""
    if args.length_think == "len_explicit":
        length_think = "-len_explicit"
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        args.tokenizer = tokenizer

    data_file = args.data_path
    data_file_name = os.path.splitext(os.path.basename(args.data_path))[0]
    # ex. train, 10000, hard, 10x10
    data_split, data_num, data_level, world_size = data_file_name.split("_")
    if args.thought == "none":
        json_file = f"./data/{args.task}_{args.training_objective}_{data_num}_xsize{world_size.split('x')[0]}_ysize{world_size.split('x')[1]}_{data_level}_{args.method}{value_adding_options}{args.first_only}.json"
        print("target:", json_file)
    else:
        json_file = f"./data/{args.task}_{args.training_objective}_{data_num}_xsize{world_size.split('x')[0]}_ysize{world_size.split('x')[1]}_{data_level}_{args.method}_{args.thought}_{args.when_thought}{length_think}{value_adding_options}{args.cot_type}{args.first_only}.json"
        print("target:", json_file)
    json_data = []
    if os.path.exists(json_file):
        print(f"JSON file '{json_file}' already exists.")
    elif not os.path.exists(args.data_path):
        print(f"Data file '{args.data_path}' does not exist.")
    else:
        with open(args.data_path, 'r') as f:
            cnt=0
            for line in f:
                world = json.loads(line)
                game = Gridworld()
                game.init_from_sample(world)
                states, actions = game.getOptimalPath()
                
                if args.thought == "look-ahead":
                    game.getOptimalQValueTable()
                
                if args.method == "vanilla" or args.method == "kld":
                    if args.task == "gridworld":
                        data = make_kld_instance(cnt, world, game, states, actions, args)
                        json_data.append(data)
                        print(data)
                        print(f"Instance {cnt} created successfully.")
                        cnt+=1
                    else:    # args.task == "gridworld-dark"
                        data = make_kld_dark_instance(cnt, world, game, states, actions, args)
                        json_data.append(data)
                        print(f"Instance {cnt} created successfully.")
                        cnt+=1
                elif args.method == "dfs":
                    visited_state_action = {}
                    states = []
                    actions = []
                    values = []
                    initial = game.board.findPiecebyName('Player')[0].pos
                    from_action = []
                    while game.board.findPiecebyName('Player')[0].pos != game.board.findPiecebyName('Goal')[0].pos:
                        # print(f"states: {states}, actions: {actions}")
                        cur_pos = game.board.findPiecebyName('Player')[0].pos
                        if cur_pos not in visited_state_action.keys():
                            visited_state_action[cur_pos] = {'valid': [], 'from': from_action, 'invalid': []}
                            for action in game.getActions():    # ['up', 'down', 'left', 'right']
                                next = game.checkMove(game.board.findPiecebyName('Player')[0].pos, action)
                                if action not in from_action:
                                    if not(next in [item.pos for item in game.board.findPiecebyName('Pit')] or next in [item.pos for item in game.board.findPiecebyName('Wall')] or next[0] > (game.xsize + game.xstart-1) or next[1] > (game.ysize + game.ystart -1) or next[0] < game.xstart or next[1] < game.ystart):
                                        visited_state_action[cur_pos]['valid'].append(action)
                                    else:
                                        visited_state_action[cur_pos]['invalid'].append(action)
                        if len(visited_state_action[cur_pos]['valid']) == 0:
                            if visited_state_action[cur_pos]['from']==[]:
                                assert False
                            action = visited_state_action[cur_pos]['from'][0]
                            value = {"up": 0, "down": 0, "left": 0, "right": 0}
                            value[action] = 1
                            values.append(value)
                            visited_state_action[cur_pos]['from'] = []
                            states.append(game.board.findPiecebyName('Player')[0].pos)
                            game.makeMove(action)
                            actions.append(action)
                        else:
                            value = {"up": 0, "down": 0, "left": 0, "right": 0}
                            for action in visited_state_action[cur_pos]['valid']:
                                value[action] = 1/len(visited_state_action[cur_pos]['valid'])
                            values.append(value)
                            action = random.choice(visited_state_action[cur_pos]['valid'])  # random choose among valid ones, potentially cause hanging
                            visited_state_action[cur_pos]['valid'].pop(visited_state_action[cur_pos]['valid'].index(action))
                            visited_state_action[cur_pos]['invalid'].append(action)
                            states.append(game.board.findPiecebyName('Player')[0].pos)
                            game.makeMove(action)
                            actions.append(action)
                        from_action = ["up" if action == "down" else "down" if action == "up" else "left" if action == "right" else "right"]
                    if args.task == "gridworld":
                        data = make_dfs_instance(cnt, world, game, states, actions, values, args)
                    else:    # args.task == "gridworld-dark"
                        data = make_dfs_dark_instance(cnt, world, game, states, actions, values, args)
                    json_data.append(data)
                    print(f"Instance {cnt} created successfully.")
                    cnt+=1
                elif args.method =="oracle_kld":
                    for (x,y), q_dict in game.q_table.items():
                        if q_dict['up']!=0 or q_dict['down']!=0 or q_dict['left']!=0 or q_dict['right']!=0:
                            world["question"]['start'] = [x, y]
                            game = Gridworld()
                            game.init_from_sample(world)
                            states, actions = game.getOptimalPath()
                            print((x,y), states)
                            data = make_kld_instance(cnt, world, game, states, actions, args)
                            json_data.append(data)
                            print(f"Instance {cnt} created successfully.")
                            cnt+=1
                elif args.method == "vanilla_onestep":
                    states = [tuple(world['question']['start'])] + states[:-1]
                    for ord, (state, action) in enumerate(zip(states, actions)):
                        world['question']['start'] = state
                        data = make_instance(cnt, world, [], [], args)
                        data["conversations"].append({"from": "gpt", "value": f"{action}"})
                        # print(f"{state} {action} {world['question']['goal']}")
                        json_data.append(data)
                        print(f"Instance {cnt} created successfully.")
                        cnt+=1
                elif args.method == "oracle" or args.method == "oracle_onestep":
                    for (x,y), q_dict in game.q_table.items():
                        if q_dict['up']!=0 or q_dict['down']!=0 or q_dict['left']!=0 or q_dict['right']!=0:
                            world["question"]['start'] = [x, y]
                            game = Gridworld()
                            game.init_from_sample(world)
                            states, actions = game.getOptimalPath()
                            if args.method == "oracle":
                                print((x,y), states)
                                data = make_instance(cnt, world, states, actions, args)
                            elif args.method == "oracle_onestep":
                                print((x,y), actions[0], states[0])
                                data = make_instance(cnt, world, [], [], args)
                                data["conversations"].append({"from": "gpt", "value": f"{actions[0]}"})
                            json_data.append(data)
                            print(f"Instance {cnt} created successfully.")
                            cnt+=1
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=4)
        print(f"SFT data '{json_file}' created successfully.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--task",
        type=str,
        default="gridworld",
        choices=["gridworld", "gridworld-dark"],
        help="Whether or not to provide information about your initial GridworldEnv and your current location."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./envs/gridworld/data/train_10000_hard_10x10.jsonl",
        help="The number of instances to generate.",
    )
    parser.add_argument(
        "--training_objective",
        type=str,
        default="sft",
        choices=["sft", "rft", "ppo", "iter-dpo"]
    )
    parser.add_argument(
        "--method",
        type=str,
        default="vanilla",
        choices=["vanilla", "oracle", "oracle_onestep", "vanilla_onestep", "kld", "oracle_kld", "dfs"],
        help="Method to generate the data.",
    )
    parser.add_argument(
        "--thought",
        type=str,
        default="none",
        choices=["none", "implicit", "trivial", "look-ahead"],
        help="Method to generate thought in data",
    )
    parser.add_argument(
        "--when_thought",
        type=str,
        default="each-step",
        choices=["each-step", "each-crossroad", "first-step", "first-step-reversed", "each-crossroad-reversed"],
    )
    parser.add_argument(
        "--add_current_coord_for_dark",
        action='store_true',
        help="Add current coordinate for each planning step."
    )
    parser.add_argument(
        "--add_possible_coord_for_dark",
        action='store_true',
        help="Add the coordinate of every possible move."
    )
    parser.add_argument(
        "--add_current_coord_for_basic",
        action="store_true",
        help="Add current coordinate for each planning step."
    )
    parser.add_argument(
        "--add_possible_action_for_basic",
        action="store_true",
        help="Add the possible actions for each planning step."
    )
    parser.add_argument(
        "--add_possible_coord_for_basic",
        action="store_true",
        help="Add the coordinate of every possible move."
    )
    parser.add_argument(
        "--basic_mode",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2, 3, 4, 5],
        help="Select a specific mode for basic DFS " \
        "(-1: Manual Mode, " \
        "0: No info, 1: Possible Action, 2: Current Coord, " \
        "3: Current Coord + Possible Action, 4: Current Coord + Possible Coord, " \
        "5: Current Coord + Possible Coord + Possible Action)"
    )
    parser.add_argument(
        "--dark_mode",
        type=int,
        default=-1,
        choices=[-1, 0, 1, 2],
        help="Select a specific mode for basic DFS " \
        "(-1: Manual Mode, " \
        "0: Possible Action, 1: Possible ACtion + Current Coord, " \
        "3: Possible Action + Current Coord + Possible Coord)"
    )
    parser.add_argument(
        "--length_think",
        type=str,
        default="none",
        choices=["none", "len_explicit"],
    )
    parser.add_argument(
        "--cot_type",
        type=str,
        default="none",
        choices=["none", "all", "possible", "all_possible", "all_possible_backtrack", "all_backtrack", "possible_backtrack", "backtrack"],
    )
    parser.add_argument(
        "--first_only",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--backtrack",
        type=bool,
        default=False,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="path of model.",
    )
    args = parser.parse_args()
    
    main(args)