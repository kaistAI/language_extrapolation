import argparse, os, json, glob

from gridworld.gridworld_agent.envs.GridworldTextEnv import Gridworld

def is_cot_form(text):
    if text.startswith("thought:\nstep") and "action:\n" in text and text.split("action:\n")[1] in ["up", "down", "left", "right"]:
        return True
    else:
        return False
    
def get_cot_step_prediction(text_step_info_list):
    output = {}
    for idx, step_info in enumerate(text_step_info_list[1:]):
        step_num = int(step_info.split(":")[0].strip())
        assert step_num == idx+1
        possible_moves = [text for text in step_info.split(":")[1].strip().split("\n") if text != "<think>"]
        assert len(possible_moves) % 2 == 0
        possible_moves_list = []
        for i in range(int(len(possible_moves) / 2)):
            location = tuple([int(num) for num in possible_moves[2*i].replace("(", "").replace(")", "").split(", ")])
            direction = possible_moves[2*i+1].strip()
            assert direction in ["up", "down", "left", "right"]
            possible_moves_list.append((location, direction))
        output[idx+1] = possible_moves_list
    return output

def possible_next_state(current_state, game, visited):
    output = []
    for direction, action in zip([(current_state[0], current_state[1]+1), 
                                  (current_state[0], current_state[1]-1), 
                                  (current_state[0]-1, current_state[1]), 
                                  (current_state[0]+1, current_state[1])],
                                 ["up", "down", "left", "right"]):
        if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
            output.append((direction, action))
    return output

def get_cot_step_label(game, start_state, is_reverse):
    cnt = 0
    output = {}
    goal_state = game.board.findPiecebyName('Goal')[0].pos
    if is_reverse:
        visited = [goal_state]
        curr = [goal_state]
        target_state = start_state
        action_list = ["down", "up", "right", "left"]
    else: 
        visited = [start_state]
        curr = [start_state]
        target_state = goal_state
        action_list = ["up", "down", "left", "right"]
    next = []

    while target_state not in visited:
        cnt +=1
        for key in curr:
            for direction, action in zip([(key[0], key[1]+1), 
                                        (key[0], key[1]-1), 
                                        (key[0]-1, key[1]), 
                                        (key[0]+1, key[1])],
                                        action_list):
                if direction not in visited and not(direction in [item.pos for item in game.board.findPiecebyName('Pit')]) and not(direction in [item.pos for item in game.board.findPiecebyName('Wall')]) and direction[0] <= (game.xstart + game.xsize-1) and direction[1] <= (game.ystart + game.ysize-1) and direction[0] >= game.xstart and direction[1] >= game.ystart:
                    next.append((direction, action))
        output[cnt] = next

        visited.extend([key for key, action in next if key != ""])
        curr = [key for key, action in next if key != ""]
        next = []
    return output
    
    
def compare_cot_step_with_true_value(thought_turn, game, start_state, is_reverse):
    if not is_cot_form(thought_turn):
        print(f"thought format is incorrect")
        return False
    text_step_info_list = thought_turn.split("thought:\n")[1].split("action:\n")[0].split("step ")
    try:
        # {1: [([11, 6], 'up'), ([12, 5], 'right')], 2: [([11, 7], 'up'), ([13, 5], 'right')], 3: [([11, 8], 'up'), ([12, 7], 'right'), ([14, 5], 'right')], 4: [([11, 9], 'up'), ([14, 6], 'up')], 5: [([11, 10], 'up'), ([14, 7], 'up'), ([15, 6], 'right')], 6: [([11, 11], 'up'), ([14, 8], 'up')], 7: [([11, 12], 'up'), ([12, 11], 'right'), ([14, 9], 'up')], 8: [([13, 9], 'left'), ([15, 9], 'right')], 9: [([13, 10], 'up'), ([15, 10], 'up')], 10: [([15, 11], 'up')], 11: [([15, 12], 'up'), ([14, 11], 'left')]}
        step_info_dict_prediction = get_cot_step_prediction(text_step_info_list)
        print(f"step_info_dict_prediction: {step_info_dict_prediction}")
    except:
        print("error!")
        return False
    
    # Get true label CoT
    step_info_dict_label = get_cot_step_label(game, start_state, is_reverse)
    print(f"step_info_dict_label: {step_info_dict_label}")
    
    print(step_info_dict_prediction == step_info_dict_label)
    
    return step_info_dict_prediction == step_info_dict_label

def load_game(history):
    # Extracting variables from text
    world_info = history["conversations"][2]["value"].lower()
    # "(11, 5) to (15, 12). "
    grid_info = world_info.split("grid is from ")[1].split("goal")[0]
    start = [int(grid_info.split(" to ")[0].split(",")[0].split("(")[-1].strip()), int(grid_info.split(" to ")[0].split(",")[1].split(")")[0].strip())]
    end = [int(grid_info.split(" to ")[1].split(",")[0].split("(")[-1].strip()), int(grid_info.split(" to ")[1].split(",")[1].split(")")[0].strip())]
    world_size_x = end[0] - start[0] + 1
    world_size_y = end[1] - start[1] + 1
    
    # ": (15, 12)"
    goal_info = world_info.split("goal")[1].split("the pit")[0].strip()
    goal = [int(goal_info.split(",")[0].split("(")[-1].strip()), int(goal_info.split(",")[1].split(")")[0].strip())]
    
    if len(world_info.split("the pit is at ")) > 1:
        # (12, 9), (12, 12), (12, 8
        pit_info = world_info.split("the pit is at ")[1].split(").")[0].replace(", and ", ", ")
        # ["12, 9", "12, 12", "12, 8"]
        pit = [list(map(int, p.replace("(", "").split(", "))) for p in pit_info.split("), ")]
    else:
        pit = []
    if len(world_info.split("the wall is at ")) > 1: 
        # (15, 5), (13, 6), (13, 12), (15, 8), (12, 10), (14, 12), (12, 6), (15, 7), (13, 8), (13, 11), (13, 7), (14, 10)
        wall_info = world_info.split("the wall is at ")[1].split("\ncurrent")[0].replace(", and ", ", ")
        wall = [list(map(int, p.split("(")[-1].split(", "))) for p in wall_info.split(")")[:-1]]
    else:
        wall = []
            
    game = Gridworld()
    game.init_from_sample({"question": {
        "world_size_x": world_size_x, 
        "world_size_y": world_size_y, 
        "wall": wall, 
        "pit": pit, 
        "start": start, 
        "goal": goal, 
        "gridstart_x": start[0], 
        "gridstart_y": start[1]
    }})
    
    return game


def main(args):
    
    model_name = args.model_path.split("/")[-1]
    output_path = os.path.join(args.output_dir, model_name, args.exp_config+args.exp_name, args.split, "shot-1", "inference_output")
    
    for output_file in glob.glob(f"{output_path}/*"):
        if os.path.isfile(output_file) and output_file.endswith(".json"):
            # histories = json.load(open(output_file))
            # for history_num, history in enumerate(histories):
            history = json.load(open(output_file))
            game = load_game(history)
            if args.when_thought == "first-step":
                thought_turn = history["conversations"][3]["value"].lower()
                current_state = (game.xstart, game.ystart)
                history["conversations"][3]["cot_correctness"] = compare_cot_step_with_true_value(thought_turn, game, start_state=current_state, is_reverse=args.reverse)
            else:   # each-crossroad
                current_state = (game.xstart, game.ystart)
                visited = []
                for idx, turn in enumerate(history["conversations"][3:]):
                    thought_turn = turn["value"].lower()
                    if turn["from"] == "human":
                        current_state = tuple([int(num) for num in thought_turn.split("current:")[1].split("possible:")[0].strip().replace("(", "").replace(")", "").split(", ")])
                        # print(f"current_state: {current_state}")
                    else:   # turn["from"] == "gpt"
                        possible_next_states = possible_next_state(current_state, game, visited)
                        if (len(possible_next_states) > 1 and not thought_turn.startswith("thought:\nstep")) or (len(possible_next_states) <= 1 and thought_turn.startswith("thought:\nstep")):
                            history["conversations"][idx+3]["cot_correctness"] = False
                        elif len(possible_next_states) <= 1:
                            visited = [current_state]
                            continue
                        else:
                            history["conversations"][idx+3]["cot_correctness"] = compare_cot_step_with_true_value(thought_turn, game, current_state, args.reverse)
                            visited = [current_state]                     


if __name__=="__main__":
    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="output directory path",
    )
    parser.add_argument(
        "--exp_config",
        type=str,
        default="gridworld",
        help="Config of experiment.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="The name of the experiemnt.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Evaluation split.",
    )
    parser.add_argument(
        "--when_thought",
        type=str,
        default="each-crossroad",
        choices=["each-crossroad", "first-step"],
    )
    parser.add_argument(
        "--reverse",
        action="store_true"
    )
    args = parser.parse_args()
    main(args)

