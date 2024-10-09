import json


def main():
    
    file_paths = []
    icl_file_names = []
    
    for file_path, icl_file_name in zip(file_paths, icl_file_names):
        data_num = 1
        output = []
        with open(file_path, "r") as f:
            json_file = json.load(f)
        for instance in json_file:
            if data_num > 8:
                break
            conversation = []
            for turn in instance["conversations"][2:]:
                new_turn = {}
                if turn["from"] == "human":
                    new_turn["role"] = "user"
                    new_turn["content"] = turn["value"]
                else:
                    new_turn["from"] = "gpt"
                    new_turn["content"] = turn["value"]
                conversation.append(new_turn)
            output.append(conversation)
            data_num += 1
        with open(f"eval_agent/prompt/icl_examples/{icl_file_name}.json", "w") as f:
            json.dump(output, f, indent=4)
        


if __name__=="__main__":
    main()