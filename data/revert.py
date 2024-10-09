import json

file_path = ""
new_file_path = ""

with open(file_path, "r") as file:
    data = json.load(file)
with open(new_file_path, "w") as f:

    for item in data:
        #Grid is from (0, 0) to (6, 3). The goal state is at (6, 3). The pit is at (4, 2), and (2, 3). The wall is at (6, 2), (4, 0), (1, 2), (4, 3), (2, 0), (3, 3), (6, 0), (1, 0), and (3, 2). Current: (0, 0)."
        # {"question": {"world_size_x": 5, "world_size_y": 2, "wall": [[1, 1], [2, 1]], "pit": [[4, 0]], "start": [0, 0], "goal": [4, 1], "gridstart_x": 0, "gridstart_y": 0}}

        world_size_x = int(item["conversations"][2]["value"].split("Grid is from (0, 0) to (")[1].split(",")[0])+1
        world_size_y = int(item["conversations"][2]["value"].split("Grid is from (0, 0) to (")[1].split(",")[1].split(").")[0])+1
        goal_x = item["conversations"][2]["value"].split("The goal state is at (")[1].split(",")[0]
        goal_y = item["conversations"][2]["value"].split("The goal state is at (")[1].split(",")[1].split(").")[0]
        goal = [int(goal_x), int(goal_y)]
        start_x = item["conversations"][2]["value"].split("Current: (")[1].split(",")[0]
        start_y = item["conversations"][2]["value"].split("Current: (")[1].split(",")[1].split(").")[0]
        start = [int(start_x), int(start_y)]
        
        pit_list = item["conversations"][2]["value"].split("The pit is at ")
        if len(pit_list) > 1:
            pit_list = pit_list[1].split(". ")[0].replace("and ", "").split("), (")
            #["(4, 2", "2, 3)"]
            pit = []
            for pit_item in pit_list:
                pit.append([int(pit_item.split(", ")[0].replace("(", "").replace(")", "")), int(pit_item.split(", ")[1].replace("(", "").replace(")", ""))])
            #[[4, 2], [2, 3]]
        else:
            pit = []
        wall_list = item["conversations"][2]["value"].split("The wall is at ")
        if len(wall_list) > 1:
            wall_list = wall_list[1].split(". ")[0].replace("and ", "").split("), (")
            #["(6, 2)", "(4, 0)", "(1, 2)", "(4, 3)", "(2, 0)", "(3, 3)", "(6, 0)", "(1, 0)", "and (3, 2)"]
            wall = []
            for wall_item in wall_list:
                wall.append([int(wall_item.split(", ")[0].replace("(", "").replace(")", "")), int(wall_item.split(", ")[1].replace("(", "").replace(")", ""))])
            #[[6, 2], [4, 0], [1, 2], [4, 3], [2, 0], [3, 3], [6, 0], [1, 0], [3, 2]]
        else:
            wall = []
        world= {
            "question": {
                "world_size_x": world_size_x,
                "world_size_y": world_size_y,
                "wall": wall,
                "pit": pit,
                "start": start,
                "goal": goal,
                "gridstart_x": 0,
                "gridstart_y": 0
            }   
        }

        json.dump(world, f)
        f.write('\n')

    # Now you can use the 'data' variable to access the contents of the JSON file
