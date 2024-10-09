from hmac import new
from math import pi
import os
import json
import random
import argparse
import time

from gridworld.gridworld_agent.envs.GridworldTextEnv import Gridworld
from torch import ne


def single_neighbors(x, y, world_size_x, world_size_y, visited):
    neighbors = set()
    if x > 0:
        if (x-1, y) not in visited:
            neighbors.add((x-1, y))
    if x < world_size_x - 1:
        if (x+1, y) not in visited:
            neighbors.add((x+1, y))
    if y > 0:
        if (x, y-1) not in visited:
            neighbors.add((x, y-1))
    if y < world_size_y - 1:
        if (x, y+1) not in visited:
            neighbors.add((x, y+1))
    return neighbors


def set_neighbors(visited, world_size_x, world_size_y):
    neighbors = set()
    for (x, y) in visited:
        tmp_neighbors = single_neighbors(x, y, world_size_x, world_size_y, visited)
        for (tmp_x, tmp_y) in tmp_neighbors:
            neighbors.add((tmp_x, tmp_y))
    return neighbors


class TreeNode:
    def __init__(self, x, y, parent=None):
        self.x = x
        self.y = y
        self.parent = parent
        self.banned = set()
        self.neighbors = set()
        self.children =  None
        self.visit = [] if parent is None else parent.visit.copy()
        self.visit.append((x, y))
    def make_children(self):
        if self.children is None:
            self.children = self.neighbors.difference(self.banned)
        return self.children


def make_dfs_path(world_size_x, world_size_y, start, goal):
    curr = TreeNode(start[0], start[1])
    
    curr.banned = set()
    curr.neighbors = single_neighbors(curr.x, curr.y, world_size_x, world_size_y, curr.visit)
    curr.make_children()
    history = []
    while curr.x != goal[0] or curr.y != goal[1]:
        item = {"visit": curr.visit.copy(), "banned": curr.banned.copy(), "neighbor": curr.neighbors.copy(), "children": curr.children.copy()}
        if item in history:
            history.append(item)
            for it in history:
                print(it)
            assert False
        else:
            history.append(item)
        
        if len(curr.children)<=0:
            removed = (curr.x, curr.y)
            parent = curr.parent
            del curr
            curr = parent
            ansestor = curr
            while ansestor is not None:
                if removed in ansestor.children:
                    ansestor.children.remove(removed)
                ansestor = ansestor.parent
            curr = parent
        else:
            new_x, new_y = random.sample(curr.children, 1)[0]
            new_node = TreeNode(new_x, new_y, curr)
            curr_neighber = curr.neighbors.copy()
            new_banned = curr_neighber.intersection(single_neighbors(new_node.x, new_node.y, world_size_x, world_size_y, new_node.visit))
            new_node.neighbors = curr_neighber.union(single_neighbors(new_node.x, new_node.y, world_size_x, world_size_y, new_node.visit))-set(new_node.visit)
            # print(new_x, new_y, new_node.neighbors)
            # print(new_x, new_y, new_node.neighbors)
            curr_banned = curr.banned.copy()
            new_node.banned = curr_banned.union(new_banned)
            # new_node.neighbors = new_node.neighbors.union(new_node.children)
            new_node.make_children()
            curr = new_node
    return curr.visit

    
def make_world(gridstart_x, gridstart_y, world_size_x, world_size_y, p_obstacle, p_pit):
    assert world_size_x > 1 and world_size_y > 1
    assert 0 <= p_obstacle <= 1 and 0 <= p_pit <= 1
    world = {"question": {}}
    world["question"]["world_size_x"] = world_size_x
    world["question"]["world_size_y"] = world_size_y
    world_pieces = set()
    for i in range(world_size_x):
        for j in range(world_size_y):  
            world_pieces.add((i, j))
    start = (0, 0)
    world_pieces.discard(start)
    goal = (world_size_x-1, world_size_y-1)
    world_pieces.discard(goal)
    walls = []
    pits = []
    visited = make_dfs_path(world_size_x, world_size_y, start, goal)
    world_pieces = world_pieces - set(visited)
    for piece in world_pieces:
        rand_var = random.random() 
        if rand_var < p_obstacle:
            if random.random() < p_pit:
                pits.append(piece)
            else:
                walls.append(piece)
    world["question"]["wall"] = walls
    world["question"]["pit"] = pits
    world["question"]["start"] = start
    world["question"]["goal"] = goal
    world["question"]["gridstart_x"] = gridstart_x
    world["question"]["gridstart_y"] = gridstart_y
    return world



if __name__ == "__main__":
    json_data = []

    parser = argparse.ArgumentParser("Run the interactive loop.")
    parser.add_argument(
        "--num",
        type=int,
        default=10000,
        help="The number of instances to generate.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "inference"],
        help="The split of the data to generate instances for",
    )
    parser.add_argument(
        "--max_world_x_size",
        type=int,
        default=10,
        help="maximum value of world's x_size"
    )
    parser.add_argument(
        "--max_world_y_size",
        type=int,
        default=10,
        help="maximum value of world's y_size"
    )
    parser.add_argument(
        "--world_limit_x",
        type=int,
        default=20,
        help="world's x_size"
    )
    parser.add_argument(
        "--world_limit_y",
        type=int,
        default=20,
        help="world's y_size"
    )
    args = parser.parse_args()
    jsonl_file = f'./envs/gridworld/data/{args.split}_{args.num}_hard_{args.max_world_x_size}x{args.max_world_y_size}.jsonl'

    # if not os.path.exists(jsonl_file):
    if os.path.exists(jsonl_file):
        with open(jsonl_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                json_data.append(data)
                # Process the data
                # ...
    else:
        print(f"JSONL file '{jsonl_file}' does not exist.")
        
    with open(jsonl_file, 'a') as f:
        for i in range(args.num-len(json_data)):
            
            while True:
                gridstart_x = 0
                gridstart_y = 0
                world_size_x = random.randint(2, args.max_world_x_size)
                world_size_y = random.randint(2, args.max_world_y_size)
                p_obstacle = 1.0
                p_pit = random.uniform(0.1, 0.5)
                world = make_world(gridstart_x, gridstart_y, world_size_x, world_size_y, p_obstacle, p_pit)
                if world not in json_data:
                    break
            print(f"Instance {i} created successfully.")
            json_data.append(world)
            pad_x = random.randint(0, args.world_limit_x - world_size_x)
            pad_y = random.randint(0, args.world_limit_y - world_size_y)
            world["question"]["gridstart_x"] = pad_x
            world["question"]["gridstart_y"] = pad_y
            world["question"]["start"] = [world["question"]["start"][0]+pad_x, world["question"]["start"][1]+pad_y]
            world["question"]["goal"] = [world["question"]["goal"][0]+pad_x, world["question"]["goal"][1]+pad_y]
            for i in range(len(world["question"]["wall"])):
                world["question"]["wall"][i] = [world["question"]["wall"][i][0]+pad_x, world["question"]["wall"][i][1]+pad_y]
            for i in range(len(world["question"]["pit"])):
                world["question"]["pit"][i] = [world["question"]["pit"][i][0]+pad_x, world["question"]["pit"][i][1]+pad_y]
    
            json.dump(world, f)
            f.write('\n')
            f.flush()

    print(f"JSONL file '{jsonl_file}' created successfully.")
