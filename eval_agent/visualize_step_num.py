import matplotlib.pyplot as plt
import numpy as np
from eval_agent.utils.gridworld_visualize import *
import json
from gridworld.gridworld_agent.envs.GridworldTextEnv import Gridworld
import os
import seaborn as sns
from tqdm import tqdm

PATH = ["", ""]
NAME = ["Ours", "DFS"]
COLOR = ["blue", "red"]

def main():

    max_actual_step = -1
    dfs_value = {}
    apb_value = {}
    
    # Creating the scatter plot
    plt.figure(figsize=(10, 10))
    
    for path, name, color in zip(PATH, NAME, COLOR):
        optimal_steps = []
        actual_steps = []
        data = {}

        for file in tqdm(os.listdir(path)):
            if not ".json" in file:
                continue
            history = json.load(open(os.path.join(path, file)))
        
            xsize, ysize, goal, pit, wall, start, _, _ = start_env(history)
            world_info = {
                "question": {
                "world_size_x": xsize,
                "world_size_y": ysize,
                "wall": [list(e) for e in wall],
                "pit": [list(e) for e in pit],
                "start": list(start),
                "goal": list(goal),
                "gridstart_x": list(start)[0],
                "gridstart_y": list(start)[1],
                }
            }
            game = Gridworld()
            game.init_from_sample(world_info)
            # print(game.getOptimalPath())
            optimal_step = len(game.getOptimalPath()[1])
            if history["meta"]["terminate_reason"] == "success":
                actual_step = history["meta"]["steps"]
                actual_steps.append(actual_step)
                optimal_steps.append(optimal_step)
                if not optimal_step in data:
                    data[optimal_step] = [actual_step]
                else:
                    data[optimal_step].append(actual_step)
            
            if max_actual_step < actual_step:
                max_actual_step = actual_step
                
            # if len(optimal_steps) > 50:
            #     break
        
        keys = []
        avg_values = []
        for key in data.keys():
            keys.append(key)
            avg_values.append(sum(data[key])/len(data[key]))
        # plt.scatter(keys, avg_values, color=color, marker=)
        #     plt.scatter(key, sum(data[key])/len(data[key]), color="red", marker="x", s=70, label="Average Actual Steps for each Optimal Steps" if key == list(data.keys())[0] else "")
        plt.scatter(optimal_steps, actual_steps, c=color, label=f"Actual Steps for {name}", alpha=0.3, s=20)

        # Add regression plot
        if name == "apb":
            sns.regplot(x=keys, y=avg_values, scatter=False, color=color, line_kws={"lw": 3}, label=f'Regression Line for {name}')
        else:
            sns.regplot(x=keys, y=avg_values, scatter=False, order=2, color=color, line_kws={"lw": 3}, label=f'Regression Line for {name}')
        
    plt.title('Optimal Steps vs. Actual Steps')
    plt.xlabel('Optimal Steps')
    plt.ylabel('Actual Steps')
    plt.grid(True)
    plt.legend()
    
    # Save the plot as a PNG file
    plt.savefig(f'./result.png', format='png')
    plt.close()


if __name__=="__main__":
    main()