import numpy as np
from itertools import count
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.animation
from matplotlib.colors import CenteredNorm

def start_env(history, model_name):
    # Extracting variables from text
    if "llama" in model_name.lower():
        text = history["conversations"][2]["value"].lower()
    else:   # GPT
        text = history["conversations"][0]["value"].lower()
    grid_info = text.split("grid is from ")[1].split(".")[0]
    xsize = int(grid_info.split(" to ")[1].replace("(", "").split(", ")[0])- int(grid_info.split(" to ")[0].replace("(", "").split(", ")[0])+1
    ysize = int(grid_info.split(" to ")[1].replace(")", "").split(", ")[1])- int(grid_info.split(" to ")[0].replace(")", "").split(", ")[1])+1
    goal_info = text.split("goal: ")[1].split("\n")[0]
    goal = tuple(map(int, goal_info.replace("(", "").replace(")", "").split(", ")))
    if len(text.split("the pit is at "))>1:
        pit_info = text.split("the pit is at ")[1].split(").")[0].replace(", and ", ", ")
        pit = [tuple(map(int, p.replace("(", "").split(", "))) for p in pit_info.split("), ")]
    else:
        pit = []
    if len(text.split("the wall is at "))>1:    
        wall_info = text.split("the wall is at ")[1].split(").")[0].replace(", and ", ", ")
        wall = [tuple(map(int, p.replace("(", "").split(", "))) for p in wall_info.split("), ")]
    else:
        wall = []
    # start = (0, 0)
    start = tuple(map(int, text.split("grid is from ")[1].split(".")[0].split("to")[0].strip().replace("(", "").replace(")", "").split(", ")))
    actions, states = [], []
    # for i in range(3, len(history["conversations"]), 2):
    #     action = history["conversations"][i]["value"]
    #     state = history["conversations"][i+1]["value"].split("Current: ")[1].split(".")[0]
    #     state_x = int(state.split("(")[1].split(",")[0])
    #     state_y = int(state.split(",")[1].split(")")[0])
    #     actions.append(action)
    #     states.append((state_x, state_y))
    # Return extracted variables
    return xsize, ysize, goal, pit, wall, start, actions, states

# def plot_gridworld(xsize, ysize, goal, pit, wall, start):

def quatromatrix(left, bottom, right, top, ax=None, triplotkw={},tripcolorkw={}):

    if not ax: ax=plt.gca()
    n = left.shape[0]; m=left.shape[1]

    a = np.array([[0,0],[0,1],[.5,.5],[1,0],[1,1]])
    tr = np.array([[0,1,2], [0,2,3],[2,3,4],[1,2,4]])

    A = np.zeros((n*m*5,2))
    Tr = np.zeros((n*m*4,3))

    for i in range(n):
        for j in range(m):
            k = i*m+j
            A[k*5:(k+1)*5,:] = np.c_[a[:,0]+j, a[:,1]+i]
            Tr[k*4:(k+1)*4,:] = tr + k*5

    C = np.c_[ left.flatten(), bottom.flatten(), 
              right.flatten(), top.flatten()   ].flatten()

    triplot = ax.triplot(A[:,0], A[:,1], Tr, **triplotkw)
    tripcolor = ax.tripcolor(A[:,0], A[:,1], Tr, facecolors=C, **tripcolorkw)
    return tripcolor


def animate(state):
    frames = len(state)
    print("Rendering %d frames..." % frames)
    fig = plt.figure(figsize=(6, 2))
    fig_grid = fig.add_subplot(121)
    fig_health = fig.add_subplot(243)
    fig_visible = fig.add_subplot(244)
    fig_health.set_autoscale_on(False)
    health_plot = np.zeros((frames, 1))

    def render_frame(i):
        grid, visible, health = state[i]
        # Render grid
        fig_grid.matshow(grid, vmin=-1, vmax=1, cmap='jet')
        fig_visible.matshow(visible, vmin=-1, vmax=1, cmap='jet')
        # Render health chart
        health_plot[i] = health
        fig_health.clear()
        fig_health.axis([0, frames, 0, 2])
        fig_health.plot(health_plot[:i + 1])

    anim = matplotlib.animation.FuncAnimation(
        fig, render_frame, frames=frames, interval=100
    )

    anim.save("FuncAni0.gif", fps=24)


def plot_accuracy():
    pass
def plot_q_values(q_table):
   pass 