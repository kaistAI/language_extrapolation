from math import inf
import os
import gym
import json
import random
import string
import time
from regex import D
import torch
import pickle
from os.path import dirname, abspath, join
import numpy as np

BASE_DIR = join(dirname(abspath(__file__)), '../..')
DEBUG_PROD_SIZE = None  # set to `None` to disable

DEFAULT_FILE_PATH = join(BASE_DIR, '../../data/inference_3000_hard_20x20.jsonl')

class GridworldTextEnvSingle():
    """Gym environment for Text mode of Gridworld environment"""
    def __init__(
            self,
            file_path=DEFAULT_FILE_PATH,
            **kwargs
        ):
        """
        Constructor for text environment

        Arguments:
        observation_mode (`str`) -- ['html' | 'text'] (default 'html')
        get_image
        filter_goals
        limit_goals
        num_products
        human_goals
        session
        session_prefix
        show_attrs
        """
        super(GridworldTextEnvSingle, self).__init__()
        self.kwargs = kwargs
        self.session = self.kwargs.get('session')
        self.session_prefix = self.kwargs.get('session_prefix')
        self.prev_obs = []
        self.prev_actions = []
        self.num_prev_obs = self.kwargs.get('num_prev_obs', 0)
        self.num_prev_actions = self.kwargs.get('num_prev_actions', 0)
        
        self.game = Gridworld()
        # self.reset()
        self.samples = []
        with open(file_path, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
    
    def step(self, action):
        """
        Takes an action, updates environment, and returns (observation, reward, done, info)

        Arguments:
        action (`str`): An action should be of the following structure:
          - search[keywords]
          - click[value]
        If action not valid, perform nothing.
        """
        info = None

        # Determine action type (click, search) and argument
        status = self.game.makeMove(action)
        # Update observation, state with the new action
        self.game.first = False
        ob = self.observation
        text_list = [ob]
        self.prev_actions.append(action)
        for i in range(1, 1 + max(self.num_prev_obs, self.num_prev_actions)):
            if len(self.prev_actions) >= i and self.num_prev_actions >= i:
                text_list.append(self.prev_actions[-i])
            if len(self.prev_obs) >= i and self.num_prev_obs >= i:
                text_list.append(self.prev_obs[-i])
        state = ' [SEP] '.join(text_list[::-1])
        self.prev_obs.append(ob)
        return state, status['reward'], status['done'], info
    
    @property
    def observation(self):
        string = ""
        state = self.game.board.findPiecebyName('Player')[0].pos
        if self.game.first:
            goal = self.game.board.findPiecebyName('Goal')[0].pos
            wall_string = ""
            walls = [item.pos for item in  self.game.board.findPiecebyName('Wall')]
            if len(walls) > 0:
                wall_string += "The wall is at "
                for i in walls:
                    if i != walls[-1]:
                        wall_string += f"{tuple(i)}, " 
                    else:
                        if len(walls) > 1:
                            wall_string += "and "
                        wall_string += f"{tuple(i)}"
            else:
                wall_string = "There is no wall"
            pit_string = ""
            pits = [item.pos for item in  self.game.board.findPiecebyName('Pit')]
            if len(pits) > 0:
                pit_string += "The pit is at "
                for i in pits:
                    if i != pits[-1]:
                        pit_string += f"{tuple(i)}, "
                    else:
                        if len(pits) > 1:
                            pit_string += "and "
                        pit_string += f"{tuple(i)}"
            else:
                pit_string = "There is no pit"
            condition = f"Grid is from ({self.game.xstart}, {self.game.ystart}) to {goal}. Goal: {goal}\nCurrent: {state}\n{pit_string}. {wall_string}.\n"
            string += condition
        
        if self.add_current_coord:
            string += f"\nCurrent:\n{state}"
        
        possible = ""
        if self.add_possible_action or self.add_possible_coord:
            deltas = [(0, 1, "up"), (0, -1, "down"), (-1, 0, "left"), (1, 0, "right")]
            for delta in deltas:
                if self.is_safe_move(state, delta) and self.is_effective_move(state, delta):
                    if self.add_possible_coord:
                        possible += f"\n({state[0] + delta[0]}, {state[1] + delta[1]})"
                    if self.add_possible_action:
                        possible += f"\n{delta[-1]}"
            if possible == "":
                possible += "\nThere is no possible move."
            else:
                possible = "\nPossible:" + possible
        
        string += possible

        if self.game.Success():
            string += f"You have reached the goal state."
        elif self.game.Fail():
            string += f"You have failed to reach the goal state."
        return string.strip()
        
    def is_safe_move(self, state, delta):
        if (state[0] + delta[0], state[1] + delta[1]) in [item.pos for item in self.game.board.findPiecebyName("Wall")]:
            return False
        if (state[0] + delta[0], state[1] + delta[1]) in [item.pos for item in self.game.board.findPiecebyName("Pit")]:
            return False
        return True
    
    def is_effective_move(self, state, delta):
        goal_lst =  self.game.board.findPiecebyName('Goal')
        assert len(goal_lst) == 1, f"Goal is NOT singleton: {' '.join([item.pos for item in goal_lst])}"
        goal_x, goal_y = goal_lst[0].pos

        next_x, next_y = (state[0] + delta[0], state[1] + delta[1])
        if self.game.xstart <= next_x <= goal_x and self.game.ystart <= next_y <= goal_y:
            return True
        else:
            return False

    @property
    def state(self):
        """
        State that includes all information. The actual observation are
        likely to be a subset or reduced form of the state.
        """
        
        return self.game.components.items()
    
    def reset(self, session=None, instruction_text=None):
        """Create a new session and reset environment variables"""
        session_int = None
        if session is not None:
            self.session = str(session)
            if isinstance(session, int):
                session_int = session
        else:
            self.session = ''.join(random.choices(string.ascii_lowercase, k=10))
        if self.session_prefix is not None:
            self.session = self.session_prefix + self.session
        self.game.init_from_sample(self.samples[session_int])
        obs = self.observation
        self.prev_obs = [obs]
        self.prev_actions = []
        return obs, None

    def render(self, mode='human'):
        pass

    def close(self):
        pass


def randPair(s,e):
    return np.random.randint(s,e), np.random.randint(s,e)


class BoardPiece:
    
    def __init__(self, name, code, pos):
        self.name = name #name of the piece
        self.code = code #an ASCII character to display on the board
        self.pos = pos #2-tuple e.g. (1,4)
        self.id = id(self)


class GridBoard:
    
    def __init__(self, xsize=4, ysize=4, xstart=0, ystart=0):
        self.xsize = xsize
        self.ysize = ysize   #Board dimensions, e.g. 4 x 4
        self.xstart = xstart
        self.ystart = ystart
        self.components = {} #name : board piece
    
    def addPiece(self, name, code, pos=(0,0)):
        newPiece = BoardPiece(name, code, pos)
        self.components[newPiece.id] = newPiece 
    
    def isPiece(self, pos):
        for id, piece in self.components.items():
            if piece.pos == pos:
                return True
        return False

    def findPiecebyID(self, id):
        return self.components[id]

    def findPiecebyName(self, name):
        pieces = []
        for id, piece in self.components.items():
            if piece.name == name:
                pieces.append(piece)
        return pieces
    
    def findPiecebyPOS(self, pos):
        for name, piece in self.components.items():
            if piece.pos == pos:
                return piece
        return None
    
    def movePiece(self, id, pos):
        self.components[id].pos = pos
    
    def delPiece(self, id):
        del self.components[id]
    
    def render(self):
        dtype = '<U2'
        displ_board = np.zeros((self.ysize+self.ystart, self.xsize+self.xstart), dtype=dtype)
        displ_board[:] = ' '
        
        for id, piece in self.components.items():
                displ_board[(self.ysize+self.ystart-piece.pos[1]-1, piece.pos[0])] = piece.code
        return displ_board
        
        
def addTuple(a,b):
    return tuple([sum(x) for x in zip(a,b)])


class Gridworld:
    def __init__(self, mode='varaible'):
        if mode == 'demo':
            xsize = 4
            ysize = 4
            xstart = 0
            ystart = 0
            self.board = GridBoard(xsize=xsize, ysize=ysize, xstart=xstart, ystart=ystart)
            self.xsize = xsize
            self.ysize = ysize
            self.xstart = xstart
            self.ystart = ystart
            self.init_board((0, 3), (0, 0), [(0, 1)], [(1, 1)])
        self.q_table = dict()
        self.l_table = dict()
        self.gamma = 0.9
        self.pit_reward = -1
        self.goal_reward = 1
        self.board_reward = 0
        self.first = True
        self.actions = ['up', 'down', 'left', 'right']
    
    def init_board(self, player, goal, pits, walls):
        
        self.board.addPiece('Player','P',tuple(player))
        self.board.addPiece('Goal','+',tuple(goal))
        for pit in pits:
            self.board.addPiece('Pit','-',tuple(pit))
        for wall in walls:
            self.board.addPiece('Wall','W',tuple(wall))
    
    def init_from_sample(self, sample, actions = ['up', 'down', 'left', 'right']):
        # question: {"world_size_x": 4, "world_size_y": 4, "start": [0, 3], "goal": [0, 0], "wall": [[1, 1]], "pit": [[0, 1]]}, "answer": []}
        self.xsize = sample['question']['world_size_x']
        self.ysize = sample['question']['world_size_y']
        self.board = GridBoard(xsize=self.xsize, ysize=self.ysize)
        self.init_board(sample['question']['start'], sample['question']['goal'], sample['question']['pit'], sample['question']['wall'])
        self.first = True
        self.xstart = sample['question']['gridstart_x']
        self.ystart = sample['question']['gridstart_y']
        # print("Initialized board from sample")
        self.actions = actions
    
    # Check if board is initialized appropriately (no overlapping pieces)
    def validateBoard(self):
        all_positions = [piece.pos for id,piece in self.board.components.items()]
        if len(all_positions) > len(set(all_positions)):
            return False
        else:
            return True

    # Initialize player in random location, but keep wall, goal and pit stationary
    def initGridPlayer(self):
        # height x width x depth (number of pieces)
        self.initGridStatic()
        # place player
        self.board.components['Player'].pos = randPair(0,self.board.size)

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridPlayer()

    #Initialize grid so that goal, pit, wall, player are all randomly placed
    def initGridRand(self):
        #height x width x depth (number of pieces)
        self.board.findPiecebyName('Player')[0].pos = randPair(0,self.board.size)
        self.board.findPiecebyName('Goal')[0].pos = randPair(0,self.board.size)
        self.board.findPiecebyName('Pit')[0].pos = randPair(0,self.board.size)
        self.board.findPiecebyName('Wall')[0].pos = randPair(0,self.board.size)
        

        if (not self.validateBoard()):
            #print('Invalid grid. Rebuilding..')
            self.initGridRand()

    def checkMove(self, old_pos, action):
        if action == self.getActions()[0]:
            addpos = (0,1)
        elif action == self.getActions()[1]:
            addpos = (0,-1)
        elif action == self.getActions()[2]:
            addpos = (-1,0)
        elif action == self.getActions()[3]:
            addpos = (1,0)
        else:
            addpos = (0,0)
        new_pos = addTuple(old_pos, addpos)
        return new_pos

    def makeMove(self, action):
        #need to determine what object (if any) is in the new grid spot the player is moving to
        #actions in {u,d,l,r}
        prompt = ""
        gold_states, gold_actions = self.getOptimalPath()
        for _, gold_action in zip(gold_states, gold_actions):
            prompt += f"{gold_action}\n"
        if action.strip() == prompt.strip():    # caution: (action: "right\nup\nright", gold: "right\nup\nright\n")
            reward = 1
        else:
            reward = -1
        print(f"prompt: {prompt}, action: {action}, reward: {reward}")
        return {'invalid': False, 'reward': reward, 'done': True}

    def getReward(self, pos):
        if pos in [item.pos for item in self.board.findPiecebyName('Wall')] or pos in [item.pos for item in self.board.findPiecebyName('Pit')] or pos[0] > self.xstart + self.xsize -1  or pos[1] > self.ystart + self.ysize -1 or pos[0] < self.xstart or pos[1] < self.ystart:
            return self.pit_reward
        elif pos in [item.pos for item in self.board.findPiecebyName('Goal')]:
            return self.goal_reward
        else:
            return self.board_reward

    def dispGrid(self):
        return self.board.render()

    def getOptimalQValueTable(self):
        for x in range(self.xstart, self.xstart+self.xsize):
            for y in range(self.ystart, self.ystart + self.ysize):
                self.q_table[(x,y)] = {self.getActions()[0]:0, self.getActions()[1]:0, self.getActions()[2]:0, self.getActions()[3]:0}
        # returns the optimal q-value table for each state, action pair
        while True:
            flag = True
            for x in range(self.xstart, self.xstart+self.xsize):
                for y in range(self.ystart, self.ystart + self.ysize):
                    if not (self.board.isPiece((x, y)) and self.board.findPiecebyPOS((x,y)).name in ['Pit', 'Goal', 'Wall']):
                        for action in self.getActions():
                            update_qvalue = self.getOptimalQValue((x,y), action)
                            if update_qvalue != self.q_table[(x,y)][action]:
                                self.q_table[(x,y)][action] = update_qvalue
                                flag = False
            if flag == True:
                break
        data = {}
        for x in range(self.xstart, self.xstart+self.xsize):
            for y in range(self.ystart, self.ystart + self.ysize):
                data[f"{x},{y}"] = self.q_table[(x,y)]
        return data

    def getOptimalLengthTable(self):
        for x in range(self.xstart, self.xstart+self.xsize):
            for y in range(self.ystart, self.ystart + self.ysize):
                self.l_table[(x,y)] = {self.getActions()[0]:inf, self.getActions()[1]:inf, self.getActions()[2]:inf, self.getActions()[3]:inf}
        # returns the optimal q-value table for each state, action pair
        while True:
            flag = True
            for x in range(self.xstart, self.xstart+self.xsize):
                for y in range(self.ystart, self.ystart + self.ysize):
                    if not (self.board.isPiece((x, y)) and self.board.findPiecebyPOS((x,y)).name in ['Pit', 'Goal', 'Wall']):
                        for action in self.getActions():
                            update_length = self.getOptimalLength((x,y), action)
                            if update_length < self.l_table[(x,y)][action]:
                                self.l_table[(x,y)][action] = update_length
                                flag = False
            if flag == True:
                break
        data = {}
        for x in range(self.xstart, self.xstart+self.xsize):
            for y in range(self.ystart, self.ystart + self.ysize):
                data[f"{x},{y}"] = self.l_table[(x,y)]
        return data

    
    def getOptimalPath(self):
        #returns the optimal path for the agent
        """
        if self.q_table == dict():
            self.getOptimalQValueTable()
        if self.l_table == dict():
            self.getOptimalLengthTable()
        """
        self.q_table = {}
        self.getOptimalQValueTable()
        self.l_table = {}
        self.getOptimalLengthTable()

        path = []
        actions = []
        pos = self.board.findPiecebyName('Player')[0].pos
        while pos != self.board.findPiecebyName('Goal')[0].pos:
            action = max(self.q_table[pos], key=self.q_table[pos].get)
            path.append(pos)
            actions.append(action)
            pos = self.checkMove(pos, action)
            
        return path, actions
    
    def getOptimalPolicy(self, pos):
        #returns the list of optimal actions for the state
        max_value = max(self.q_table[pos].values())
        possible_moves = []
        for action in self.getActions():
            if self.q_table[pos][action] == max_value:
                possible_moves.append(action)
        return possible_moves
        
    def getOptimalQValue(self, pos, move):
        #returns the optimal q-value for the state
        if self.board.isPiece(self.checkMove(pos, move)) and self.board.findPiecebyPOS(self.checkMove(pos, move)).name=='Goal':
            max_q_value = self.goal_reward
        elif self.board.isPiece(self.checkMove(pos, move)) and self.board.findPiecebyPOS(self.checkMove(pos, move)).name=='Pit' or self.board.isPiece(self.checkMove(pos, move)) and self.board.findPiecebyPOS(self.checkMove(pos, move)).name=='Wall' or self.checkMove(pos, move)[0] > self.xstart + self.xsize -1 or self.checkMove(pos, move)[1] > self.ysize + self.ystart -1 or self.checkMove(pos, move)[0] < self.xstart or self.checkMove(pos, move)[1] < self.ystart:
            max_q_value = self.pit_reward
        else:  
            max_q_value =max([ self.q_table[self.checkMove(pos, move)][action] for action in self.getActions()])
        return self.getReward(pos) + self.gamma * max_q_value
    
    def getOptimalLength(self, pos, move):
        #returns the optimal q-value for the state
        if self.board.isPiece(self.checkMove(pos, move)) and self.board.findPiecebyPOS(self.checkMove(pos, move)).name=='Goal':
            min_length = 0
        elif self.board.isPiece(self.checkMove(pos, move)) and self.board.findPiecebyPOS(self.checkMove(pos, move)).name=='Pit' or self.board.isPiece(self.checkMove(pos, move)) and self.board.findPiecebyPOS(self.checkMove(pos, move)).name=='Wall' or self.checkMove(pos, move)[0] > self.xstart + self.xsize -1 or self.checkMove(pos, move)[1] > self.ysize + self.ystart -1 or self.checkMove(pos, move)[0] < self.xstart or self.checkMove(pos, move)[1] < self.ystart:
            min_length = inf
        else:  
            min_length =min([ self.l_table[self.checkMove(pos, move)][action] for action in self.getActions()])
        return 1+ min_length

    def Success(self):
        if self.board.findPiecebyName('Player')[0].pos in [item.pos for item in self.board.findPiecebyName('Goal')]:
            return True
        else:
            return False
        
    def Fail(self):
        if self.board.findPiecebyName('Player')[0].pos in [item.pos for item in self.board.findPiecebyName('Pit')] or self.board.findPiecebyName('Player')[0].pos in [item.pos for item in self.board.findPiecebyName('Wall')] or self.board.findPiecebyName('Player')[0].pos[0] > (self.xstart + self.xsize-1) or self.board.findPiecebyName('Player')[0].pos[1] > (self.ystart + self.ysize-1) or self.board.findPiecebyName('Player')[0].pos[0] < self.xstart or self.board.findPiecebyName('Player')[0].pos[1] < self.ystart:
            return True
        else:
            return False
    def Done(self):
        if not self.Success() and not self.Fail():
            return False
        else:
            return True
    def getActions(self):
        return self.actions