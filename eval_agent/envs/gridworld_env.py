import re
import json
import logging
from typing import Tuple, Optional

from eval_agent.envs import BaseEnv
from eval_agent.tasks import GridworldTask
from eval_agent.prompt import prompt_with_icl
from eval_agent.utils.datatypes import State
from gridworld.gridworld_agent.envs import GridworldTextEnv, GridworldTextEnvSingle


logger = logging.getLogger("agent_frame")


class GridworldEnv(BaseEnv):
    def __init__(
        self,
        task: GridworldTask,
        env: GridworldTextEnv,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: GridworldTask = task
        self.session_id = self.task.session_id
        self.session = {}
        self.env = env

        if kwargs.get("dark"):
            self.mode = kwargs.get("mode", -100)
            if self.mode == 0:
                self.env.add_current_coord = False
                self.env.add_possible_action = True
                self.env.add_possible_coord = False
            elif self.mode == 1:
                self.env.add_current_coord = True
                self.env.add_possible_action = True
                self.env.add_possible_coord = False
            elif self.mode == 2:
                self.env.add_current_coord = True
                self.env.add_possible_action = True
                self.env.add_possible_coord = True
            elif self.mode == -100:
                print("\n\ndark_mode is NOT set: Working as default for dark:\n'0' Possible Action\n\n")
                self.env.add_current_coord = False
                self.env.add_possible_action = True
                self.env.add_possible_coord = False
            else:
                ValueError(f"GridWorld-DARK supports [0, 1, 2] not {self.mode}")
        else:
            self.mode = kwargs.get("mode", -100)
            if self.mode == 0:
                self.env.add_current_coord = False
                self.env.add_possible_action = False
                self.env.add_possible_coord = False
            elif self.mode == 1:
                self.env.add_current_coord = False
                self.env.add_possible_action = True
                self.env.add_possible_coord = False
            elif self.mode == 2:
                self.env.add_current_coord = True
                self.env.add_possible_action = False
                self.env.add_possible_coord = False
            elif self.mode == 3:
                self.env.add_current_coord = True
                self.env.add_possible_action = True
                self.env.add_possible_coord = False
            elif self.mode == 4:
                self.env.add_current_coord = True
                self.env.add_possible_action = False
                self.env.add_possible_coord = True
            elif self.mode == 5:
                self.env.add_current_coord = True
                self.env.add_possible_action = True
                self.env.add_possible_coord = True
            elif self.mode == -100:
                print("\n\nbasic_mode is NOT set: Working as default for basic:\n5 Current Coord + Possible Coord + Possible Action\n\n")
                self.env.add_current_coord = True
                self.env.add_possible_action = True
                self.env.add_possible_coord = True
            else:
                raise ValueError(f"GridWorld supports [0, 1, 2, 3, 4, 5] not {self.mode}")
        
        self.state = State()
    
    def parse_action(self, llm_output: str) -> str:
        if 'Plan:\n' in llm_output:
            llm_output = llm_output.split('Plan:\n')[1]
            llm_output = llm_output.replace("<think>", "")
        elif 'Action:\n' in llm_output:
            llm_output = llm_output.split('Action:\n')[1]
            llm_output = llm_output.replace("<think>", "")
        else:
            if llm_output in ['up', 'down', 'left', 'right']:
                llm_output = llm_output
            else:
                llm_output = "invalid action"
            
        # jw: ERROR BY LENGTH (mapped into 'pit'')
        return llm_output
    
    def step(self, llm_output: str, prob: Optional[str] = None) -> Tuple[str, State]:
        self.state.history.append({
            "role": "assistant",
            "content": llm_output,
            "prob": prob
        })
        try:
            action = self.parse_action(llm_output)
            print('action:', action)

        except:
            observation = f"Invalid format. The input must be one of up, down, left, or right."
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.state.reward = -1
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            return observation, self.state
        try:
            observation, reward, done, info = self.env.step(action=action)
            observation = f"{observation}"
        except:
            observation = 'Invalid action!'
            done = False

        self.state.history.append({
            "role": "user",
            "content": f"{observation}",
            "prob": prob
        })
        self.state.steps += 1
        if self.state.steps >= self.max_steps or llm_output=="":
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = 0
        elif type(self.env) != GridworldTextEnvSingle and action not in self.env.game.getActions():
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "invalid_action"
            self.state.reward = -1
        elif done:
            if reward == 1:
                self.state.success = True
                self.state.terminate_reason = "success"
            elif reward == -1:
                self.state.success = False
                self.state.terminate_reason = "pit"
            self.state.finished = True
            self.state.reward = reward

        return observation, self.state
    
    def reset(self, n_icl=0) -> Tuple[str, State]:
        self.state = State()
        self.env.reset(self.session_id)
        cur_task = self.env.observation

        observation, messages = prompt_with_icl(self.instruction, self.raw_icl, cur_task, n_icl)
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state
