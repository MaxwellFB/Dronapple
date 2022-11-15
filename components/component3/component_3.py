"""
Component 3 - Navigation and suction
"""

from .suction.suction import Suction


class Component3:
    def __init__(self):
        self.suction = Suction(path_model='model/suction_network.pth', path_scaler='model/suction_std_scaler.bin')

        # Only works if load network in the same path that saved when trained
        from tensorforce.agents import Agent
        self.agent = Agent.load(directory='data/agent_checkpoints')
        # print(self.agent.get_specification())
        # print('******************')
        # print(self.agent.get_architecture())
        self.internals = None

    def start_agent(self):
        self.internals = self.agent.initial_internals()

    def act_agent(self, input_agent):
        actions, self.internals = self.agent.act(states=input_agent, internals=self.internals, independent=True)
        # Convert to tuple and Python float
        actions = (actions[0].item(), actions[1].item(), actions[2].item())
        return actions

    def suction_predict(self, bbox, velocity):
        is_suck_apple = self.suction.predict(tuple(bbox + velocity))
        return is_suck_apple