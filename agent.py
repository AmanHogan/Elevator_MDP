import numpy as np
from environment import Environment
from globals import (ACTION_SET, ACTION_SPACE, FLOORS)

class QLearningAgent:

    def __init__(self, action_set, action_space, floors, qtable):

        self.actions = action_set
        self.floors = floors
        self.q_table = qtable
        self.action_space = action_space
        self.alpha = .01
        self.gamma = 1
        self.explore = 1
        self.exploit = 1 - self.explore
        self.epochs = 100
        self.start_state = (('A', 1), ('B', 1))  # Initial state without passengers
        self.env = Environment(self.start_state)
        self.init_q_table()



    def init_q_table(self):
        # Initialize Q-table
        for floor_A in self.floors:
            for floor_B in self.floors:
                state = (('A', floor_A), ('B', floor_B))
                self.q_table[state] = {}
                for action_pair in self.action_space.flatten():
                    self.q_table[state][action_pair] = 0


    def learn(self):
        state = self.start_state
        self.env.print_info()

        for _ in range(self.epochs):
            
            # Agent chooses action based on epsilon greedy and q-table
            action = self.get_action(state)

            # Agent Executes action and Environment returns next state based on state-action
            next_state, reward = self.env.simulate_action(state, action)
            
            print(f"OLD STATE = {state}")
            print(f"REWARD = {reward} | ACTION = {action}")
            print(f"NEW STATE = {next_state}")
            print(f"DOORS OPENED A = {self.env.elevatorA.isDoorOpen}, B = {self.env.elevatorB.isDoorOpen}")
            self.env.print_info()
            

            # Environment also returns reward for state-action
            # TODO: reward(next)

            # Agent updates q table based on reward
            # TODO:

            state = next_state

    def get_action(self, state):
        strategy = np.random.choice(['explore', 'exploit'], 1, p=[self.explore, self.exploit])

        if strategy == 'explore':
            return [
                (np.random.choice(self.actions), 'A'),
                (np.random.choice(self.actions), 'B')
            ]
        else:
            
            max_actions = max(self.q_table[state].items(), key=lambda x: x[1])

            return [
                (max_actions[0][0]),
                (max_actions[0][1])
            ]


agent = QLearningAgent(ACTION_SET, ACTION_SPACE, FLOORS, {})
agent.learn()
