import numpy as np
from environment import Environment
from globals import * 
from collections import defaultdict
from itertools import product
from joblib import Parallel, delayed

WAITING = 0
IN_A = 1
IN_B = 2

class QLearningAgent:

    def __init__(self, action_set, action_space, floors, qtable):

        self.actions = action_set # set of actions [UP, DOWN, HOLD, DOORS]
        self.floors = floors # set of floors

        # avg reward values given action and state, ex: q_table[(('A', 1), ('B', 1), (0,0))][(('UP', 'A'), ('DOWN', 'B'))] = 0
        self.q_table = qtable

        # space of all possible combinations of actions, EX: [(('UP', 'A'), ('DOWN', 'B'))]
        self.action_space = action_space

        self.alpha = .01 # learnging rate
        self.gamma = 1 # discounted sum of future rewards

        # epsilon greedy strategy
        self.explore = 1 
        self.exploit = 1 - self.explore
        self.epochs = 10

        # Initial state
        self.start_state = START_STATE

        # Environment object
        self.env = Environment(self.start_state)

        self.init_q_table()

    def init_q_table(self):
        """
        initializes populates q-table with values of 0
        """

        # Initialize Q-table
        for floor_A in self.floors:
            for floor_B in self.floors:
                for door_A in [True, False]:  # 0 for open, 1 for closed
                    for door_B in [True, False]:  # 0 for open, 1 for closed
                        for call_floor_A in self.floors:
                            for exit_floor_A in self.floors:
                                for call_floor_B in self.floors:
                                    for exit_floor_B in self.floors:
                                        for loc_a in [IN_A, IN_B, WAITING]:
                                            for loc_b in [IN_A, IN_B, WAITING]:
                                                # Remove current floor for each person from the tuple
                                                state = (('A', floor_A, door_A), ('B', floor_B, door_B), ((call_floor_A, exit_floor_A, loc_a), (call_floor_B, exit_floor_B, loc_b)))
                                                self.q_table[state] = {}
                                                for action_pair in self.action_space.flatten():
                                                    self.q_table[state][action_pair] = 0



    def q_learn(self):
        """
        performs q learning to find optimal actions for each elevator at given timesteps
        """
        state = self.start_state
        # self.env.print_info()

        for _ in range(self.epochs):
            
            # Agent chooses action based on epsilon greedy and q-table
            action = self.get_action(state)

            # Agent Executes action and Environment returns next state based on state-action
            next_state, reward = self.env.simulate_action(state, action)
            
            print(f"OLD STATE = {state}")
            print(f"REWARD = {reward} | ACTION = {action}")
            print(f"NEW STATE = {next_state}")
            print("______________________")
            
            self.env.print_info(next_state)
            

            # Environment also returns reward for state-action
            # TODO: reward(next)

            # Agent updates q table based on reward
            # TODO:

            state = next_state

            pass

    def get_action(self, state):
        """
        Gets an action following epsilon greedy strategy. Explores at a given p, exploits (chooses best)
        action given state, with p - 1.

        Args: state (('A', 1), ('B', 1), (0,0)): elevator - floor - passenger_floors tuple

        Returns: action [(('UP', 'A'), ('DOWN', 'B'))]: best action or explored action
        """

        strategy = np.random.choice(['explore', 'exploit'], 1, p=[self.explore, self.exploit])

        if strategy == 'explore':
            return [(np.random.choice(self.actions), 'A'), (np.random.choice(self.actions), 'B')]
        else:
            max_actions = max(self.q_table[state].items(), key=lambda x: x[1])
            return [(max_actions[0][0]),(max_actions[0][1])]


# EXAMPLE USAGE: agent.q_table[(('A', 1), ('B', 1), ((0,0),(0,0)))][(('UP', 'A'), ('DOWN', 'B'))]
agent = QLearningAgent(ACTION_SET, ACTION_SPACE, FLOORS, {})
agent.q_learn()
