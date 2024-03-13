import numpy as np
import matplotlib.pyplot as plt
from .. ENVIRONMENT.globals import *
from .. ENVIRONMENT.environment import EnvironmentModel
from .. GRAPHER.grapher import *

class QLearningAgent:
    
    """
    Agent implments q-learning and chooses actions actions to learn from the 
    dual elevator envirnoment model.
    """

    def __init__(self, action_set, action_space, floors, qtable, floors_zero):
        self.actions = action_set # set of actions [UP, DOWN, HOLD, DOORS]
        self.floors = floors # set of floors
        self.q_table = qtable # keeps track of q-values of state-action pair
        self.action_space = action_space
        self.alpha = .01 # learnging rate
        self.gamma = 1 # discounted sum of future rewards
        self.explore = .5# p that explores other action
        self.exploit = 1 - self.explore # p that chooses best action
        self.epochs = ITERATIONS # number of runs of agent loop
        self.start_state = START_STATE # initial state of system
        self.env = EnvironmentModel(self.start_state) # Models the envirnoment
        self.floors_zero = floors_zero
        self.avg_times = []
        self.init_q_table()
        self.iteration_list = [i*TIMESTEP for i in range(ITERATIONS)]
        self.avg_rewards = []
        self.rewards = []

        # NOTE: This does not use exploration decay

    def init_q_table(self):
        """
        Populates q-table with values of 0
        """
        for floor_A in self.floors:
            for floor_B in self.floors:
                for door_A in [True, False]:
                    for door_B in [True, False]:
                        for call_floor_A in self.floors_zero:
                            for exit_floor_A in self.floors_zero:
                                for call_floor_B in self.floors_zero:
                                    for exit_floor_B in self.floors_zero:
                                        for loc_a in [IN_A, IN_B, WAITING]:
                                            for loc_b in [IN_A, IN_B, WAITING]:
                                                # Remove current floor for each person from the tuple
                                                state = (('A', floor_A, door_A), ('B', floor_B, door_B), ((call_floor_A, exit_floor_A, loc_a), (call_floor_B, exit_floor_B, loc_b)))
                                                #print(state)
                                                self.q_table[state] = {}
                                                for action_pair in self.action_space.flatten():
                                                    self.q_table[state][action_pair] = 0

    def get_action(self, state):
        """
        Chooses action based on epsilon-greedy method.
        Args: state (state): snapshot of environment
        Returns: action: action to take, either random or best action.
        """

        strategy = np.random.choice(STRATEGY, 1, p=[self.explore, self.exploit])

        if strategy == EXPLORE:
            return [(np.random.choice(self.actions), 'A'), (np.random.choice(self.actions), 'B')]
        
        else:
            max_actions = max(self.q_table[state].items(), key=lambda x: x[1])
            return [(max_actions[0][0]),(max_actions[0][1])]

    def print_agent(self, state, next_state, reward, action):
        print("-------------------------------------------------------------------")
        print(f"OLD STATE = {state}")
        print(f"REWARD = {reward} | ACTION = {action}")
        print(f"NEW STATE = {next_state}")

    def update_q_table(self, state, action, reward, new_state):
        # Parse state and action
        # Calculate the new Q-value using the Bellman equation
        current_q_value = self.q_table[state][action]
        max_next_q_value = max(self.q_table[new_state].values())
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * max_next_q_value - current_q_value)
        
        # Update the Q-table entry for the current state-action pair
        self.q_table[state][action] = new_q_value

    def q_learn(self):
        """
        Performs q learning to find optimal actions for each elevator at given timesteps
        """

        state = self.start_state
        self.print_agent(state, None, 0, 0)
        self.env.print_environment(state)

        # Continue for certain epochs or convergence
        for _ in range(self.epochs):

            action = self.get_action(state)
            next_state, reward = self.env.get_new_state_and_reward(state, action)
            
            self.print_agent(state, next_state, reward, action)
            self.env.print_environment(next_state)

            self.update_q_table(state, tuple(action), reward, next_state)

            state = next_state

            self.rewards.append(reward)
            self.avg_times.append((self.env.current_time + TIMESTEP)/(self.env.t_l+1))
            self.avg_rewards.append(sum(self.rewards) / (_ + 1))

