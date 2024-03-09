import numpy as np
from environment import Environment

class QLearningAgent:

    def __init__(self, actions, floors, qtable):
        self.actions = actions
        self.floors = floors
        self.q_table = qtable
        self.alpha = .01
        self.gamma = 1
        self.explore = .5
        self.exploit = 1 - self.explore
        self.epochs = 3
        self.start_state = (('A', 1), ('B', 1))  # Initial state without passengers
        self.env = Environment(self.start_state)
        self.init_q_table()

    def init_q_table(self):
        for floor_A in self.floors:
            for floor_B in self.floors:
                state = (('A', floor_A), ('B', floor_B))
                self.q_table[state] = {}
                for action_A in self.actions:
                    for action_B in self.actions:
                        action_pair = (action_A, 'A'), (action_B, 'B')
                        self.q_table[state][action_pair] = 0

    def learn(self):
        state = self.start_state
        self.env.print_info()

        for _ in range(self.epochs):
            
            # Environment chooses passengers based on arrival rate
            

            # Agent chooses action based on epsilon greedy and q-table
            action = self.get_action(state)
            print("Action Chosen:", action)

            # Agent Executes action and Environment returns next state based on state-action
            next_state = self.env.execute_action(state, action)
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
        
agent = QLearningAgent(['UP', 'DOWN', 'HOLD', 'DOORS'], [1, 2, 3, 4, 5, 6], {})
agent.learn()
