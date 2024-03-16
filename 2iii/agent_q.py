"""Performs Q learning for the elevator problem"""

import numpy as np
import matplotlib.pyplot as plt
from .. ENVIRONMENT.globals import *
from .environment import EnvironmentModel
from .. HELPER.helper import *


# CONSTANTS FOR 2iii
ARRIVAL_RATE = .025


class QLearningAgent:
    """
    Agent implments Q learning and chooses actions to learn from the 
    dual elevator envirnoment model.
    """

    def __init__(self, qtable, alpha, gamma, epsilon):
        """
        Initializes agent object with the neccesary information

        Args:
            qtable (dict): qtable of values for state-action pairs
            alpha (float): learning rate
            gamma (float): discount factor
            epsilon (float): exploration rate
        """
        self.q_table = qtable
        self.alpha = alpha
        self.gamma = gamma
        self.explore = epsilon
        self.exploit = 1 - self.explore
        self.iterations = ITERATIONS 
        self.start_state = START_STATE
        self.env = EnvironmentModel(self.start_state)
        self.avg_wait_times = [] # avg time to find exit floor
        self.iteration_list = [i*TIMESTEP for i in range(ITERATIONS)]
        self.avg_rewards = [] # avg rewards per iteration
        self.rewards = [] # reward per iteration

    def policy(self, state):
        """
        Chooses action based on epsilon-greedy method.
        Args: state (state): snapshot of environment
        Returns: action: action to take, either random or best action.
        """

        strategy = np.random.choice(STRATEGY, 1, p=[self.explore, self.exploit])

        if strategy == EXPLORE:
            return tuple([(np.random.choice(ACTION_SET), 'A'), (np.random.choice(ACTION_SET), 'B')])
        
        else:
            max_actions = max(self.q_table[state].items(), key=lambda x: x[1])
            return tuple([(max_actions[0][0]),(max_actions[0][1])])

    def q_learn(self):
        """
        Performs q learning to find optimal actions for each elevator at given timesteps
        """
        # 1. Initialize s
        state = self.start_state

        print_state(state, None, 0, 0)
        print_environment(state, self.env.current_time)

        # 2. Continue for certain iterations or convergence
        for _ in range(self.iterations):

            # 3. Choose action a from s using policy
            action = self.policy(state)

            # 4. Take action a and observe reward and s'
            next_state, reward = self.env.step(state, action)
          
            # 5. Q(s,a) = Q(s,a) + alpha * [r + gamma * max_a'(Q(s',a') - Q(s,a))]
            q_new = self.q_table[state][action] + self.alpha * (reward + self.gamma*max(self.q_table[next_state].values()) - self.q_table[state][action])
            self.q_table[state][action] = q_new
            
            print_state(state, next_state, reward, action)
            print_environment(next_state, self.env.current_time)

            # 6. s = s'
            state = next_state
            
            # Keep track of learning
            self.rewards.append(reward)
            self.avg_wait_times.append((sum(self.env.total_wait_times_list)+1)/(self.env.total_exits+1))
            self.avg_rewards.append(sum(self.rewards) / (_ + 1))
  

alphas = [.1, .3, .5, .9] # learning rates
gammas = [.1, .3, .5, .9] # discounted sums
epsilons = [.1, .3, .5, .9] # exploration rates
agents = [] 

# Fixed variables to compare trials
alpha_fixed = .3
gamma_fixed = .3
epsilon_fixed = .5

# Compare Learning Rates
for i in range(len(alphas)):
    agent = QLearningAgent(QTABLE, alphas[i], gamma_fixed, epsilon_fixed)
    agent.q_learn()
    agents.append(agent)    
    reset_q_table(i)
compare_data(agents, 'alpha', 'Learning Rates', '2iii', 'q/')

# Compare Discounted Sums
agents = []
for i in range(len(gammas)):
    agent = QLearningAgent(QTABLE, alpha_fixed, gammas[i], epsilon_fixed)
    agent.q_learn()
    agents.append(agent)    
    reset_q_table(i)
compare_data(agents, 'gamma', 'Discounted Sums', '2iii', 'q/')

# Compare Epsilon values
agents = []
for i in range(len(epsilons)):
    agent = QLearningAgent(QTABLE, alpha_fixed, gamma_fixed, epsilons[i])
    agent.q_learn()
    agents.append(agent)    
    reset_q_table(i)
compare_data(agents, 'explore', 'Epsilon Values', '2iii', 'q/')
