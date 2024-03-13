import numpy as np
import matplotlib.pyplot as plt
from .. ENVIRONMENT.globals import *
from .. ENVIRONMENT.environment import EnvironmentModel
from ..HELPER.helper import *

class QLearningAgent:

    """
    Agent implments Q-learning and chooses actions to learn from the 
    dual elevator envirnoment model.
    """

    def __init__(self, qtable, alpha, gamma, epsilon):
        self.q_table = qtable
        self.alpha = alpha
        self.gamma = gamma
        self.explore = epsilon
        self.exploit = 1 - self.explore
        self.epochs = ITERATIONS
        self.start_state = START_STATE
        self.env = EnvironmentModel(self.start_state)
        self.avg_times = []
        self.iteration_list = [i*TIMESTEP for i in range(ITERATIONS)]
        self.avg_rewards = []
        self.rewards = []

    def get_action(self, state):
        """
        Chooses action based on epsilon-greedy method.
        Args: state (state): snapshot of environment
        Returns: action: action to take, either random or best action.
        """

        strategy = np.random.choice(STRATEGY, 1, p=[self.explore, self.exploit])

        if strategy == EXPLORE:
            return [(np.random.choice(ACTION_SET), 'A'), (np.random.choice(ACTION_SET), 'B')]
        
        else:
            max_actions = max(self.q_table[state].items(), key=lambda x: x[1])
            return [(max_actions[0][0]),(max_actions[0][1])]

    def print_agent(self, state, next_state, reward, action):
        print("-------------------------------------------------------------------")
        print(f"OLD STATE = {state}")
        print(f"REWARD = {reward} | ACTION = {action}")
        print(f"NEW STATE = {next_state}")
        print("-------------------------------------------------------------------")

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


alphas = [.01,.1,.3,.5,1]
gammas = [.01,.1,.3,.5,1]
epsilons = [.5, .5, .5, .5, .5]
agents = []
alpha_fixed = .1
gamma_fixed = 1
epsilon_fixed = .5



####################### Experiments for Learning Rates #################################
for i in range(len(alphas)):
    agent = QLearningAgent(QTABLE, alphas[i], gamma_fixed, epsilon_fixed)
    agent.q_learn()
    agents.append(agent)
    print("Finished simulation for Agent: ", i)
    print("Resetting QTBALE to 0. This may take some time ...")
    reset_q_table()
    print("Finsihed resetting QTBALE to 0.")

graph_avg_times(agents, 'a', "alpha values")
graph_avg_rewards(agents, 'a', "alpha values")
print_agent_info(agents)
#########################################################################################

####################### Experiments for Discounted Sums #################################

agents = []

for i in range(len(gammas)):
    agent = QLearningAgent(QTABLE, alpha_fixed, gammas[i], epsilon_fixed)
    agent.q_learn()
    agents.append(agent)
    print("Finished simulation for Agent: ", i)
    print("Resetting QTBALE to 0. This may take some time ...")
    reset_q_table()
    print("Finsihed resetting QTBALE to 0.")

graph_avg_times(agents, 'g', "gamma values")
graph_avg_rewards(agents, 'g', "gamma values")
print_agent_info(agents)
#########################################################################################




    
