import numpy as np
import matplotlib.pyplot as plt
from .. ENVIRONMENT.globals import *
from .. ENVIRONMENT.environment import EnvironmentModel
from .. HELPER.helper import *

class QLearningAgent:
    """
    Agent implments Q-learning and chooses actions to learn from the 
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

    def update_q_table(self, state, action, reward, new_state):
        """
        Updates the q-table using the bellman update formula for q-learning.

        Args:
            state (state): snapshot of env
            action (action): action taken
            reward (float): reward for taking action a
            new_state (state): new state arrived in from action a
        """

        # Q_current = Q(s,a)
        current_q_value = self.q_table[state][action]

        # max_a [Q(s', a)] || q value from the given list of actions using s'
        max_next_q_value = max(self.q_table[new_state].values())

        # UPDATE:
        # Q_new = Q_current + alpha * (reward + gamma * max_a[Q(s', a)] - Q_curr)
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * max_next_q_value - current_q_value)


        self.q_table[state][action] = new_q_value

    def q_learn(self):
        """
        Performs q learning to find optimal actions for each elevator at given timesteps
        """

        # 1. Initialize start state
        state = self.start_state
        print_state_info(state, None, 0, 0)
        self.env.print_environment(state)

        # NOTE: Each step taken results in a terminal state in my implementation
        # NOTE: Im using iterations rather than episodes which is a valid approach

        # 2. Continue for certain iterations or convergence
        for _ in range(self.epochs):

            # 3. Choose action a from s using policy
            action = self.get_action(state)

            # 4. Take action a and observe reward and s'
            next_state, reward = self.env.get_new_state_and_reward(state, action)
            print_state_info(state, next_state, reward, action)
            self.env.print_environment(next_state)

            # 5. Update Qtable using qlearning bellman update
            self.update_q_table(state, tuple(action), reward, next_state)

            # 6. Set s = s'
            state = next_state
            
            # Keep track of learning
            self.rewards.append(reward)
            self.avg_times.append((self.env.current_time + TIMESTEP)/(self.env.t_l+1))
            self.avg_rewards.append(sum(self.rewards) / (_ + 1))


alphas = [.01,.1,.3,.5,1] # learning rates
gammas = [.01,.1,.3,.5,1] # discounted sums
epsilons = [.1,.3,.5,.8,1] # exploration rates
agents = [] 

# Fixed variables to compare trials
alpha_fixed = .1
gamma_fixed = 1
epsilon_fixed = .5

# Compare Learning Rates
for i in range(len(alphas)):
    agent = QLearningAgent(QTABLE, alphas[i], gamma_fixed, epsilon_fixed)
    agent.q_learn()
    agents.append(agent)    
    reset_q_table(i)
compare_data(agents, 'a', 'Learning Rates', '2i', 'qlearn')

# Compare Discounted Sums
agents = []
for i in range(len(gammas)):
    agent = QLearningAgent(QTABLE, alpha_fixed, gammas[i], epsilon_fixed)
    agent.q_learn()
    agents.append(agent)    
    reset_q_table(i)
compare_data(agents, 'g', 'Discounted Sums', '2i', 'qlearn')

# Compare Epsilon values
agents = []
for i in range(len(epsilons)):
    agent = QLearningAgent(QTABLE, alpha_fixed, gamma_fixed, epsilons[i])
    agent.q_learn()
    agents.append(agent)    
    reset_q_table(i)
compare_data(agents, 'e', 'Epsilon Values', '2i', 'qlearn')

