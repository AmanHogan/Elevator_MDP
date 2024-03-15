"""Performs SARSA lambda learning given the elevator problem. Change the variables in globals.py for your paramaters. """

import numpy as np
import matplotlib.pyplot as plt
from .. ENVIRONMENT.globals import *
from .. ENVIRONMENT.environment import EnvironmentModel
from .. HELPER.helper import *

# NOTE: To speed up efficiency, reduce the state space using the isntructions in globals.py

# CONSTANTS FOR 2i
ARRIVAL_RATE = .1
START_FLOORS = [1]
START_PROB = [1]
EXIT_FLOORS = [2,3,4,5,6]
EXIT_PROB = [.20, .20, .20, .20, .20]

class SARSALambdaAgent:
    """
    Agent implements SARSA(lambda) learning and chooses actions to learn from the 
    dual elevator environment model. 
    """

    def __init__(self, qtable, alpha, gamma, epsilon, lambda_val):
        self.q_table = qtable
        self.alpha = alpha
        self.gamma = gamma
        self.explore = epsilon
        self.exploit = 1 - self.explore
        self.iterations = ITERATIONS
        self.start_state = START_STATE
        self.env = EnvironmentModel(self.start_state)
        self.avg_wait_times = []
        self.iteration_list = [i*TIMESTEP for i in range(ITERATIONS)]
        self.avg_rewards = []
        self.rewards = []
        self.lambda_val = lambda_val  # Lambda value for eligibility traces
        self.e_trace = {} # eligibiltiy trace
        self.init_e_trace() # populates e trace with 0
        self.visited_pairs = set() # contains visited state-action pairs

    def init_e_trace(self):
        """
        Initialize eligibility traces for all state-action pairs.
        """

        print("Initializing Eligibility Trace ... ")
        for floor_A in FLOORS:
            for floor_B in FLOORS:
                for door_A in [True, False]:
                    for door_B in [True, False]:
                        for call_floor_A in FLOORS_ZERO:
                            for exit_floor_A in FLOORS_ZERO:
                                for call_floor_B in FLOORS_ZERO:
                                    for exit_floor_B in FLOORS_ZERO:
                                        for loc_a in [IN_A, IN_B, WAITING]:
                                            for loc_b in [IN_A, IN_B, WAITING]:
                                                state = (('A', floor_A, door_A), ('B', floor_B, door_B), ((call_floor_A, exit_floor_A, loc_a), (call_floor_B, exit_floor_B, loc_b)))
                                                self.e_trace[state] = {}
                                                for action_pair in ACTION_SPACE.flatten():
                                                    self.e_trace[state][action_pair] = 0

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
    
    def sarsa_lambda_learn(self):
        """
        Performs SARSA(lamda) learning to find optimal actions for each elevator at given timesteps
        """

        # 1. Initialize a, a
        state = self.start_state
        action = self.policy(state)

        print_state(state, None, 0, 0)
        print_environment(state, self.env.current_time)

        # 2. Continue for certain iterations or convergence
        for _ in range(self.iterations):
                        
            # 3. Take action a and observe r and s'
            next_state, reward = self.env.step(state, action)

            # 4. Choose a' from s' using policy
            next_action = self.policy(next_state)

            # 5. td = r + gamma * Q(s',a') - Q(s,a)
            td = reward + self.gamma * self.q_table[next_state][next_action] - self.q_table[state][action]

            # 6. e(s,a) = e(s,a) + 1
            self.e_trace[state][action] = self.e_trace[state][action] + 1
            
            # 7. for all s, a
            print("Updating Q(s,a) by looping through all (s,a). This may take some time ...")
            for s in self.q_table:
                for a in self.q_table[s]:

                    # 8. Q(s,a) = Q(s,a) + alpha * td * e(s,a)
                    self.q_table[s][a] += self.alpha * td * self.e_trace[s][a]

                    # 9. e(s,a) = gamma * lambda * e(s,a)
                    self.e_trace[s][a] = self.gamma * self.lambda_val * self.e_trace[s][a] 
            
            print_state(state, next_state, reward, action)
            print_environment(next_state, self.env.current_time)

            # 10. s = s', a = a'
            state = next_state
            action = next_action

            self.rewards.append(reward)
            self.avg_wait_times.append((agent.env.current_time + TIMESTEP) / (agent.env.t_l + 1))
            self.avg_rewards.append(sum(self.rewards) / (_ + 1))


alphas = [.01,.1,.3,.5,1] # learning rates
gammas = [.01,.1,.3,.5,1] # discounted sums
epsilons = [.1,.3,.5,.8,1] # exploration rates
agents = [] 

# Fixed variables to compare trials
alpha_fixed = .1
gamma_fixed = 1
epsilon_fixed = .5
lambda_val = .5 
        
# Compare Learning Rates
for i in range(len(alphas)):
    agent = SARSALambdaAgent(QTABLE, alphas[i], gamma_fixed, epsilon_fixed, lambda_val)
    agent.sarsa_lambda_learn()
    agents.append(agent)    
    reset_q_table(i)
    reset_trace_table(agent.e_trace)
compare_data(agents, 'a', 'Learning Rates', '2i', 'sarsa_lam')

# Compare Discounted Sums
agents = []
for i in range(len(gammas)):
    agent = SARSALambdaAgent(QTABLE, alpha_fixed, gammas[i], epsilon_fixed,lambda_val)
    agent.sarsa_lambda_learn()
    agents.append(agent)
    reset_q_table(i)
    reset_trace_table(agent.e_trace)
compare_data(agents, 'g', 'Discounted Sums', '2i', 'sarsa_lam')

# Compare Epsilon values
agents = []
for i in range(len(epsilons)):
    agent = SARSALambdaAgent(QTABLE, alpha_fixed, gamma_fixed, epsilons[i],lambda_val)
    agent.sarsa_lambda_learn()
    agents.append(agent)    
    reset_q_table(i)
    reset_trace_table(agent.e_trace)
compare_data(agents, 'e', 'Epsilon Values', '2i', 'sarsa_lam')