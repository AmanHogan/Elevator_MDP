import numpy as np
import matplotlib.pyplot as plt
from .. ENVIRONMENT.globals import *
from .. ENVIRONMENT.environment import EnvironmentModel
from .. HELPER.helper import *


class QLearningLambdaAgent:
    """
    Agent implements Q(lamdab)-learning and chooses actions to learn from the 
    dual elevator environment model.
    """

    def __init__(self, qtable, alpha, gamma, epsilon, lambda_val):
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
        self.lambda_val = lambda_val  # Lambda value for eligibility traces
        self.eligibility_traces = {} # eligibiltiy trace
        self.init_eligibility_traces()
        self.visited_pairs = set() # contains visited state-action pairs

    def init_eligibility_traces(self):
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
                                                # Remove current floor for each person from the tuple
                                                state = (('A', floor_A, door_A), ('B', floor_B, door_B), ((call_floor_A, exit_floor_A, loc_a), (call_floor_B, exit_floor_B, loc_b)))
                                                #print(state)
                                                self.eligibility_traces[state] = {}
                                                for action_pair in ACTION_SPACE.flatten():
                                                    self.eligibility_traces[state][action_pair] = 0

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
    
    def update_q_values(self, state, action, reward, new_state, new_action):
        """
        Updates the q-table using the bellman update formula for q-lambda-learning.

        Args:
            state (state): snapshot of env
            action (action): action taken
            reward (float): reward for taking action a
            new_state (state): new state arrived in from action a
            new_action (action): new action a
        """

        # td = reward + gamma * Q(s', a*) - Q(s,a)
        temporal_diff = reward + self.gamma * self.q_table[new_state][new_action] - self.q_table[state][action]

        # for all s,a:
        # Q(s,a) = Q(s,a) + alpha * td * eligibility_trace[s][a]
        # Update Q-values for all state-action pairs
        for s in self.q_table:
            for a in self.q_table[s]:
                self.q_table[s][a] += self.alpha * temporal_diff * self.eligibility_traces[s][a]

    

    def update_eligibility_traces(self, state, action):
        """
        Update eligibility traces for visited state-action pairs.
        """
        # If the current state-action pair (state, action) is in self.visited_pairs, 
        # you set the eligibility trace for that pair to 1
        if (state, action) in self.visited_pairs:
            self.eligibility_traces[state][action] = 1

        # iterate over all state-action pairs (s, a) in self.eligibility_traces and decay their values by lambda_val
        # to ensure that the influence of past actions gradually diminishes over time
        else:
            # If not visited, decay the eligibility trace for all state-action pairs
            for s in self.eligibility_traces:
                for a in self.eligibility_traces[s]:
                    self.eligibility_traces[s][a] *= self.lambda_val

    def q_lambda_learn(self):
        """
        Performs Q(lamda)-learning to find optimal actions for each elevator at given timesteps
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

            # 4. Take action a and observe reward r and s'
            next_state, reward = self.env.get_new_state_and_reward(state, action)

            # 5. Choose a' from s' using policy
            next_action = self.get_action(next_state)

            # 6. 
            # Increment eligibility trace for E(s,a)=1
            # And decay when applicaple
            self.update_eligibility_traces(state, tuple(action))
            print_state_info(state, next_state, reward, action)
            self.env.print_environment(next_state)

            # 7. For all state-action pairs
            # Update Q-table using the updated E(s,a) matrix
            self.update_q_values(state, tuple(action), reward, next_state, tuple(next_action))
            self.visited_pairs.add((state, tuple(action)))

            # 8. s = s'
            state = next_state

            self.rewards.append(reward)
            self.avg_times.append((agent.env.current_time + TIMESTEP) / (agent.env.t_l + 1))
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
    agent = QLearningLambdaAgent(QTABLE, alphas[i], gamma_fixed, epsilon_fixed, lambda_val)
    agent.q_lambda_learn()
    agents.append(agent)    
    reset_q_table(i)
    reset_trace_table(agent.eligibility_traces)
compare_data(agents, 'a', 'Learning Rates', '2i', 'q_lam')

# Compare Discounted Sums
agents = []
for i in range(len(gammas)):
    agent = QLearningLambdaAgent(QTABLE, alpha_fixed, gammas[i], epsilon_fixed,lambda_val)
    agent.q_lambda_learn()
    agents.append(agent)
    reset_q_table(i)
    reset_trace_table(agent.eligibility_traces)
compare_data(agents, 'g', 'Discounted Sums', '2i', 'q_lam')

# Compare Epsilon values
agents = []
for i in range(len(epsilons)):
    agent = QLearningLambdaAgent(QTABLE, alpha_fixed, gamma_fixed, epsilons[i],lambda_val)
    agent.q_lambda_learn()
    agents.append(agent)    
    reset_q_table(i)
    reset_trace_table(agent.eligibility_traces)
compare_data(agents, 'e', 'Epsilon Values', '2i', 'q_lam')