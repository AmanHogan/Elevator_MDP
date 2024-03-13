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
        self.explore = .5 # p that explores other action
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
            
    def get_next_action(self, state):
        """
        Chooses next action based on epsilon-greedy method.
        Args: state (state): snapshot of environment
        Returns: action: action to take, either random or best action.
        """

        strategy = np.random.choice(STRATEGY, 1, p=[self.explore, self.exploit])

        if strategy == EXPLORE:
            return [(np.random.choice(self.actions), 'A'), (np.random.choice(self.actions), 'B')]
        
        else:
            max_actions = max(self.q_table[state].items(), key=lambda x: x[1])
            return [(max_actions[0][0]), (max_actions[0][1])]

    def print_agent(self, state, next_state, reward, action):
        print("-------------------------------------------------------------------")
        print(f"OLD STATE = {state}")
        print(f"REWARD = {reward} | ACTION = {action}")
        print(f"NEW STATE = {next_state}")

    def update_q_table(self, state, action, reward, new_state, new_action):
        # Parse state and action
        # Calculate the new Q-value using the SARSA update rule
        current_q_value = self.q_table[state][action]
        next_q_value = self.q_table[new_state][new_action]
        new_q_value = current_q_value + self.alpha * (reward + self.gamma * next_q_value - current_q_value)
        
        # Update the Q-table entry for the current state-action pair
        self.q_table[state][action] = new_q_value

    def sarsa_learn(self):
        """
        Performs SARSA learning to find optimal actions for each elevator at given timesteps
        """

        state = self.start_state
        self.print_agent(state, None, 0, 0)
        self.env.print_environment(state)

        # Continue for certain epochs or convergence
        for _ in range(self.epochs):

            action = self.get_action(state)
            next_state, reward = self.env.get_new_state_and_reward(state, action)
            next_action = self.get_next_action(next_state)
            
            self.print_agent(state, next_state, reward, action)
            self.env.print_environment(next_state)

            self.update_q_table(state, tuple(action), reward, next_state, tuple(next_action))

            state = next_state

            self.rewards.append(reward)
            self.avg_times.append((agent.env.current_time + TIMESTEP)/(agent.env.t_l+1))
            self.avg_rewards.append(sum(self.rewards) / (_ + 1))


# STATE FORMAT: (('A',<FLOOR>,<DOOR>),('B',<FLOOR>,<DOOR>),((<CALL>,<EXIT>,<LOCATION>),(<CALL>,<EXIT>,<LOCATION>)))
# ACTION FORMAT: [(<ELEV ACTION>, 'A'), (<ELEV ACTION, 'B')]
# QTABLE FORMAT: (('A',<F>,<D>),('B',<F>,<D>),((<C>,<E>,<L>),(<C>,<E>,<L>)))[(<E ACTION>,'A'),(<E ACTION, 'B')]
        
# Start learning
agent = QLearningAgent(ACTION_SET, ACTION_SPACE, FLOORS, {}, FLOORS_ZERO)
agent.sarsa_learn()

agent.iteration_list.pop(0)
agent.avg_times.pop(0)
agent.avg_rewards.pop(0)

print("Total People that entered an elevator:", agent.env.t_p)
print("Total People who arrived on their floor:", agent.env.t_l)
print(f"AVG time to find exit per person: {((agent.env.current_time + float(TIMESTEP))/(agent.env.t_l+float(1)))} s or {((agent.env.current_time + float(TIMESTEP))/(agent.env.t_l+float(1)))/60} minutes", )
print(f"Simulated leaning time: {agent.env.current_time } s or {agent.env.current_time/60} minutes or {(agent.env.current_time/60)/60} hours")

plt.plot(np.array(agent.iteration_list), np.array(agent.avg_times))
plt.xlabel("Iterations (#)")
plt.ylabel("Average time to transport person (s)")
plt.title("AVG Time VS Iterations")
plt.show()

# Plot average times
plt.plot(np.array(agent.iteration_list), np.array(agent.avg_rewards))
plt.xlabel("Iterations")
plt.ylabel("Average Reward")
plt.title("Average Reward vs Iterations")
plt.show()