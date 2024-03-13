import numpy as np
import matplotlib.pyplot as plt
from .. ENVIRONMENT.globals import *
from .. ENVIRONMENT.environment import EnvironmentModel
from .. GRAPHER.grapher import *
from agent_q import QLearningAgent


class QLearningLambdaAgent(QLearningAgent):
    """
    Agent implements Q(λ)-learning and chooses actions to learn from the 
    dual elevator environment model.
    """

    def __init__(self, action_set, action_space, floors, qtable, floors_zero, lambda_val):
        super().__init__(action_set, action_space, floors, qtable, floors_zero)
        self.lambda_val = lambda_val  # Lambda value for eligibility traces
        self.eligibility_traces = {}
        self.init_eligibility_traces()

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
    
    def init_eligibility_traces(self):
        """
        Initialize eligibility traces for all state-action pairs.
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
                                                self.eligibility_traces[state] = {}
                                                for action_pair in self.action_space.flatten():
                                                    self.eligibility_traces[state][action_pair] = 0

    def update_eligibility_traces(self, state, action):
        """
        Update eligibility traces for visited state-action pairs.
        """
        action_space = list(ACTION_SPACE.flatten())

        for s in self.eligibility_traces:
            for a in action_space:
                if (state, action) == (s, a):
                    self.eligibility_traces[state][action] = 1
                else:
                    self.eligibility_traces[state][action] *= self.lambda_val



    def q_lambda_learn(self):
        """
        Performs Q(λ)-learning to find optimal actions for each elevator at given timesteps
        """
        state = self.start_state
        self.print_agent(state, None, 0, 0)
        self.env.print_environment(state)

        # Continue for certain epochs or convergence
        for _ in range(self.epochs):

            action = self.get_action(state)
            next_state, reward = self.env.get_new_state_and_reward(state, action)
            next_action = self.get_next_action(next_state)
            print(action)

            self.update_eligibility_traces(state, tuple(action))

            self.print_agent(state, next_state, reward, action)
            self.env.print_environment(next_state)

            self.update_q_values(state, tuple(action), reward, next_state, tuple(next_action))

            state = next_state

            self.rewards.append(reward)
            self.avg_times.append((agent.env.current_time + TIMESTEP) / (agent.env.t_l + 1))
            self.avg_rewards.append(sum(self.rewards) / (_ + 1))

    def update_q_values(self, state, action, reward, new_state, new_action):
        """
        Update Q-values using Q(λ) learning algorithm.
        """
        delta = reward + self.gamma * self.q_table[new_state][new_action] - self.q_table[state][action]
        
        action_space = list(ACTION_SPACE.flatten())
        for s in self.q_table:
            for a in action_space:
                self.q_table[s][a] += self.alpha * delta * self.eligibility_traces[s][a]

        action_space = list(ACTION_SPACE.flatten())
        for s in self.eligibility_traces:
            for a in action_space:
                self.eligibility_traces[s][a] *= self.gamma * self.lambda_val


# STATE FORMAT: (('A',<FLOOR>,<DOOR>),('B',<FLOOR>,<DOOR>),((<CALL>,<EXIT>,<LOCATION>),(<CALL>,<EXIT>,<LOCATION>)))
# ACTION FORMAT: [(<ELEV ACTION>, 'A'), (<ELEV ACTION, 'B')]
# QTABLE FORMAT: (('A',<F>,<D>),('B',<F>,<D>),((<C>,<E>,<L>),(<C>,<E>,<L>)))[(<E ACTION>,'A'),(<E ACTION, 'B')]
        
# Start learning
agent = QLearningLambdaAgent(ACTION_SET, ACTION_SPACE, FLOORS, {}, FLOORS_ZERO, .5)
agent.q_lambda_learn()

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