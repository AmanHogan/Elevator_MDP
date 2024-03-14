import numpy as np
import os
import matplotlib.pyplot as plt
from ..ENVIRONMENT.globals import *

def compare_data(agents, exp_type, title, problem_no, algo):
    """
    Compares experiment values by controlling for a specifc variable ex:
    alpha, gamma, epsilon

    Args:
        agents (list[agents]): list of agents
        exp_type (str): type of experiment, ex: a, g, e [alpha, gamma, epsilon]
        problem_no (str): problem number ex: 2i
        algo (str): alogirthm used, ex: qlearn, qlambda, sarsa
    """

    graph_avg_times(agents, exp_type, title, problem_no, algo)
    graph_avg_rewards(agents, exp_type, title, problem_no, algo)
    print_agent_info(agents, exp_type, problem_no, algo)

def graph_avg_times(agents, exp_type, title, problem_no, algo):
    """
    Graphs agent avg times to a file for a given experiment

    Args:
        agents (list[agents]): list of agents
        exp_type (str): type of experiment, ex: a, g, e [alpha, gamma, epsilon]
        title (str): title of graph
        problem_no (str): problem number ex: 2i
        algo (str): alogirthm used, ex: qlearn, qlambda, sarsa
    """

    if exp_type =='a':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_times), label=str(agents[i].alpha))

    if exp_type == 'g':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_times), label=str(agents[i].gamma))

    if exp_type == 'e':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_times), label=str(agents[i].explore))
    
    directory = "./project_1/output/"
    os.makedirs(directory, exist_ok=True)
    filename = str(f"{directory}times_{algo}_exp_{exp_type}_problem_{problem_no}.png")
    plt.legend()
    plt.xlabel("Iterations (#)")
    plt.ylabel("Average time to transport person (s)")
    plt.title(f"AVG Time VS Iterations for {title}")
    plt.savefig(filename) 
    plt.show()
    
def graph_avg_rewards(agents, exp_type, title, problem_no, algo):
    """
    Graphs avg reward info to a file for a given experiment

    Args:
        agents (list[agents]): list of agents
        exp_type (str): type of experiment, ex: a, g, e [alpha, gamma, epsilon]
        title (str): title of graph
        problem_no (str): problem number ex: 2i
        algo (str): alogirthm used, ex: qlearn, qlambda, sarsa
    """

    if exp_type =='a':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_rewards), label=str(agents[i].alpha))

    if exp_type == 'g':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_rewards), label=str(agents[i].gamma))

    if exp_type == 'e':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_rewards), label=str(agents[i].explore))
    
    directory = "./project_1/output/"
    os.makedirs(directory, exist_ok=True)
    filename = str(f"{directory}rewards_{algo}_exp_{exp_type}_problem_{problem_no}.png")
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward vs Iterations for {title}")
    plt.savefig(filename) 
    plt.show()

def print_agent_info(agents, exp_type, problem_no, algo):
    """
    Prints agent info to a file for a given experiment

    Args:
        agents (list[agents]): list of agents
        exp_type (str): type of experiment, ex: a, g, e [alpha, gamma, epsilon]
        problem_no (str): problem number ex: 2i
        algo (str): alogirthm used, ex: qlearn, qlambda, sarsa
    """

    directory = "./project_1/output/"
    os.makedirs(directory, exist_ok=True)
    filename = str(f"{directory}{problem_no}_{algo}_{exp_type}") + ".txt"

    file = open(filename,"w")
    for i in range(len(agents)):
        file.write("-------------------------------------------------------\n")
        file.write(f"AGENT NUMBER: {i} | LEARNING RATE = {agents[i].alpha}\n")
        file.write(f"AGENT NUMBER: {i} | DISCOUNT SUM = {agents[i].gamma}\n")
        file.write(f"AGENT NUMBER: {i} | EPSILON VAL = {agents[i].explore}\n")
        file.write("Total People that called elevator {}\n".format(agents[i].env.t_p))
        file.write("Total People who arrived at exit floor {}\n".format(agents[i].env.t_l))
        file.write(f"AVG time in elevator per person: {round(((agents[i].env.current_time + float(TIMESTEP))/(agents[i].env.t_l+float(1))), 2)} SECS or {round(((agents[i].env.current_time + float(TIMESTEP))/(agents[i].env.t_l+float(1)))/60, 2)} MINS\n")
        file.write(f"Total Simulation/Learning time: {round(agents[i].env.current_time,2)} SECS or {round(agents[i].env.current_time/60,2)} MINS or {round((agents[i].env.current_time/60)/60,2)} HOURS\n")
        file.write("-------------------------------------------------------\n")
    file.close()

def reset_q_table(agent_num):
    """
    Resets the qtable to 0 for each value\n
    Args: agent_num (int): agent number
    """

    print("Finished simulation for Agent:", agent_num)
    print("Resetting QTBALE to 0. This takes about 10 secs ...")
    for state in QTABLE:
        for action_pair in QTABLE[state]:
            QTABLE[state][action_pair] = 0

def reset_trace_table(eligibility_traces):
    for state in eligibility_traces:
        for action_pair in eligibility_traces[state]:
            eligibility_traces[state][action_pair] = 0



def print_state_info(state, next_state, reward, action):
    """Prints details about the current state

    Args:
        state (state): snapshot of env
        next_state (state): snapshot of state
        reward (float): reward for takign action a
        action (action): action taken by agent
    """
    print("-------------------------------------------------------------------")
    print(f"OLD STATE = {state}")
    print(f"REWARD = {reward} | ACTION = {action}")
    print(f"NEW STATE = {next_state}")
    print("-------------------------------------------------------------------")