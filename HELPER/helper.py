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

    graph_avg_wait_times(agents, exp_type, title, problem_no, algo)
    graph_avg_rewards(agents, exp_type, title, problem_no, algo)
    print_agent_info(agents, exp_type, problem_no, algo)

def graph_avg_wait_times(agents, exp_type, title, problem_no, algo):
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
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_wait_times), label=str(agents[i].alpha))

    if exp_type == 'g':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_wait_times), label=str(agents[i].gamma))

    if exp_type == 'e':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_wait_times), label=str(agents[i].explore))
    
    directory = "./project_1/output/"
    os.makedirs(directory, exist_ok=True)
    filename = str(f"{directory}times_{algo}_exp_{exp_type}_problem_{problem_no}.png")
    plt.legend()
    plt.xlabel("Iterations (#)")
    plt.ylabel("Average time to transport person (s)")
    plt.title(f"AVG Time VS Iterations for {title}")
    plt.savefig(filename) 
    #plt.show()
    plt.clf()
    
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
    #plt.show()
    plt.clf()


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

def reset_trace_table(e_trace):
    for state in e_trace:
        for action_pair in e_trace[state]:
            e_trace[state][action_pair] = 0

def print_state(state, next_state, reward, action):

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

def print_environment(state, curr_time):
    """
    Prints the current envirnoment for each timestep
    Args: state (state): snapshot of envirnoment
    """
    # Parse state information
    elevator_A_state, elevator_B_state, passenger_info = state
    
    # Information about each elevator
    a_elev, a_floor, a_door  = elevator_A_state
    b_elev, b_floor, b_door  = elevator_B_state

    # Information on arriving passengers
    p1, p2 = passenger_info
    p1_call, p1_exit, p1_loc = p1
    p2_call, p2_exit, p2_loc = p2

    print("TIME =", curr_time)

    for floor in range(NFLOORS,0, -1):
        print(floor,"-", end="")
        if a_floor == floor:
            print(f"[A", end="")
            
            if p1_loc == IN_A:
                print(f"[P1_A] = {p1_call, p1_exit}", end="")

            if p2_loc == IN_A:
                print(f" [P2_A] = {p2_call, p2_exit}", end="")

            if a_door == OPEN:
                print("] Open ", end ="")

            if a_door == CLOSED:
                print("] Closed ", end ="")
            
        if b_floor == floor:
            print(f"[B", end="")
            
            if p1_loc == IN_B:
                print(f"[P1_B] = {p1_call, p1_exit}", end="")

            if p2_loc == IN_B:
                print(f"[P2_B] = {p2_call, p2_exit}", end="")

            if b_door == OPEN:
                print("] Open ", end ="")

            if b_door == CLOSED:
                print("] Closed ", end ="")

        if p1_loc == WAITING and floor == p1_call:
            print(f"[Wait P1 = {p1_call, p1_exit}]", end="")

        if p2_loc == WAITING and floor == p2_call:
            print(f"[Wait P2 = {p2_call, p2_exit}]", end="")
        
        print()
    print("-------------------------------------------------------------------")