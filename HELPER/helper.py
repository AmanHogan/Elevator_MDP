import numpy as np
import matplotlib.pyplot as plt
from ..ENVIRONMENT.globals import *

def graph_avg_times(agents, exp_type, title):

    if exp_type =='a':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_times), label=str(agents[i].alpha))

    if exp_type == 'g':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_times), label=str(agents[i].gamma))

    if exp_type == 'e':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_times), label=str(agents[i].explore))
    
   
    plt.legend()
    plt.xlabel("Iterations (#)")
    plt.ylabel("Average time to transport person (s)")
    plt.title(f"AVG Time VS Iterations for {title}")
    plt.show()

def graph_avg_rewards(agents, exp_type, title):
    if exp_type =='a':

        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_rewards), label=str(agents[i].alpha))

    if exp_type == 'g':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_rewards), label=str(agents[i].gamma))

    if exp_type == 'e':
        for i in range(len(agents)):
            plt.plot(np.array(agents[i].iteration_list), np.array(agents[i].avg_rewards), label=str(agents[i].explore))

    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("Average Reward")
    plt.title(f"Average Reward vs Iterations for {title}")
    plt.show()

def print_agent_info(agents):
    
    for i in range(len(agents)):
        print("-------------------------------------------------------")
        print(f"AGENT NUMBER: {i} | LEARNING RATE = {agents[i].alpha}")
        print(f"AGENT NUMBER: {i} | DISCOUNT SUM = {agents[i].gamma}")
        print("Total People that called elevator", agents[i].env.t_p)
        print("Total People who arrived at exit floor", agents[i].env.t_l)
        print(f"AVG time in elevator per person: {round(((agents[i].env.current_time + float(TIMESTEP))/(agents[i].env.t_l+float(1))), 2)} SECS or {round(((agents[i].env.current_time + float(TIMESTEP))/(agents[i].env.t_l+float(1)))/60, 2)} MINS", )
        print(f"Total Simulation/Learning time: {round(agents[i].env.current_time,2)} SECS or {round(agents[i].env.current_time/60,2)} MINS or {round((agents[i].env.current_time/60)/60,2)} HOURS")
        print("-------------------------------------------------------")

def reset_q_table():
    for state in QTABLE:
        for action_pair in QTABLE[state]:
            QTABLE[state][action_pair] = 0