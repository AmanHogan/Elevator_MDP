import numpy as np

ACTION_SET = ['UP', 'DOWN', 'HOLD', 'DOORS']
"""The simple actions the agent can take: UP, DOWN, HOLD, and DOORS"""

ACTION_SPACE = np.array(np.zeros((len(ACTION_SET), len(ACTION_SET))), dtype='object')
""" The action + elevator pair comprising of all actions that can be taken. EX: (('UP', 'A'), ('DOWN', 'B'))"""
for action_a in range(len(ACTION_SET)):
    for action_b in range(len(ACTION_SET)):
        ACTION_SPACE[action_a][action_b] = (ACTION_SET[action_a], 'A'), (ACTION_SET[action_b], 'B')

