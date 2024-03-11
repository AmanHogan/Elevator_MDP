
import numpy as np
NFLOORS = 6

ARRIVAL_RATE = .1
START_FLOORS = [1]
START_PROB = [1]

WAITING = 0
IN_A = 1
IN_B = 2

OPEN = True
CLOSED = False


EXIT_FLOORS = [2,3,4,5,6]
EXIT_PROB = [.20, .20, .20, .20, .20]
TIMESTEP = 5

START_STATE = (('A', 1, False), ('B', 1, False), ((0,0,WAITING),(0,0,WAITING)))

ACTION_SET = ['UP', 'DOWN', 'HOLD', 'DOORS']
"""The simple actions the agent can take: UP, DOWN, HOLD, and DOORS"""

ACTION_SPACE = np.array(np.zeros((len(ACTION_SET), len(ACTION_SET))), dtype='object')
""" The action + elevator pair comprising of all actions that can be taken. EX: (('UP', 'A'), ('DOWN', 'B'))"""
for action_a in range(len(ACTION_SET)):
    for action_b in range(len(ACTION_SET)):
        ACTION_SPACE[action_a][action_b] = (ACTION_SET[action_a], 'A'), (ACTION_SET[action_b], 'B')


FLOORS = [1, 2, 3, 4, 5, 6]
"""The floors each elevator will move through"""