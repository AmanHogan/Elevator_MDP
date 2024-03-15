
import sys
sys.dont_write_bytecode = True
import numpy as np
import sys

# STATE FORMAT: (('A',<FLOOR>,<DOOR>),('B',<FLOOR>,<DOOR>),((<CALL>,<EXIT>,<LOCATION>),(<CALL>,<EXIT>,<LOCATION>)))
# ACTION FORMAT: [(<ELEV ACTION>, 'A'), (<ELEV ACTION, 'B')]
# QTABLE FORMAT: (('A',<F>,<D>),('B',<F>,<D>),((<C>,<E>,<L>),(<C>,<E>,<L>)))[(<E ACTION>,'A'),(<E ACTION, 'B')]

########################### REWARD VARS ###########################
MOVEMENT_REWARD = 10
""" Reward for elevator moving the right direction"""

MOVEMENT_PENALTY = 1
""" Penalty for elevator moving in the wrong direction"""

PASSENGER_PICKUP_REWARD = 10
"""Reward for picking up a passenger"""

DOOR_OPEN_PENALTY = 2
""" Penalty for opening the door when there is no one there"""

PASSENGER_DROP_OFF_REWARD = 10
""" Reward for dropping off a passenger on their call floor """

DOOR_HOLD_PENALTY = 2
""" Penalty for holding the door for no reason """

########################### ELEVATOR VARS ###########################
NFLOORS = 6
"""Number of floors in envirnoment"""

ARRIVAL_RATE = .1
"""Rate at which people arrive per second"""

START_FLOORS = [1]
""" The floors that people call from """

START_PROB = [1]
""" Chance of someone calling from that floor """

EXIT_FLOORS = [2,3,4,5,6]
""" The floors that people exit from """

EXIT_PROB = [.20, .20, .20, .20, .20]
""" Chance of someone exiting from that floor """

FLOORS = [1, 2, 3,4,5,6]
"""The floors each elevator will move through"""

FLOORS_ZERO = [0,1,2,3,4,5,6]
"""The floors each elevator will move through with a zero to include non-valid states for computation"""

######################### ELEVATOR CONSTANTS (DO NOT CHANGE) ##############################
WAITING = 0
""" Person is waiting outside the elevator"""

IN_A = 1
"""Person is inside elevator A"""

IN_B = 2
"""Person is inside elevator B"""

OPEN = True
""" Door is open"""

CLOSED = False
""" Door is closed """

START_STATE = (('A', 1, False), ('B', 1, False), ((0,0,WAITING),(0,0,WAITING)))
"""Format: (('A',<FLOOR>,<DOOR>),('B',<FLOOR>,<DOOR>),((<CALL>,<EXIT>,<LOCATION>),(<CALL>,<EXIT>,<LOCATION>)))"""

ACTION_SET = ['UP', 'DOWN', 'HOLD', 'DOORS']
"""The simple actions the agent can take: UP, DOWN, HOLD, and DOORS"""

ACTION_SPACE = np.array(np.zeros((len(ACTION_SET), len(ACTION_SET))), dtype='object')
""" The action + elevator pair comprising of all actions that can be taken. EX: (('UP', 'A'), ('DOWN', 'B'))"""
for action_a in range(len(ACTION_SET)):
    for action_b in range(len(ACTION_SET)):
        ACTION_SPACE[action_a][action_b] = (ACTION_SET[action_a], 'A'), (ACTION_SET[action_b], 'B')

######################### SIMULATOR VARS ##############################

ITERATIONS = 3000
""" Number of steps in the simulation """

TIMESTEP = 5
""" How many seconds pass for each timestep"""

STRATEGY = ['explore', 'exploit']
EXPLORE = 'explore'
EXPLOIT = 'exploit'

###################### Q LAMBDA AND SARSA LAMBDA VARIABLES ######################
# FOR Q LAMBDA AND SARSA LAMDA LOOK AT THIS NOTE
# NOTE: FOR Q LAMDA AND SARSA LAMBDA: Because of the large state 
# space and size of the e(s,a), it will take a
# long time using 6 floors. Uncomment this code to 
# see the q(lambda) algoirthm with a reduced state 
# space size (Going from 6 floors to 3)

# NFLOORS = 3
# """Number of floors in envirnoment"""

# ARRIVAL_RATE = .1
# """Rate at which people arrive per second"""

# START_FLOORS = [1]
# """ The floors that people call from """

# START_PROB = [1]
# """ Chance of someone calling from that floor """

# ITERATIONS = 10
# """ Number of steps in the simulation """

# EXIT_FLOORS = [2,3]
# """ The floors that people exit from """

# EXIT_PROB = [.5, .5]
# """ Chance of someone exiting from that floor """

# FLOORS = [1, 2, 3]
# """The floors each elevator will move through"""

# FLOORS_ZERO = [0, 1, 2, 3]
# """The floors each elevator will move through with a zero to include non-valid states for computation"""

############################################ QTABLE ############################################
QTABLE = {}
"""
Q-table that keeps track of the avg rewards of each state-action value.
"""
print(f"Initializing new Q-Table. This takes about 30 seconds ...")
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
                                        QTABLE[state] = {}
                                        for action_pair in ACTION_SPACE.flatten():
                                            QTABLE[state][action_pair] = 0
