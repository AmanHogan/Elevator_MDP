"""Models the envirnoment of the dual elevator schedule RL problem"""

import sys
sys.dont_write_bytecode = True
import numpy as np
import time
from ..ENVIRONMENT.globals import * 
import numpy as np

class EnvironmentModel:
    """
    Models the Dual elevator environment and handles state transitions, rewards, and passenger simulation
    """

    def __init__(self, state):
        self.curr_state = state # snapshot of env
        self.current_time = 0 # time that increments by timestep
        self.people = [] # a list of people who are wating on an elevator
        self.MAX_PEOPLE = 2 #max capacity of simualtion
        self.elev_a_passengers = 0
        self.elev_b_passengers = 0
        self.t_p = 0 # total passengers that entered the system
        self.t_l = 0 # total passengers who left the system 

    
    def state_transition(self, state, action):
        """
        Transitions the environment to a new state given actions.
        Only allows valid transitions.

        Args:
            state (state): snapshot of environment
            action (action): action chosen by agent

        Returns:
            next_state: new snapshot of environment
        """

        # Parse state information
        elevator_A_state, elevator_B_state, passenger_info = state
        
        # Information about each elevator
        a_elev, a_floor, a_door  = elevator_A_state
        b_elev, b_floor, b_door  = elevator_B_state

        # Information on each action for each elevator
        action_A, action_B = action

        # Information on arriving passengers
        p1, p2 = passenger_info
        p1_call, p1_exit, p1_loc = p1
        p2_call, p2_exit, p2_loc = p2

        # If action is up, move elevator up
        if action_A[0] == 'UP' and a_floor < NFLOORS and a_door == False:
            elevator_A_state = ('A', a_floor + 1, a_door)

        # If action is down, move elevator down
        elif action_A[0] == 'DOWN' and a_floor > 1 and a_door == False:
            elevator_A_state = ('A', a_floor - 1, a_door)
        
        # If action is hold, add or remove passengers given the state of the door and location
        elif action_A[0] == 'HOLD':
            if a_door == OPEN and p1_loc == WAITING and p1_call == a_floor:
                p1_loc = IN_A
                self.elev_a_passengers += 1
                self.t_p += 1
            
            if a_door == OPEN and p1_loc == IN_A and p1_exit == a_floor:
                self.people.remove((p1_call, p1_exit, WAITING))
                p1_call, p1_exit, p1_loc = 0, 0, WAITING
                self.elev_a_passengers -= 1
                self.t_l += 1

            if a_door == OPEN and p2_loc == WAITING and p2_call == a_floor:
                p2_loc = IN_A
                self.elev_a_passengers += 1
                self.t_p += 1
            
            if a_door == OPEN and p2_loc == IN_A and p2_exit == a_floor:
                self.people.remove((p2_call, p2_exit, WAITING))
                p2_call, p2_exit, p2_loc = 0, 0, WAITING
                self.elev_a_passengers -= 1
                self.t_l += 1


        # If action is doors, add or remove passengers given the state of the door and location
        elif action_A[0] == 'DOORS':
            if a_door == OPEN and p1_loc == WAITING and p1_call == a_floor:
                 p1_loc = IN_A
                 self.elev_a_passengers += 1
                 self.t_p += 1
            
            if a_door == OPEN and p1_loc == IN_A and p1_exit == a_floor:
                self.people.remove((p1_call, p1_exit, WAITING))
                p1_call, p1_exit, p1_loc = 0, 0, WAITING
                self.elev_a_passengers -= 1
                self.t_l += 1


            if a_door == OPEN and p2_loc == WAITING and p2_call == a_floor:
                 p2_loc = IN_A
                 self.elev_a_passengers += 1
                 self.t_p += 1
            
            if a_door == OPEN and p2_loc == IN_A and p2_exit == a_floor:
                self.people.remove((p2_call, p2_exit, WAITING))
                p2_call, p2_exit, p2_loc = 0, 0, WAITING
                self.elev_a_passengers -= 1
                self.t_l += 1


            # Update door state
            elevator_A_state = ('A', a_floor, not a_door)

        # If action is up, move elevator up
        if action_B[0] == 'UP' and b_floor < NFLOORS and b_door == False:
            elevator_B_state = ('B', b_floor + 1, b_door)

        # If action is down, move elevator down
        elif action_B[0] == 'DOWN' and elevator_B_state[1] > 1 and b_door == False:
            elevator_B_state = ('B', b_floor - 1, b_door)

        # If action is hold, add or remove passengers given the state of the door and location
        elif action_B[0] == 'HOLD':
            if b_door == OPEN and p1_loc == WAITING and p1_call == b_floor:
                 p1_loc = IN_B
                 self.elev_b_passengers += 1
                 self.t_p += 1

            if b_door == OPEN and p1_loc == IN_B and p1_exit == b_floor:
                self.people.remove((p1_call, p1_exit, WAITING))
                p1_call, p1_exit, p1_loc = 0, 0, WAITING
                self.elev_b_passengers -= 1
                self.t_l += 1


            if b_door == OPEN and p2_loc == WAITING and p2_call == b_floor:
                 p2_loc = IN_B
                 self.elev_b_passengers += 1
                 self.t_p += 1

            if b_door == OPEN and p2_loc == IN_B and p2_exit == b_floor:
                self.people.remove((p2_call, p2_exit, WAITING))
                p2_call, p2_exit, p2_loc = 0, 0, WAITING
                self.elev_b_passengers -= 1
                self.t_l += 1


        # If action is doors, add or remove passengers given the state of the door and location
        elif action_B[0] == 'DOORS':
            if b_door == OPEN and p1_loc == WAITING and p1_call == b_floor:
                 p1_loc = IN_B
                 self.elev_b_passengers += 1
                 self.t_p += 1

            if b_door == OPEN and p1_loc == IN_B and p1_exit == b_floor:
                self.people.remove((p1_call, p1_exit, WAITING))
                p1_call, p1_exit, p1_loc = 0, 0, WAITING
                self.elev_b_passengers -= 1
                self.t_l += 1


            if b_door == OPEN and p2_loc == WAITING and p2_call == b_floor:
                 p2_loc = IN_B
                 self.elev_b_passengers += 1
                 self.t_p += 1
           
            if b_door == OPEN and p2_loc == IN_B and p2_exit == b_floor:
                self.people.remove((p2_call, p2_exit, WAITING))
                p2_call, p2_exit, p2_loc = 0, 0, WAITING
                self.elev_b_passengers -= 1
                self.t_l += 1

            
            # Update door state
            elevator_B_state = ('B', b_floor, not b_door)

        # Update all variables and repack the values
        p1 = p1_call, p1_exit, p1_loc
        p2 = p2_call, p2_exit, p2_loc
        passenger_info = p1, p2

        new_state = (elevator_A_state, elevator_B_state, passenger_info)
        return new_state

    def step(self, state, action):
        """
        Gets a new state and reward based on passenger simualtion and agent action.\n
        Args:
            state (state): snapshot of environment
            action (action): action chosen by agent
        Returns:
            new_state, reward: the new state of envirnoment and the reward for being in that state
        """

        state_ = self.simulate_people(state)
        next_state = self.state_transition(state_, action)
        next_reward = self.reward_func(state_, action, next_state)

        return next_state, next_reward
    
    def print_environment(self, state):
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

        print("TIME =", self.current_time)

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
        
    def simulate_people(self, state):
        """
        Generates new people with Probability distribution of call and exiting elevator.
        Adds these people to the the people list in envirnoment. Max of 2 people in system
        """


        elevator_A_state, elevator_B_state, p_floors = state
        new_people = []

        # If there are 2 or more people waiting skip simulation
        if len(self.people) >= self.MAX_PEOPLE:
            pass
        
        # If there are less than 2 total people in the system ...
        # Simulate 5 seconds passing using arrival rate ...
        # Get total people arriving in that moment
        # Add people to simulation if SUM(ARRIVALS) + SUM(PEOPLE ALREADY) <= 2
        else:
            arrivals = []
            for _ in range(TIMESTEP):
                arrivals.append(int(np.random.choice([1, 0], 1, p=[ARRIVAL_RATE, 1 - ARRIVAL_RATE])))
            if sum(arrivals) + len(self.people) <= 2:
                for person in arrivals:
                    if person == 1:

                        worker = np.random.choice(['call-from-floor-2', 'call-from-floor-2-to-6'], 1, p=[.50, .50])
                        
                        # 50% call from floor 2 and leave with uniform dist
                        if worker == 'call-from-floor-2':
                            START_FLOORS = [2]
                            EXIT_FLOORS = [1,3,4,5,6]
                            START_PROB = [1]
                            EXIT_PROB = [.20, .20, .20, .20, .20]

                            call_floor = np.random.choice(START_FLOORS, 1, p=START_PROB)
                            exit_floor = np.random.choice(EXIT_FLOORS, 1, p=EXIT_PROB)

                        # 50% call from other floors and exit on 1
                        else:
                            START_FLOORS = [2,3,4,5,6]
                            EXIT_FLOORS = [1]
                            START_PROB = [.20, .20, .20, .20, .20]
                            EXIT_PROB = [1]
                            call_floor = np.random.choice(START_FLOORS, 1, p=START_PROB)
                            exit_floor = np.random.choice(EXIT_FLOORS, 1, p=EXIT_PROB)
                        
                        call_floor = int(np.random.choice(START_FLOORS, 1, p=START_PROB))
                        exit_floor = int(np.random.choice(EXIT_FLOORS, 1, p=EXIT_PROB))
                        self.people.append((call_floor, exit_floor, WAITING))
                        new_people.append((call_floor, exit_floor, WAITING))


        # Get a list of all people
        previous_people = list(p_floors)
        all_people = previous_people + new_people
        all_people_filtered = [tup for tup in all_people if tup != (0, 0, WAITING)]

        # If there is a passenger spot available replace the (0,0,0) with the new passenger
        while len(all_people_filtered) < 2:
            all_people_filtered.insert(0, (0, 0, WAITING))

        # If there are no passenger spots available, pop top of passenger list
        while len(all_people_filtered) > 2:
            all_people_filtered.pop()

        # Update timestep
        self.current_time += TIMESTEP

        return elevator_A_state, elevator_B_state, (all_people_filtered[0], all_people_filtered[1])

    def reward_func(self, state, action, new_state):

        # Parse state information
        elevator_A_state, elevator_B_state, passenger_info = state
        
        # Information about each elevator
        a_elev, a_floor, a_door  = elevator_A_state
        b_elev, b_floor, b_door  = elevator_B_state

        # Information on each action for each elevator
        action_A, action_B = action

        # Information on arriving passengers
        p1, p2 = passenger_info
        p1_call, p1_exit, p1_loc = p1
        p2_call, p2_exit, p2_loc = p2


        # Parse state information
        n_elevator_A_state, n_elevator_B_state, n_passenger_info = new_state
        
        # Information about each elevator
        n_a_elev, n_a_floor, n_a_door  = n_elevator_A_state
        n_b_elev, n_b_floor, n_b_door  = n_elevator_B_state

        # Information on arriving passengers
        n_p1, n_p2 = n_passenger_info
        n_p1_call, n_p1_exit, n_p1_loc = n_p1
        n_p2_call, n_p2_exit, n_p2_loc = n_p2

        # Calculate reward based on the action and the new state
        reward = 0

        if action_A[0] == 'UP' or action_A[0] == 'DOWN':

            # If youre in elevator B for p1 ...
            if n_p1_loc == IN_A:

                # If you moved closer to exit, reward
                if abs(n_a_floor - n_p1_exit) < abs(a_floor - p1_exit):
                    reward += MOVEMENT_REWARD
                else:  
                    reward -= MOVEMENT_PENALTY

            # If youre in elevator B for p2 ...
            if n_p2_loc == IN_A:

                # If you moved closer to exit, reward
                if abs(n_a_floor - n_p2_exit) < abs(a_floor - p2_exit):
                    reward += MOVEMENT_REWARD
                else:   
                    reward -= MOVEMENT_PENALTY

            # If in new state for p1 they are still waiting for elevator and moved closer, reward.
            if n_p1_loc == WAITING and n_p1_call != 0 and n_p1_exit != 0:

                if abs(n_a_floor - n_p1_call) < abs(a_floor - p1_call):
                    reward += MOVEMENT_REWARD
                else:  
                    reward -= MOVEMENT_PENALTY

            # If in new state for p2 they are still waiting for elevator and moved closer, reward.
            if n_p2_loc == WAITING and n_p2_call != 0 and n_p2_exit != 0:

                if abs(n_a_floor - n_p2_call) < abs(a_floor - p2_call):
                    reward += MOVEMENT_REWARD
                else:  
                    reward -= MOVEMENT_PENALTY
            
        if action_B[0] == 'UP' or action_B[0] == 'DOWN':
            
            # If in new state they are in elevator B for p1...
            if n_p1_loc == IN_B:

                # If you moved closer to exit, reward
                if abs(n_b_floor - n_p1_exit) < abs(b_floor - p1_exit):
                    reward += MOVEMENT_REWARD
                else:  
                    reward -= MOVEMENT_PENALTY

            # If in new state they are in elevator B for p2 ...
            if n_p2_loc == IN_B:

                # If you moved closer to exit, reward
                if abs(n_b_floor - n_p2_exit) < abs(b_floor - p2_exit):
                    reward += MOVEMENT_REWARD
                else:  
                    reward -= MOVEMENT_PENALTY

            # If in new state for p1 they are still waiting for elevator and moved closer, reward.
            if n_p1_loc == WAITING and n_p1_call != 0 and n_p1_exit != 0:
                if abs(n_b_floor - n_p1_call) < abs(b_floor - p1_call):
                    reward += MOVEMENT_REWARD
                else:  
                    reward -= MOVEMENT_PENALTY

            # If in new state for p2 they are still waiting for elevator and moved closer, reward.
            if n_p2_loc == WAITING and n_p2_call != 0 and n_p2_exit != 0:

                if abs(n_b_floor - n_p2_call) < abs(b_floor - p2_call):
                    reward += MOVEMENT_REWARD
                else:  
                    reward -= MOVEMENT_PENALTY

        if action_A[0] == 'DOORS':
            
            # If the previous state p1 or p2 was waiting and now they are in elevator A, give reward
            if p1_loc == WAITING and p1_call != 0 and p1_exit != 0 and n_p1_loc == IN_A:
                reward += PASSENGER_PICKUP_REWARD 

            if p2_loc == WAITING and p2_call != 0 and p2_exit != 0 and n_p2_loc == IN_A:
                reward += PASSENGER_PICKUP_REWARD
            
            # If p1 or p2 is still waiting after opening the door, give penalty
            if p1_loc == WAITING and p1_call != 0 and p1_exit != 0 and n_p1_loc == WAITING:
                reward -= DOOR_OPEN_PENALTY

            if p2_loc == WAITING and p2_call != 0 and p2_exit != 0 and n_p2_loc == WAITING:
                reward -= DOOR_OPEN_PENALTY

            # If was in elevtaor and now not in elevator, reward
            if p1_loc == IN_A and n_p1_loc == WAITING:
                reward += PASSENGER_DROP_OFF_REWARD

            if p2_loc == IN_A and n_p2_loc == WAITING:
                reward += PASSENGER_DROP_OFF_REWARD
            
        if action_B[0] == 'DOORS':

            # If the previous state p1 or p2 was waiting and now they are in elevator A, give reward
            if p1_loc == WAITING and p1_call != 0 and p1_exit != 0 and n_p1_loc == IN_B:
                reward += PASSENGER_PICKUP_REWARD 

            if p2_loc == WAITING and p2_call != 0 and p2_exit != 0 and n_p2_loc == IN_B:
                reward += PASSENGER_PICKUP_REWARD
            
            # If p1 or p2 is still waiting after opening the door, give penalty
            if p1_loc == WAITING and p1_call != 0 and p1_exit != 0 and n_p1_loc == WAITING:
                reward -= DOOR_OPEN_PENALTY

            if p2_loc == WAITING and p2_call != 0 and p2_exit != 0 and n_p2_loc == WAITING:
                reward -= DOOR_OPEN_PENALTY

            # If was in elevtaor and now not in elevator, reward
            if p1_loc == IN_B and n_p1_loc == WAITING:
                reward += PASSENGER_DROP_OFF_REWARD

            if p2_loc == IN_B and n_p2_loc == WAITING:
                reward += PASSENGER_DROP_OFF_REWARD
        
        if action_A[0] == 'HOLD':
            
            # If the previous state p1 or p2 was waiting and now they are in elevator A, give reward
            if p1_loc == WAITING and n_p1_loc == IN_A:
                reward += PASSENGER_PICKUP_REWARD

            if p2_loc == WAITING and n_p2_loc == IN_A:
                reward += PASSENGER_PICKUP_REWARD
            
            # If p1 or p2 is still waiting after opening the door, give penalty
            if p1_loc == WAITING and p1_call != 0 and p1_exit != 0 and n_p1_loc == WAITING:
                reward -= DOOR_HOLD_PENALTY

            if p2_loc == WAITING and p2_call != 0 and p2_exit != 0 and n_p2_loc == WAITING:
                reward -= DOOR_HOLD_PENALTY

            # If was in elevtaor and now not in elevator, reward
            if p1_loc == IN_A and n_p1_loc == WAITING:
                reward += PASSENGER_DROP_OFF_REWARD

            if p2_loc == IN_A and n_p2_loc == WAITING:
                reward += PASSENGER_DROP_OFF_REWARD
            
        if action_B[0] == 'HOLD':

            # If the previous state p1 or p2 was waiting and now they are in elevator A, give reward
            if p1_loc == WAITING and p1_call != 0 and p1_exit != 0 and n_p1_loc == IN_B:
                reward += PASSENGER_PICKUP_REWARD

            if p2_loc == WAITING and p2_call != 0 and p2_exit != 0 and n_p2_loc == IN_B:
                reward += PASSENGER_PICKUP_REWARD
            
            # If p1 or p2 is still waiting after opening the door, give penalty
            if p1_loc == WAITING and p1_call != 0 and p1_exit != 0 and n_p1_loc == WAITING:
                reward -= DOOR_HOLD_PENALTY

            if p2_loc == WAITING and p2_call != 0 and p2_exit != 0 and n_p2_loc == WAITING:
                reward -= DOOR_HOLD_PENALTY

            # If was in elevtaor and now not in elevator, reward
            if p1_loc == IN_B and n_p1_loc == WAITING:
                reward += PASSENGER_DROP_OFF_REWARD

            if p2_loc == IN_B and n_p2_loc == WAITING:
                reward += PASSENGER_DROP_OFF_REWARD

        return reward
