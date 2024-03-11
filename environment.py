import numpy as np
import time
from globals import * 
WAITING = 0
IN_A = 1
IN_B = 2

class Environment:

    def __init__(self, state):

        self.curr_state = state
        self.elevatorA = Elevators("A", state[0][1])
        self.elevatorB = Elevators("B", state[1][1])

        self.new_state = None
        self.current_time = 0
        self.people = [] # a list of people who are wating on an elevator
        self.MAX_PEOPLE = 2
    
    def get_next_state(self, state, action):
        """
        Proccesses the action made by the agent and reflect it on the environment. It ensures that
        no invalid actions are reflected into the environment. Returns the  next state

        Args:
            state (state): current state of the world
            action (action): agent chosen action

        Returns:
            next_state: next state given the actions
        """

        # Get elevator states and passengers floors
        # ex: (('A', 1, False), ('B', 1, False), ((0,0,0), (0,0,0)))
        elevator_A_state, elevator_B_state, passenger_info = state
        
        a_elev, a_floor, a_door  = elevator_A_state
        b_elev, b_floor, b_door  = elevator_B_state

        action_A, action_B = action

        p1, p2 = passenger_info
        p1_call, p1_exit, p1_loc = p1
        p2_call, p2_exit, p2_loc = p2


        ################# Check the action for Elevator A ##############################################

        if action_A[0] == 'UP' and a_floor < NFLOORS and a_door == False:
            elevator_A_state = ('A', a_floor + 1, a_door)

        elif action_A[0] == 'DOWN' and a_floor > 1 and a_door == False:
            elevator_A_state = ('A', a_floor - 1, a_door)
            
        elif action_A[0] == 'HOLD':
            pass

        # Check the action for Elevator A
        elif action_A[0] == 'DOORS':

            ####################################################################################
            # NOTE: People can only leave and enter the elevator as it is open

            # Door open and going to close and person waiting and call floor same as elevator floor
            if a_door == OPEN and p1_loc == WAITING and p1_call == a_floor:
                 # adds passenger
                 p1_loc = IN_A
            
            # Door open and person in A  and elve same as exit floor
            # Remove person from elevator and premove from list of people
            if a_door == OPEN and p1_loc == IN_A and p1_exit == a_floor:
                self.people.remove((p1_call, p1_exit, WAITING))
                p1_call, p1_exit, p1_loc = 0, 0, WAITING

            # Door open and person waiting and call floor same as elevator floor
            if a_door == OPEN and p2_loc == WAITING and p2_call == a_floor:
                 p2_loc = IN_A
            
            # Door open and person in A  and elve same as exit floor
            # Remove person from elevator and premove from list of people
            if a_door == OPEN and p2_loc == IN_A and p2_exit == a_floor:
                self.people.remove((p2_call, p2_exit, WAITING))
                p2_call, p2_exit, p2_loc = 0, 0, WAITING

            # If door is closed, yopu cant do antyhontg

            elevator_A_state = ('A', a_floor, not a_door)
        
        ################# Check the action for Elevator B ##############################################

        if action_B[0] == 'UP' and b_floor < NFLOORS and b_door == False:
            elevator_B_state = ('B', b_floor + 1, b_door)

        elif action_B[0] == 'DOWN' and elevator_B_state[1] > 1 and b_door == False:
            elevator_B_state = ('B', b_floor - 1, b_door)

        elif action_B[0] == 'HOLD':
            pass

        # Check the action for Elevator B
        elif action_B[0] == 'DOORS':
            
            ####################################################################################
            # NOTE: People can only leave and enter the elevator as it is open

            # Door open and going to close and person waiting and call floor same as elevator floor
            if b_door == OPEN and p1_loc == WAITING and p1_call == b_door:
                 # adds passenger
                 p1_loc = IN_B

            # Door open and person in A  and elve same as exit floor
            # Remove person from elevator and premove from list of people
            if b_door == OPEN and p1_loc == IN_B and p1_exit == b_door:
                self.people.remove((p1_call, p1_exit, WAITING))
                p1_call, p1_exit, p1_loc = 0, 0, WAITING

            ############################################################################
            # Door open and person waiting and call floor same as elevator floor
            if b_door == OPEN and p2_loc == WAITING and p2_call == b_door:
                 p2_loc = IN_A

            # Door open and person in A  and elve same as exit floor
            # Remove person from elevator and premove from list of people
            if b_door == OPEN and p2_loc == IN_B and p2_exit == b_door:
                self.people.remove((p2_call, p2_exit, WAITING))
                p2_call, p2_exit, p2_loc = 0, 0, WAITING

            #######################################################################
            
            # If door is closed, yopu cant do antyhontg
          

            
            elevator_B_state = ('B', b_floor, not b_door)

            #################################################################################################
        
        #################################################################################################
        
        p1 = p1_call, p1_exit, p1_loc
        p2 = p2_call, p2_exit, p2_loc
        passenger_info = p1, p2
        new_state = (elevator_A_state, elevator_B_state, passenger_info)

        return new_state

    def simulate_action(self, state, action):

        # Adds people to system given not at capacity
        next_state = self.simulate_people(state)

        # Gets next state based on actions and current state
        next_state = self.get_next_state(next_state, action)

        next_reward = self.reward_func(self.new_state, action)



        return next_state, next_reward
    # (('A', 2, True), ('B', 1, True), ((1, 3, 2), (1, 6, 2)))
    def print_info(self, state):

        elevator_A_state, elevator_B_state, passenger_info = state
        
        a_elev, a_floor, a_door  = elevator_A_state
        b_elev, b_floor, b_door  = elevator_B_state


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

            if p2_loc == WAITING and floor == p1_call:
                print(f"[Wait P2 = {p2_call, p2_exit}]", end="")
            
            print()
        print("-----------------------------------------------------------------")
        

    def simulate_people(self, state):
        """
        Generates new people with Probability distribution of call and exiting elevator.
        Adds these people to the the people list in envirnoment. Max of 2 people in system
        """
        elevator_A_state, elevator_B_state, p_floors = state
        people = []

        # If there are 2 or more people waiting skip simulation
        if len(self.people) >= self.MAX_PEOPLE:
            pass
        
        # If there are less than 2 total people in the system..
        else:
            
            # Simulate 5 seconds using arrival rate, (1 denotes a person arrived in length 5 array)
            arrivals = []
            for _ in range(TIMESTEP):
                arrivals.append(int(np.random.choice([1, 0], 1, p=[ARRIVAL_RATE, 1 - ARRIVAL_RATE])))

            # If total people in elevator + total arrived less than 2, simulate their call and exit chance.
            if sum(arrivals) + len(self.people) <= 2:
                for person in arrivals:
                    if person == 1:
                        call_floor = int(np.random.choice(START_FLOORS, 1, p=START_PROB))
                        exit_floor = int(np.random.choice(EXIT_FLOORS, 1, p=EXIT_PROB))
                        self.people.append((call_floor, exit_floor, WAITING))
                        people.append((call_floor, exit_floor, WAITING))
            else:
                pass

        
        old_people = list(p_floors)
        both_people = old_people + people

        # print("Old people:", old_people)
        # print("New people:", people)
        # print("People:", self.people)
        # print("Combined:", both_people)

        # Remove (0, 0) tuples from the combined list
        n = [tup for tup in both_people if tup != (0, 0, WAITING)]

        # If there are fewer than 2 tuples, insert (0, 0) tuples at the beginning until there are 2 tuples
        while len(n) < 2:
            n.insert(0, (0, 0, WAITING))

        # If there are more than 2 tuples, remove the last two
        while len(n) > 2:
            n.pop()

        #print("Filtered:", n)

        #print(tuple([(person.call_floor, person.exit_floor) for person in self.people]))
        # Update timestep
        self.current_time += TIMESTEP

        
        return elevator_A_state, elevator_B_state, (n[0], n[1])

    def reward_func(self, new_state, action):
        return 0


class Elevators:

    def __init__(self, name, start):
        self.name = name
        self.passengers = []
        self.floor = start
        self.timeTaken = 0
        self.capacity = 0
        self.doorCount = 0
        self.isDoorOpen = False
        self.exit_floor = []


class Person:

    def __init__(self, call_floor, exit_floor):
        self.call_floor = call_floor
        self.exit_floor = exit_floor
        self.floor = call_floor
        self.isInElevator = False
