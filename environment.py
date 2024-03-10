import numpy as np
import time
from globals import (ACTION_SET, ACTION_SPACE, FLOORS)


NFLOORS = 6

ARRIVAL_RATE = .1
START_FLOORS = [1]
START_PROB = [1]

EXIT_FLOORS = [2,3,4,5,6]
EXIT_PROB = [.20, .20, .20, .20, .20]
TIMESTEP = 5


class Environment:

    def __init__(self, state):

        self.curr_state = state
        elevator_A_state, elevator_B_state = self.curr_state
        self.elevatorA = Elevators("A", elevator_A_state[1])
        self.elevatorB = Elevators("B", elevator_B_state[1])
        self.new_state = None
        self.current_time = 0
        self.people = []
        self.passengers = []
    
    def get_next_state(self, state, action):

        elevator_A_state, elevator_B_state = state
        action_A, action_B = action

        # Check the action for Elevator A 
        if action_A[0] == 'UP' and elevator_A_state[1] < NFLOORS and self.elevatorA.isDoorOpen == False:
            elevator_A_state = ('A', elevator_A_state[1] + 1)

        elif action_A[0] == 'DOWN' and elevator_A_state[1] > 1 and self.elevatorA.isDoorOpen == False:
            elevator_A_state = ('A', elevator_A_state[1] - 1)
            
        elif action_A[0] == 'HOLD':
            pass

        elif action_A[0] == 'DOORS':
            self.elevatorA.toggle_door()  # Toggle the door state

            # If the door is open, check for passengers boarding or disembarking
            if self.elevatorA.isDoorOpen:
                for passenger in list(self.passengers):  # Use list() to avoid modifying the list while iterating
                    if passenger.call_floor == elevator_A_state[1]:
                        self.elevatorA.add_passenger(passenger)
                        self.passengers.remove(passenger)  # Remove the passenger from the global list
                    elif passenger.exit_floor == elevator_A_state[1]:
                        self.elevatorA.remove_passenger(passenger)

        # Check the action for Elevator B
        if action_B[0] == 'UP' and elevator_B_state[1] < NFLOORS and self.elevatorB.isDoorOpen == False:
            elevator_B_state = ('B', elevator_B_state[1] + 1)

        elif action_B[0] == 'DOWN' and elevator_B_state[1] > 1 and self.elevatorB.isDoorOpen == False:
            elevator_B_state = ('B', elevator_B_state[1] - 1)

        elif action_B[0] == 'HOLD':
            pass

        elif action_B[0] == 'DOORS':
            self.elevatorB.toggle_door()  # Toggle the door state
            if self.elevatorB.isDoorOpen:
                for passenger in list(self.passengers):
                    if passenger.call_floor == elevator_B_state[1]:
                        self.elevatorB.add_passenger(passenger)
                        self.passengers.remove(passenger)
                    elif passenger.exit_floor == elevator_B_state[1]:
                        self.elevatorB.remove_passenger(passenger)

        new_state = (elevator_A_state, elevator_B_state)

        self.elevatorA.floor = elevator_A_state[1]
        self.elevatorB.floor = elevator_B_state[1]
        self.new_state = new_state

        return self.new_state

    def simulate_action(self, state, action):

        self.simulate_passengers()

        next_state = self.get_next_state(state,action)

        next_reward = self.reward_func(self.new_state, action)

        return next_state, next_reward
    
    def print_info(self):
        print("TIME =", self.current_time)
        for floor in range(len(FLOORS),0, -1):
            print(floor,"-", end="")
            if self.elevatorA.floor == floor:
                print(f"| A (P = {[(passenger.call_floor, passenger.exit_floor) for passenger in self.elevatorA.passengers]}) |", end="")
                
            if self.elevatorB.floor == floor:
                print(f"| B (P = {[(passenger.call_floor, passenger.exit_floor) for passenger in self.elevatorA.passengers]}) |", end="")

            for passenger in self.passengers:
                 if passenger.call_floor == floor:
                     print(f"|P({ passenger.call_floor}, {passenger.exit_floor})|", end="")
            
            print()
        print("-----------------------------------------------------------------")

    def simulate_passengers(self):
        total_elev_passengers = sum(len(elevator.passengers) for elevator in [self.elevatorA, self.elevatorB])

        if total_elev_passengers >= 2:
            pass

        total_waiting_passengers = len(self.passengers)

        if total_waiting_passengers>= 2:
            pass
        else:
            possible_passengers = []
            for _ in range(TIMESTEP):
                possible_passengers.append(int(np.random.choice([1, 0], 1, p=[ARRIVAL_RATE, 1 - ARRIVAL_RATE])))

            if sum(possible_passengers) == 0:
                pass

            if sum(possible_passengers) + total_elev_passengers <= 2:
                for passenger in possible_passengers:
                    if passenger == 1:
                        call_floor = int(np.random.choice(START_FLOORS, 1, p=START_PROB))
                        exit_floor = int(np.random.choice(EXIT_FLOORS, 1, p=EXIT_PROB))
                        self.passengers.append(Passenger(call_floor, exit_floor))
            else:
                pass

        self.current_time += TIMESTEP

    def reward_func(self, new_state, action):
        elevatorA_time = 0
        elevatorB_time = 0

        for passenger in self.passengers:
            elevatorA_time += 5 * abs(passenger.call_floor - passenger.exit_floor) + (2 * 5) + 5 * abs(self.elevatorA.floor - passenger.call_floor)
            elevatorB_time += 5 * abs(passenger.call_floor - passenger.exit_floor) + (2 * 5) + 5 * abs(self.elevatorB.floor - passenger.call_floor)

        return -elevatorA_time + -elevatorB_time


class Elevators:

    def __init__(self, name, start):
        self.name = name
        self.passengers = []
        self.floor = start
        self.timeTaken = 0
        self.capacity = 0
        self.doorCount = 0
        self.isDoorOpen = False

    def add_passenger(self, passenger):
        self.passengers.append(passenger)
        passenger.floor = self.floor

    def remove_passenger(self, passenger):
        if passenger in self.passengers:
            self.passengers.remove(passenger)

    def toggle_door(self):
        self.isDoorOpen = not self.isDoorOpen

class Passenger:

    def __init__(self, call_floor, exit_floor):
        self.call_floor = call_floor
        self.exit_floor = exit_floor
        self.floor = call_floor
        self.isInElevator = False
