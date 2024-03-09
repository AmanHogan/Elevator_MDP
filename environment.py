import numpy as np
import time

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
    
    def execute_action(self, state, action):

        self.simulate()

        self.curr_state = state
        elevator_A_state, elevator_B_state = self.curr_state
        action_A, action_B = action

        # Check the action for Elevator A
        if action_A[0] == 'UP' and elevator_A_state[1] < NFLOORS:
            elevator_A_state = ('A', elevator_A_state[1] + 1)

        elif action_A[0] == 'DOWN' and elevator_A_state[1] > 1:
            elevator_A_state = ('A', elevator_A_state[1] - 1)

        elif action_A[0] == 'HOLD':
            pass  # No change for holding action

        elif action_A[0] == 'DOORS':
            self.elevatorA.isElevatorOpen = not self.elevatorA.isElevatorOpen

        # Check the action for Elevator B
        if action_B[0][0] == 'UP' and elevator_B_state[1] < NFLOORS:
            elevator_B_state = ('B', elevator_B_state[1] + 1)

        elif action_B[0][0] == 'DOWN' and elevator_B_state[1] > 1:
            elevator_B_state = ('B', elevator_B_state[1] - 1)

        elif action_B[0][0] == 'HOLD':
            pass  # No change for holding action

        elif action_B[0][0] == 'DOORS':
            self.elevatorB.isElevatorOpen = not self.elevatorB.isElevatorOpen

        # Update the current state with the new state
        self.curr_state = (elevator_A_state, elevator_B_state)
        self.new_state = self.curr_state
        return self.new_state

    def print_pic():
        pass

    def update_elevators():
        pass
    
    def update_passengers():
        pass


    def print_info(self):
            print("Current time =", self.current_time)
            print("Current state:", self.curr_state, "A and B Doors Opend?", self.elevatorA.isElevatorOpen, self.elevatorB.isElevatorOpen)
            for passenger in self.passengers:
                print("Passenger: Call Floor =", passenger.call_floor, ", Exit Floor =", passenger.exit_floor)
            print("-----------------------------------------------------------------")

    def simulate(self):
        
        if len(self.passengers) >= 2:
            pass
        
        else:
            possible_passengers = []
            
            for _ in range(TIMESTEP):
                possible_passengers.append(int(np.random.choice([1,0], 1, p=[ARRIVAL_RATE, 1- ARRIVAL_RATE])))
            
            if sum(possible_passengers) == 0:
                pass

            if sum(possible_passengers) + len(self.passengers) <= 2:
                for passenger in possible_passengers:
                    if passenger == 1:
                        call_floor = int(np.random.choice(START_FLOORS, 1, p=START_PROB))
                        exit_floor = int(np.random.choice(EXIT_FLOORS, 1, p=EXIT_PROB))
                        self.passengers.append(Passenger(call_floor, exit_floor))
    
            else:
                pass


        self.current_time += TIMESTEP

class Elevators:

    def __init__(self, name, start):
        self.name = name
        self.passnegers = []
        self.current_floor = start
        self.destination = None
        self.timeTaken = 0
        self.isElevatorOpen = False

    def add_passenger(self):
        pass

    def remove_passenger(self):
        pass

class Passenger:

    def __init__(self, call_floor, exit_floor):
        self.call_floor = call_floor
        self.exit_floor = exit_floor
        self.passengerA = False
        self.passngerB = False

    def getInElevator():
        pass

    def leaveElevator():
        pass