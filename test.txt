# Check the action for Elevator A
        elif action_A[0] == 'DOORS':

            # If the doors are open
            if a_door == OPEN:

                # If person 1 is not in elevator and door is open
                if p1_loc == WAITING:
                    
                    # If person 1 is on the same floor while elevator is open and not in the elevator, put them in the elevator
                    if p1_call == a_floor:
                        p1_loc = IN_A
                else:
                    # If person 1 is in the elevator and on exit floor, debark them
                    if p1_exit == a_floor:
                        self.people.remove((p1_call, p1_exit, WAITING))
                        p1_call, p1_exit, p1_loc = 0, 0, WAITING

                # If person 1 is not in elevator and door is open
                if p2_loc == WAITING:
                    # If person 1 is on the same floor while elevator is open and not in the elevator, put them in the elevator
                    if p2_call == a_floor:
                        p2_loc = IN_A
                else:
                    # If person 1 is in the elevator and on exit floor, debark them
                    if p2_exit == a_floor:
                        self.people.remove((p2_call, p2_exit, WAITING))
                        p2_call, p2_exit, p2_loc = 0, 0, WAITING


                # Update elevator A state to reflect closed doors
                elevator_A_state = ('A', a_floor, CLOSED)

            # If the doors are closed
            else:
                # If people are on the same floor while elevator is closed and aren't in the elevator, move them to the elevator
                if p1_call == a_floor and p1_loc == WAITING:
                    p1_loc = IN_A

                if p2_call == a_floor and p2_loc == WAITING:
                    p2_loc = IN_A

                # Update elevator A state to reflect open doors
                elevator_A_state = ('A', a_floor, OPEN)
