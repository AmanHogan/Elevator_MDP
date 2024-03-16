# Learning Elevator Control 

## Summary/Objective (CSE 6369 - Reinforcement Learning)
Consider again the building with 6 floors and 2 elevators, but with 2 identical elevators. Each of the elevators takes 5s to travel between floors, 5s to open the doors, 5s to close them, and we assume that the time the doors are open is sufficient for a person to enter the elevator and press the floor button. In addition to moving up or down a floor, the elevator can also hold its position/state for 5s. This means that we can model each elevator with 4 actions: up, down, hold, doors (where the latter toggles the doors (i.e. opens the doors if they are closed or closes them when they are open) and that they make decisions every 5s (and not in between). We assume that a person will immediately enter if the doors are open on the floor they are waiting on (no matter the elevator is going up or down) and will exit only (and as soon as) the elevator doors are open on their target floor). To limit the state space complexity we limit the number of users in the system at any point of time to at most 2 (i.e. if there are already 2 persons in the system - either waiting for an elevator or inside the elevator - no additional persons will arrive). Besides this limitation we will again assume that at any given second, a person arrives at a ”call floor”, fc, with a fixed probability distribution, P (fc), and wants to go to ”exit floors”, fe, with a particular distribution, P (fe|fc). This means that 2 persons will never press the button at the exact same time but it is possible that by the time the elevators make decisions (once every 5s), two new persons have arrived.

## Prerequisites
- A Windows 10 or Mac OS computer
- Internet access and administrator access
- Vscode
- `Python 2.7xx` or higher
- `matplotlib` and `numpy`

# How to Run / Usage
- Clone or download this repository
- Move the cloned or downloaded repo to a safe location like your Desktop
- Open the project in a code editor like Vscode
- Open a terminal and make sure your terminal looks similar to mine: `C:\Users\Aman\<Dowload loaction>\project_1>`
- Now you need to do `cd ..` so your terminal looks like this `C:\Users\Aman\<Dowload loaction>>` where you are one directory above the project is located.
- Now you can run any of the \<problem-algorithm\> combination by typing in the console: `python -m project_1.2<i or ii or iii>.agent_<algorithm>`

Below is a list of all the commands to run each probelm and algorithm:

```
python -m project_1.2i.agent_q
python -m project_1.2i.agent_sarsa
python -m project_1.2i.agent_q_lambda
python -m project_1.2i.agent_sarsa_lambda

python -m project_1.2ii.agent_q
python -m project_1.2ii.agent_sarsa
python -m project_1.2ii.agent_q_lambda
python -m project_1.2ii.agent_sarsa_lambda

python -m project_1.2iii.agent_q
python -m project_1.2iii.agent_sarsa
python -m project_1.2iii.agent_q_lambda
python -m project_1.2iii.agent_sarsa_lambda
```
In the [globals.py](./ENVIRONMENT/globals.py) file you can change specific parameters of the problem. There should be an accurate description each variable in the file.

For the lambda algorithms, I recommed changing the number of iterations or uncommenting the code in the 
[globals.py](./ENVIRONMENT/globals.py#ls97) to reduce the state space. 

I each algorithm file, you can change exactly what learning rates, discount factors, and epsilon values you want to graph. You can change these values, as long as you have at least one value in each varaible array Here is what it is set by [default](./2i/agent_q.py#ls97):

```
# agent_q
...
alphas = [.01,.1,.3,.5,.9] # learning rates
gammas = [.01,.1,.3,.5,.9] # discounted sums
epsilons = [.01,.1,.3,.5,.9] # exploration rates
agents = []
...
```

After a succesfull run, your output should show up in the [./output/](./output/) folder.

Solutions for each problem are in the ipyb files:
- [Problem 1](./problem_1.ipynb)
- [Problem 2](./problem_2.ipynb) 
- [Problem 3](./problem_3.ipynb) 

# How to read output and Understand Simulation
To understand how the simulation works, there are a few things you need to know about the output:
- The numbers below represent the floors

```
...
3 -
2 -
1 - 
```
- `[A] Open` means that elevator A is open
- `[B] Closed` means that elevator B is closed
- An elevator can not move up unless it is closed
- `[Wait P1 = (1, 4)]` means there is a person that is calling from floor 1 and wants to exit on floor floor and they are not inside an elevator
- `[Wait P1 = (1, 4)][Wait P2 = (1, 6)]` means there are two people waiting.
- `[B[P1_B] = (1, 4)] Closed` means there is a person inside elevator B and the door is closed.
- `[B[P1_B] = (1, 4)] Open` means there is a person inside elevator B and the door is open
- `TIME = 7135 SECONDS` denotes 7135 seconds have passed in the simulation
- `OLD STATE = (('A', 1, False), ('B', 1, False), ((0, 0, 0), (0, 0, 0)))` Means Both elevators are on floor 1 and doors are closed
- `(... , ((0, 0, 0), (0, 0, 0)))` means that a person has not arrived yet



## Authors
- Aman Hogan-Bailey


## Contributions and Referenes
- The University of Texas at Arlington
- Manfred Huber (2242-CSE-6369-001)
