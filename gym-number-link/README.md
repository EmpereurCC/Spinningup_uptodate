# NumberLink

**TODOS**:
- [ ] Add the board generation explanation (with comments on the diversity of the wires lengths) - Val
- [ ] incorporate examples of board representations - Alex
----------------------

The NumberLink environment is a game closely related to the PCB problem, while allowing us more flexibility on the problem definition. Here is a small summary of the key components of the environment:

## Board and Wire Entities:
A board has a rectangular shape and all the wires must live within the boundaries of the board. A wire is composed of two pins, the agent has to connect them to each other.

## Action Space:
- For each wire, there are four possible move directions: North, East, South, West. As opposed to the pcb-go environment, a path can materialize from both pins, and look for their counterpart on the board. Hence the number of actions per wire is two times the four move directions. 
- As opposed to the pcb-go environment, a wire is always allowed to come back on his track. In addition, when a wire loops back to a previously visited position, the loop is discarded and the wire is back in his previous state.
- Similarly, one pin of a wire can connect to any point in the path of the second pin. If this occurs, the path not needed to create the full wire path is discarded.

## Observation Space:
The observation space is of shape: board_size x board_size x 2.
- The first channel is a matrix containing the indices of the wires (1 … N) on their respective paths.
- The second channel contains the heads of the wire pins (two per wire except if the wire has been connected, then both pins lies in the same cell of the matrix)

## Reward Signal
A reward is received each time a wire connects, defined as 1/num_wires. Hence, if all the wires are connected at the end of the game, the return, i.e. the sum of the intermediate reward without discounting factor, sums to one.