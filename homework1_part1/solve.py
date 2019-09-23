"""
a driver to test search algorithms
"""
import time
from homework1_part1.mazes.mazes_to_test import mazeA, mazeB, mazeC, mazeD
from homework1_part1.uninformed_search import SEARCH_ALGORITHMS, breadth_first_search
from homework1_part1.maze import SequenceMaze

# graphs to pass as input in tuple format


if __name__ == '__main__':
    # create maze object here

    # iter through each search algo to compare paths and times returned
    print('#################################################################################')
    for algo in SEARCH_ALGORITHMS:
        print(f'[PROCESSING ALGORITHM] {algo.__name__}')
        start = time.time()
        seq_maze = SequenceMaze(mazeA)
        length_of_sol, num_states, sol_locations = algo(seq_maze)
        end = time.time()
        print('timed @', end - start)
        print(f'Length of solution {length_of_sol} \nTotal Numer of States Visited {num_states} \n'
              f'Solutions Locations {sol_locations}  ')
        print('#################################################################################')


# a place to isolate algos if needed
#     maze_problem = SequenceMaze(mazeA)
#     length_of_sol, num_states, sol_locations = breadth_first_search(maze_problem)
#     print(f'Length of solution {length_of_sol} \nTotal Numer of States Visisted {num_states} \n'
#               f'Solutions Locations {sol_locations}  ')
#

