"""
a driver to test search algorithms
"""
import time
from homework1_part1.mazes.mazes_to_test import mazeA
from homework1_part1.uninformed_search import SEARCH_ALGORITHMS

# graphs to pass as input in tuple format


if __name__ == '__main__':
    # create maze object here

    # iter through each search algo to compare paths and times returned
    print('#################################################################################')
    for algo in SEARCH_ALGORITHMS:
        start = time.time()
        length_of_sol, num_states, sol_locations = algo(None)
        end = time.time()
        print(algo.__name__, end - start)
        print(f'Length of solution {length_of_sol} \nTotal Numer of States Visisted {num_states} \n'
              f'Solutions Locations {sol_locations}  ')
        print('#################################################################################')




