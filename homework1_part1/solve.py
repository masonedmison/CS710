"""
a driver to test search algorithms
"""
import time
from homework1_part1.mazes.mazes_to_test import mazes_to_test
from homework1_part1.uninformed_search import SEARCH_ALGORITHMS, breadth_first_search
from homework1_part1.maze import SequenceMaze

# graphs to pass as input in tuple format


if __name__ == '__main__':
    # create maze object here
    # iter through each search algo to compare paths and times returned
    res_str = """"""
    res_str +='\n#########################################################################################'
    for algo in SEARCH_ALGORITHMS:
        res_str += f'\n[PROCESSING ALGORITHM] {algo.__name__}'
        res_str +='\n#################################################################################'
        for k in mazes_to_test.keys():
            res_str+=f'\n\t[SOLVING {k}]'
            start = time.time()
            seq_maze = SequenceMaze(mazes_to_test[k])
            length_of_sol, num_states, sol_locations = algo(seq_maze)
            end = time.time()
            res_str += f'\n\t\ttimed @ {end-start}'
            res_str+= f'\n\t\tLength of solution {length_of_sol} \n\t\tTotal Numer of States Visited {num_states} \n\t\tSolutions Locations {sol_locations} '
            res_str += '\n#########################################################################################'
    with open('seq_maze_res.txt', 'w') as out:
        out.write(res_str)

# a place to isolate algos if needed
#     mazeA = mazes_to_test['mazeA']
#     print(mazeA)
#     maze_problem = SequenceMaze(mazeA)
#     length_of_sol, num_states, sol_locations = breadth_first_search(maze_problem)
#     print(f'Length of solution {length_of_sol} \nTotal Numer of States Visisted {num_states} \n'
#               f'Solutions Locations {sol_locations}  ')


