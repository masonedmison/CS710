"""
a driver to test search algorithms
"""
import time
import pandas as pd
import plotly.graph_objects as go
from homework1.mazes.mazes_to_test import mazes_to_test
from homework1.uninformed_search import UNINFORMED_SEARCH_ALGORITHMS, breadth_first_search
from homework1.informed_search import INFORMED_SEARCH_ALGORITHMS
from homework1.informed_search import astar_search, greedy_search, euclidean_distance, manhatten_distance, hill_climbing
from homework1.maze import SequenceMaze

# graphs to pass as input in tuple format

def create_table():
    res_df = pd.DataFrame({'Algorithm':vals[0] ,'Maze Dimension':vals[1], 'Solution Length':vals[2], 'Num States Visited':vals[3],
                           'Path Cost':vals[4], 'Time to execute':vals[5]})
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(res_df.columns),
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[res_df['Algorithm'],res_df['Maze Dimension'], res_df['Solution Length'], res_df['Num States Visited'],
                           res_df['Path Cost'], res_df['Time to execute']],
                   fill_color='lavender',
                   align='left'))
    ])
    fig.show()

if __name__ == '__main__':
    # iter through each search algo to compare paths and times returned
    algos = list()
    vals = list([ [],[],[],[], [], [] ]) # algo,maze length, sol_length, num_states visited, time to execute
    res_str = """"""
    res_str +='\n#########################################################################################'
    h = manhatten_distance
    for algo in INFORMED_SEARCH_ALGORITHMS:
        res_str += f'\n[PROCESSING ALGORITHM] {algo.__name__}'
        res_str +='\n#################################################################################'
        for k in mazes_to_test.keys():
            res_str+=f'\n\t[SOLVING {k}]'
            start = time.time()
            seq_maze = SequenceMaze(mazes_to_test[k])
            length_of_sol, num_states, sol_locations, path_cost = algo(seq_maze, h)
            vals[0].append(f'{algo.__name__} with {k}')
            vals[1].append(mazes_to_test[k][0])
            vals[2].append(length_of_sol)
            vals[3].append(num_states)
            vals[4].append(path_cost)
            end = time.time()
            time_to_ex = end-start
            vals[5].append(time_to_ex)
            res_str += f'\n\t\ttimed @ {time_to_ex}'
            res_str+= f'\n\t\tLength of solution {length_of_sol} \n\t\tTotal Numer of States Visited {num_states} \n\t\tSolutions Locations {sol_locations} '
            res_str+=f'\n\t\t Path Cost {path_cost}'
            res_str += '\n#########################################################################################'
    with open('seq_maze_res.txt', 'w') as out:
        out.write(res_str)


    # create table
    create_table()

# a place to isolate algos if needed
#     maze= mazes_to_test['mazeA']
#     print(maze)
#     maze_problem = SequenceMaze(maze)
#     length_of_sol, num_states, sol_locations, p_cost = hill_climbing(maze_problem, h=euclidean_distance)
#     print(f'Length of solution {length_of_sol} \nTotal Numer of States Visisted {num_states} \n'
#               f'Solutions Locations {sol_locations}\nPath Cost {p_cost}')
#     for item in sol_locations:
#         print('\n', item)


