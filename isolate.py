from math import sqrt
from collections import namedtuple
from itertools import cycle
from homework1.maze import SequenceMaze, Node
from homework1.mazes import mazes_to_test
from random import choice



# print(one_right, one_left)

from homework1.priority_queue import PriorityQueue

node_to_tuple = lambda node: tuple([node.state['index'], node.count_star_as if node.count_star_as is not None else node.state['char_val'], node.parent])


depth = 4


# def _to_coordinate(state_index):
#     """take index of 1d list and convert it to a coordinate"""
#     return int(state_index / 6), state_index % depth
#
#
# def manhatten_distance(state_indexA, state_indexB):
#     x1, y1  = _to_coordinate(state_indexA)
#     x2, y2 = _to_coordinate(state_indexB)
#     return abs(x1 - x2) + abs(y1 - y2)
#
# def euclidean_distance(state_indexA, state_indexB):
#     x1, y1  = _to_coordinate(state_indexA)
#     x2, y2 = _to_coordinate(state_indexB)
#     return sqrt( pow((x1 - x2), 2) + pow((y1 - y2), 2) )
#
# print('md', manhatten_distance(0,6))
#
# print('ed', euclidean_distance(0,6))


input = (6, [
             'a', '*', 'a', 'c', 'b', 'c',
             'c', '*', '*', '*', '*', 'b',
             '*', 'b', 'b', 'b', 'c', '*',
             '*', 'b', 'b', 'a', 'b', 'c',
             'c', '*', 'b', '*', '*', '*',
             'a', '*', '*', 'c', 'a', 'c'
])

dim_val = 6
print(choice(list(range(dim_val*dim_val))))

