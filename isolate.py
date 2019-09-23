from math import sqrt
from collections import namedtuple
from itertools import cycle
from homework1_part1.maze import SequenceMaze, Node
from homework1_part1.mazes import mazes_to_test


input = (6, [
             'a', '*', 'a', 'c', 'b', 'c',
             'c', '*', '*', '*', '*', 'b',
             '*', 'b', 'b', 'b', 'c', '*',
             '*', 'b', 'b', 'a', 'b', 'c',
             'c', '*', 'b', '*', '*', '*',
             'a', '*', '*', 'c', 'a', 'c'
])

# print(one_right, one_left)

from homework1_part1.priority_queue import PriorityQueue

state2 = 'c'
if state2 == 'a' or 'b' or 'c':
    print('test')
