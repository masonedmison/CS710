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

# {index: <array_index>, char_val:<char_val>}
n1 = Node(dict(index=0, char_val='second'), path_cost=1)
n2 = Node(dict(index=1, char_val='a'), path_cost=0)

p_queue = PriorityQueue()

p_queue.push(n1, n1.path_cost)
p_queue.push(n2, n2.path_cost)

print(p_queue.pop()) # should be n1

print(p_queue.pop())


