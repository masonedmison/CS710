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

node_to_tuple = lambda node: tuple([node.state['index'], node.count_star_as if node.count_star_as is not None else node.state['char_val'], node.parent])


pare_node1 = Node(dict(index=0,char_val='a'), count_star_as='a')
pare_node2 = Node(dict(index=0,char_val='a'), count_star_as='b')

n1 = Node({'index':0, 'char_val':'a'}, count_star_as='b', parent=pare_node2)
n2 = Node({'index':0, 'char_val':'a'}, count_star_as='b', parent=pare_node1)
print('node1', hash(n1))
print('node2', hash(n2))
print(n1==n2)

nodes = [n1,n2]

