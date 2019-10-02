from math import sqrt
from collections import namedtuple
from itertools import cycle
from homework1.maze import SequenceMaze, Node
from homework1.mazes import mazes_to_test



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

state = {'index':14, 'char_val':'b'}
self.dim_val = 6
maze_char_list = input[1]


possible_actions = ('up','down','right', 'left', '2up', '2dwn', '2lft', '2rt', '1up1lft', '1up1rt', '1lft1up',
                    '1rt1up', '1dwn1lft', '1dwn1rt', '1rt1dwn', '1lft1dwn')



def _get_valid_index(state, action):
    """
    helper method to retrieve valid index
    return valid index or None
    """
    cur_state_i = state['index']
    if action == 'up':
        res_i = cur_state_i - self.dim_val if cur_state_i - self.dim_val >= 0 else None
    elif action == 'down':
        res_i = cur_state_i + self.dim_val if cur_state_i + self.dim_val < len(maze_char_list) else None
    elif action == 'left':
        res_i = cur_state_i - 1 if cur_state_i % self.dim_val != 0 else None
    elif action == 'right':
        res_i = cur_state_i + 1 if cur_state_i % self.dim_val != self.dim_val - 1 else None
    elif action == '2up':
        res_i = (cur_state_i - self.dim_val)-self.dim_val if (cur_state_i - self.dim_val) - self.dim_val >= 0 else None
    elif action == '2dwn':
        res_i = (cur_state_i + self.dim_val) + self.dim_val if (cur_state_i + self.dim_val) + self.dim_val < len(maze_char_list) else None
    elif action == '2lft':
        res_i = cur_state_i - 2 if (cur_state_i -1) % self.dim_val != 0 else None
    elif action == '2rt':
        res_i = cur_state_i + 2 if (cur_state_i+1) % self.dim_val != self.dim_val - 1 else None
    elif action == '1up1lft':
        res_i = (cur_state_i - self.dim_val) - 1 if cur_state_i - self.dim_val > 0 and cur_state_i % self.dim_val != 0 else None
    elif action == '1up1rt':
        res_i = (cur_state_i - self.dim_val) + 1 if cur_state_i - self.dim_val > 0 and cur_state_i % self.dim_val != self.dim_val - 1 else None
    elif action == '1lft1up':
        res_i = (cur_state_i - self.dim_val) - 1 if cur_state_i - self.dim_val > 0 and cur_state_i % self.dim_val != 0 else None
    elif action == '1rt1up':
        res_i = (cur_state_i - self.dim_val) + 1 if cur_state_i - self.dim_val > 0 and cur_state_i % self.dim_val != self.dim_val - 1 else None
    elif action == '1dwn1lft':
        res_i = (cur_state_i + self.dim_val) - 1 if cur_state_i + self.dim_val > 0 and cur_state_i % self.dim_val != 0 else None
    elif action == '1dwn1rt':
        res_i = (cur_state_i + self.dim_val) + 1 if cur_state_i + self.dim_val > 0 and cur_state_i % self.dim_val != self.dim_val - 1 else None
    elif action == '1rt1dwn':
        res_i = (cur_state_i + self.dim_val) + 1 if cur_state_i + self.dim_val > 0 and cur_state_i % self.dim_val != self.dim_val - 1 else None
    elif action == '1lft1dwn':
        res_i = (cur_state_i + self.dim_val) - 1 if cur_state_i + self.dim_val > 0 and cur_state_i % self.dim_val != 0 else None

    else:
        raise ValueError(f'incorrect action {action} passed to Maze Sequence.result() must be up, down, right, or left')

    return res_i

for act in possible_actions:
    i = _get_valid_index(state, act)
    print(f'index: {i} char val:{input[1][i] if i is not None else None} action:{act}')
