from math import sqrt
from homework1_part1.maze import SequenceMaze
from homework1_part1.mazes import mazes_to_test


input = (6, [
             'a', '*', 'a', 'c', 'b', 'c',
             'c', '*', '*', '*', '*', 'b',
             '*', 'b', 'b', 'b', 'c', '*',
             '*', 'b', 'b', 'a', 'b', 'c',
             'c', '*', 'b', '*', '*', '*',
             'a', '*', '*', 'c', 'a', 'c'
])
#
# def read_dimension_val_and_char(dim_val, char_list):
#     seq_maze = list()
#     row = list()
#     c = 1
#     for char in char_list:
#         row.append(char)
#         if c % dim_val == 0:
#             seq_maze.append(row)
#             row = list()
#         c+=1
#     return seq_maze
#
# def display_seq_maze(seq_maze):
#     for row in seq_maze:
#         for val in row:
#             print(val, end='\t')
#         print('\n')
#
# seq_maze = read_dimension_val_and_char(input[0], input[1])


# mazeA = mazes_to_test.mazeA
#
#
# seq_maze = SequenceMaze(mazeA)
# print(seq_maze)


# boundary check

#inner
indexI = 1
#outer
indexO = 17
#dim_val
dim_val = 6


# if int(index / dim_val - 1)
# boundary check!
# if index % dim_val == 0 or index % dim_val == dim_val-1 :
#     return None

dim_val = 3
flat_list =[0,1,2,
            3,4,5,
            6,7,8,]

# implied coord  is 1,1 or row = 1 col = 1
test_i = 3
# flat list movements...
row = int(test_i / dim_val)
col = test_i % dim_val

# VALID INDEXES
# up and down
one_up = test_i - dim_val if test_i - dim_val > 0 else None
one_down = test_i + dim_val if test_i + dim_val < len(flat_list) else None
# print(one_up, one_down)

# one right and left
one_right = test_i + 1 if test_i % dim_val != dim_val-1 else None
one_left = test_i - 1 if test_i % dim_val != 0 else None

print(one_right, one_left)


