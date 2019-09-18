
input = (6, [
             'a', '*', 'a', 'c', 'b', 'c',
             'c', '*', '*', '*', '*', 'b',
             '*', 'b', 'b', 'b', 'c', '*',
             '*', 'b', 'b', 'a', 'b', 'c',
             'c', '*', 'b', '*', '*', '*',
             'a', '*', '*', 'c', 'a', 'c'
])

def read_dimension_val_and_char(dim_val, char_list):
    seq_maze = list()
    row = list()
    c = 1
    for char in char_list:
        row.append(char)
        if c % dim_val == 0:
            seq_maze.append(row)
            row = list()
        c+=1
    return seq_maze

def display_seq_maze(seq_maze):
    for row in seq_maze:
        for val in row:
            print(val, end='\t')
        print('\n')

seq_maze = read_dimension_val_and_char(input[0], input[1])


display_seq_maze(seq_maze)
