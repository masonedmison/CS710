from math import sqrt
from utils import memoize
from homework1.priority_queue import PriorityQueue
from homework1.maze import Node


# node to tuple from part1 uninformed search - mod as needed
node_to_tuple = lambda node: tuple([node.state['index'], node.count_star_as if node.count_star_as is not None else node.state['char_val'], node.parent])


# heuristic functions and helpers
def _to_coordinate(state_index, depth):
    """take index of 1d list and convert it to a coordinate"""
    return int(state_index / 6), state_index % depth


def manhatten_distance(state_indexA, state_indexB, dim):
    x1, y1  = _to_coordinate(state_indexA)
    x2, y2 = _to_coordinate(state_indexB)
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean_distance(state_indexA, state_indexB, dim):
    x1, y1  = _to_coordinate(state_indexA)
    x2, y2 = _to_coordinate(state_indexB)
    return sqrt( pow((x1 - x2), 2) + pow((y1 - y2), 2) )

########################################################

def best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node.depth, len(explored), node.solution()
        explored.add(node_to_tuple(node))
        for child in node.expand(problem):
            if node_to_tuple(child) not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)

    return None, len(explored), None


def greedy_search(problem, h = None):
    pass

def astar_search(problem, h = None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    goal = problem.goal['index'] # get goal state index
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n.state['index'], goal, problem.dim_val))

# unsure of which iteration to use...
def local_search_gradient_ascent(problem, schedule):
    pass

