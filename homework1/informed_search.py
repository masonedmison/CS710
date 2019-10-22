from math import sqrt
from random import choice
from utils import memoize, argmax_random_tie, argmin_random_tie
from homework1.priority_queue import PriorityQueue
from homework1.maze import Node


INFORMED_SEARCH_ALGORITHMS = []

# node to tuple from part1 uninformed search - mod as needed
node_to_tuple = lambda node: tuple([node.state['index'], node.count_star_as if node.count_star_as is not None else node.state['char_val'], node.parent])

infinity = float('inf')

# decorator to grab all search algorithms so we can send problem to each
def algo_wrangler(algo):
    INFORMED_SEARCH_ALGORITHMS.append(algo)
    return algo

#####################################################
# heuristic functions and helpers
def _to_coordinate(state_index, depth):
    """take index of 1d list and convert it to a coordinate"""
    return int(state_index / depth), state_index % depth


def manhatten_distance(state_indexA, state_indexB, dim):
    x1, y1  = _to_coordinate(state_indexA, dim)
    x2, y2 = _to_coordinate(state_indexB, dim)
    return abs(x1 - x2) + abs(y1 - y2)

def euclidean_distance(state_indexA, state_indexB, dim):
    x1, y1  = _to_coordinate(state_indexA, dim)
    x2, y2 = _to_coordinate(state_indexB, dim)
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
            return node.depth, len(explored), node.solution(), node.path_cost
        explored.add(node_to_tuple(node))
        for child in node.expand(problem):
            if node_to_tuple(child) not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                if f(child) < frontier[child]:
                    del frontier[child]
                    frontier.append(child)

    return None, len(explored), None, None

@algo_wrangler
def greedy_search(problem, h):
    """ Greedy search is best first graph search with f(n) = h(n) where h is hueristic function"""
    goal = problem.goal['index'] # get goal state index
    return best_first_graph_search(problem, lambda n: h(n.state['index'], goal, problem.dim_val))

@algo_wrangler
def astar_search(problem, h):
    """A* search is best-first graph search with f(n) = g(n)+h(n)."""
    goal = problem.goal['index'] # get goal state index
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n.state['index'], goal, problem.dim_val))

@algo_wrangler
def hill_climbing(problem, h, rand_cur=None):
    """From the initial node, keep choosing the neighbor with highest value,
    stopping when no neighbor is better. """
    if rand_cur is None:
        current = Node(problem.initial)
    else:
        current=rand_cur
    goal = problem.goal['index'] # goal index
    # path plus hueristic function
    h_lam = lambda node: h(node.state['index'], goal, problem.dim_val)
    while True:
        if h_lam(current) == h_lam(Node(problem.goal)):
            break
        neighbors = current.expand(problem)
        if not neighbors:
            break
        neighbor = argmin_random_tie(neighbors, key=h_lam)
        if h_lam(neighbor) <= h_lam(current):
            break
        current = neighbor
    if not problem.goal_test(current.state):
        rand_i = choice(list(range(problem.dim_val*problem.dim_val)))
        rand_state= dict(index=rand_i, char_val=problem.maze_char_list[rand_i])
        hill_climbing(problem,h , rand_cur=Node(rand_state))
    return current.depth, None, current.solution(), current.path_cost

