"""
Homework Assignment 1 Part 1
@author MasonEdmison
09/17/2019
Where all uninformed search algorithms live
all search algorithms (unless noted otherwise) return
length of solution; total number of states visisted; list of locations in solution

"""
import sys
from collections import deque
from utils import memoize
from homework1_part1.priority_queue import PriorityQueue
from homework1_part1.maze import Node


SEARCH_ALGORITHMS = []
node_to_tuple = lambda node: tuple([node.state['index'], node.count_star_as if node.count_star_as is not None else node.state['char_val'], node.parent])

# decorator to grab all search algorithms so we can send problem to each
def algo_wrangler(algo):
    SEARCH_ALGORITHMS.append(algo)
    return algo


@algo_wrangler
def breadth_first_search(problem):
    """
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    explored = set()
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node.depth, len(explored), node.solution()
    frontier = deque([node])
    while frontier:
        node = frontier.popleft()
        # a putzy work around since state is a dictionary
        explored.add(node_to_tuple(node))
        children = node.expand(problem)
        for child in children:
            if node_to_tuple(child) not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child.depth, len(explored), child.solution()
                frontier.append(child)
    return  None, None, None

@algo_wrangler
def depth_first_search_with_backtrack(problem):
    """Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Does not get trapped by loops.
        If two paths reach a state, only use the first one. [Figure 3.7]"""
    node = Node(problem.initial)
    frontier = deque([node])  # Stack implementation using deque
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node.depth, len(explored), node.solution()
        explored.add(node_to_tuple(node))
        frontier.extend(child for child in node.expand(problem)
                        if node_to_tuple(child) not in explored and
                        child not in frontier)
    return None, None, None

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

    return None, None, None



@algo_wrangler
def uniform_cost_search(problem):
    return best_first_graph_search(problem, lambda node: node.path_cost)


def depth_limited_search(problem, limit=50):
    """[Figure 3.17]"""

    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node.depth, 'not available', node.solution()
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)

@algo_wrangler
def iterative_deepening_search(problem):
    """[Figure 3.18]"""
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result