"""
Homework Assignment 1 Part 1
@author MasonEdmison
09/17/2019
Where all uninformed search algorithms live
all search algorithms (unless noted otherwise) return
length of solution; total number of states visisted; list of locations in solution

"""
from collections import deque
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


@algo_wrangler
def uniform_cost_search(problem):

    return None, None, None

@algo_wrangler
def iterative_deepening(problem):
    return None, None, None