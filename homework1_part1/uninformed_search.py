"""
Homework Assignment 1 Part 1
@author MasonEdmison
09/17/2019
Where all uninformed search algorithms live
all search algorithms (unless noted otherwise) return
length of solution; total number of states visisted; list of locations in solution

"""
from collections import deque
from homework1_part1.maze import Node


SEARCH_ALGORITHMS = []

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
        return node.depth, node.solution()
    frontier = deque([node])
    node_to_tuple = lambda node: tuple([*node.state.values(), node.count_star_as])
    while frontier:
        node = frontier.popleft()
        # a putzy work around since state is a dictionary
        explored.add(node_to_tuple(node))
        print(f'expanding {node}')
        children = node.expand(problem)
        print(f'where children are {children}')
        for child in children:
            # if child not in frontier:
            #     if problem.goal_test(child.state):
            #         return child.depth, child.solution()
            #     frontier.append(child)
            print(node_to_tuple(child) not in explored, child not in frontier )
            if node_to_tuple(child) not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child.depth, len(explored), child.solution()
                frontier.append(child)

    return None, None, None

@algo_wrangler
def depth_first_search_with_backtrack(problem):
    return None, None, None

@algo_wrangler
def uniform_cost_search(problem):
    return None, None, None

@algo_wrangler
def iterative_deepening(problem):
    return None, None, None