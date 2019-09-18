"""
Homework Assignment 1 Part 1
@author MasonEdmison
09/17/2019
Where all uninformed search algorithms live
"""


SEARCH_ALGORITHMS = []

# decorator to grab all search algorithms so we can send problem to each
def algo_wrangler(algo):
    SEARCH_ALGORITHMS.append(algo)
    return algo

@algo_wrangler
def breadth_first_search(problem):
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