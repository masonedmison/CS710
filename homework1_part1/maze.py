"""
A Maze and Node Implementation
search.py within the aima-python repo is heavily referenced <https://github.com/aimacode/aima-python/blob/master/search.py>
"""

from collections import namedtuple, Iterable
from math import pow

state = namedtuple('State',('location','value'))

class MazeProblem(object):
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        raise NotImplementedError

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        raise NotImplementedError

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            return state in self.goal
        else:
            return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1



class SequenceMaze(MazeProblem):

    # a set of valid actions for Part 1
    valid_actions = {'ab', 'a*', 'bc', 'b*', 'ca', 'c*', 'a*b','b*c', 'c*a', 'a**', 'b**','c**'}

    def __init__(self, maze_dim_tuple):
        # unpack tuple values to dim and char list
        dim_val, char_list = maze_dim_tuple
        # validate maze tuple values
        if not self._is_valid_maze(dim_val, char_list):
            raise ValueError('invalid maze values passed in')
        super().__init__( {'index':0, 'char_val':char_list[0]}, {'index':int(pow(dim_val,2) - 1), 'char_val': char_list[len(char_list) - 1]} )
        self._dim_val = dim_val
        # where seq is a list of Nodes
        self.maze_char_list = maze_dim_tuple[1]

    def __str__(self):
        """
        string representation of seq maze with implied 'rows'
        """
        count = 1
        maze_str = """"""
        for char in self.maze_char_list:
            maze_str += char + '\t'
            if count % self._dim_val == 0:
                maze_str += '\n\n'
            count += 1
        return maze_str

    @staticmethod
    def _is_valid_maze(dim_val, char_list):
        """ a quick check to make sure passed in values are valid
        *does not ensure that there is an actual solution to maze*"""
        return ( isinstance(dim_val, int) and
                isinstance(char_list, list ) and
                pow(dim_val,2)  == len(char_list)
                 )

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        # get valid indexes -- where state is tuple(<index>, val)

        cur_val = state['char_val']
        valid_indices = self._get_valid_indices(state, ['up','down','right', 'left'])

        valid_actions = set()
        # for every valid index check
        for k in valid_indices.keys():
            if valid_indices[k] is not None:
                index = valid_indices[k]
                # concat current state value and index value
                move_sequence = cur_val + self.maze_char_list[index].strip()
                if move_sequence in self.valid_actions:
                    valid_actions.add(k)
        return valid_actions

    def _get_valid_indices(self,state, actions):
        return {action: self._get_valid_index(state, action) for action in actions}

    def _get_valid_index(self, state, action):
        """
        helper method to retrieve valid index
        return valid index or None
        """
        cur_state_i = state['index']
        if action == 'up':
            res_i = cur_state_i - self._dim_val if cur_state_i - self._dim_val > 0 else None
        elif action == 'down':
            res_i = cur_state_i + self._dim_val if cur_state_i + self._dim_val < len(self.maze_char_list) else None
        elif action == 'right':
            res_i = cur_state_i + 1 if cur_state_i % self._dim_val != self._dim_val - 1 else None
        elif action == 'left':
            res_i = cur_state_i - 1 if cur_state_i % self._dim_val != 0 else None
        else:
            raise ValueError('incorrect action passed to Maze Sequence.result() must be up, down, right, or left')

        return res_i

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        res_i = self._get_valid_index(state, action)
        return {'index':res_i, 'char_val': self.maze_char_list[res_i]}

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        print(f'passed in state {state} goal state {self.goal}')
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state.  Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node.  Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""
    # to asssign star val to how is is being 'counted' or 'played'
    star_char_sub = {'a':'b', 'b':'c', 'c':'a', 'a*':'b', 'b*':'c','c*':'a'}
    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state # {index: <array_index>, char_val:<char_val>}
        self.count_star_as = None
        self.parent = parent
        if self.state['char_val'] == '*':
            if self.parent.count_star_as is None:
                self.count_star_as = self.star_char_sub[self.parent.state['char_val']]
            else:
                self.count_star_as = self.star_char_sub[self.parent.count_star_as]
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {} with star counting as (if any){}>".format(self.state, self.count_star_as)

    def __lt__(self, node):
        return self.state < node.state

    def _get_star_parent(self):
        """get predecessor of star
        where parents may be a sequence of stars"""
        pass

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        # if state char val is a *, get parents val if any and concat to *
        # eg state_char = *, parents_char_val = a --> a*
        # conditions to consider:
        #    parents val = a or b or c
        #    parents char val is a *
        #    parent = None
        #        this will never happen as initial is not *
        tmp_state = self.state
        if self.state['char_val'] == '*' and self.parent is not None:
            # value is a,b, c, a*, b*, c*
            # set state['char_val'] to new value use star char sub mapping
            val_concat = self.count_star_as + '*'
            # state used to check for valid moves
            tmp_state = {'index':self.state['index'], 'char_val':val_concat}
        return [self.child_node(problem, action)
                for action in problem.actions(tmp_state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action,
                         problem.path_cost(self.path_cost, self.state,
                                           action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.state for node in self.path()[1:]]
        # return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return ( isinstance(other, Node) and self.state == other.state

                )

    def __hash__(self):
        return hash(self.state)