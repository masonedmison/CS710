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

    def actions(self, state, parent_i):
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
    # valid_move_seqs = {'ab', 'a*', 'bc', 'b*', 'ca', 'c*', 'a*b', 'b*c', 'c*a'}
    valid_move_seqs = {'ab', 'a*', 'bc', 'b*', 'ca', 'c*', 'abc', 'a**', 'ab*', 'a*c' 'bca', 'b**', 'bc*', 'b*a', 'cab',
                       'c**', 'ca*', 'c*b'}
    # possible actions with 2 move added
    possible_actions = {'up', 'down', 'right', 'left', '2up', '2dwn', '2lft', '2rt', '1up1lft', '1up1rt', '1lft1up',
                        '1rt1up', '1dwn1lft', '1dwn1rt', '1rt1dwn', '1lft1dwn'}
    # map of 2 moves to get middle - to be used by self.result
    middle_map = {'2up': 'up', '2dwn': 'down', '2lft': 'left', '2rt': 'right', '1up1lft': 'up', '1up1rt': 'up',
                  '1lft1up': 'left','1rt1up': 'right', '1dwn1lft': 'down', '1dwn1rt': 'down', '1rt1dwn': 'right', '1lft1dwn': 'left'}

    def __init__(self, maze_dim_tuple):
        # unpack tuple values to dim and char list
        dim_val, char_list = maze_dim_tuple
        # validate maze tuple values
        if not self._is_valid_maze(dim_val, char_list):
            raise ValueError('invalid maze values passed in')
        super().__init__( {'index':0, 'char_val':char_list[0]}, {'index':int(pow(dim_val,2) - 1), 'char_val': char_list[len(char_list) - 1]} )
        self.dim_val = dim_val
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
            if count % self.dim_val == 0:
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

    def get_middle_val(self, state, action):
        middle_val_state = self.result(state, self.middle_map[action])
        return middle_val_state

    def actions(self, state, parent_i):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        # get valid indexes -- where state is tuple(<index>, val)
        def make_sequence_str():
            if k in self.middle_map.keys():
                middle_val = self.get_middle_val(state, k)['char_val']
                return cur_val + middle_val + self.maze_char_list[index].strip()
            else: # one step move
                return cur_val + self.maze_char_list[index].strip()

        cur_val = state['char_val']
        valid_indices = self._get_valid_indices(state, self.possible_actions) # returns {<action>:index}


        valid_actions = list()
        # for every valid index check
        for k in valid_indices.keys():
            if valid_indices[k] is not None:

                index = valid_indices[k]
                # concat current state value and index value and middle_val if any
                move_sequence = make_sequence_str()
                if move_sequence in self.valid_move_seqs:
                    if parent_i == index:
                        continue
                    valid_actions.append(k)
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
            res_i = cur_state_i - self.dim_val if cur_state_i - self.dim_val >= 0 else None
        elif action == 'down':
            res_i = cur_state_i + self.dim_val if cur_state_i + self.dim_val < len(self.maze_char_list) else None
        elif action == 'left':
            res_i = cur_state_i - 1 if cur_state_i % self.dim_val != 0 else None
        elif action == 'right':
            res_i = cur_state_i + 1 if cur_state_i % self.dim_val != self.dim_val - 1 else None
        elif action == '2up':
            res_i = (cur_state_i - self.dim_val) - self.dim_val if (
                                                                               cur_state_i - self.dim_val) - self.dim_val >= 0 else None
        elif action == '2dwn':
            res_i = (cur_state_i + self.dim_val) + self.dim_val if (cur_state_i + self.dim_val) + self.dim_val < len(
                self.maze_char_list) else None
        elif action == '2lft':
            res_i = cur_state_i - 2 if (
                                                   cur_state_i - 1) % self.dim_val != 0 and cur_state_i % self.dim_val != 0 else None
        elif action == '2rt':
            res_i = cur_state_i + 2 if (
                                                   cur_state_i + 1) % self.dim_val != self.dim_val - 1 and cur_state_i % self.dim_val != self.dim_val - 1 else None
        elif action == '1up1lft':
            res_i = (
                                cur_state_i - self.dim_val) - 1 if cur_state_i - self.dim_val > 0 and cur_state_i % self.dim_val != 0 else None
        elif action == '1up1rt':
            res_i = (
                                cur_state_i - self.dim_val) + 1 if cur_state_i - self.dim_val >= 0 and cur_state_i % self.dim_val != self.dim_val - 1 else None
        elif action == '1lft1up':
            res_i = (
                                cur_state_i - self.dim_val) - 1 if cur_state_i - self.dim_val - 1 >= 0 and cur_state_i % self.dim_val != 0 else None
        elif action == '1rt1up':
            res_i = (
                                cur_state_i - self.dim_val) + 1 if cur_state_i - self.dim_val > 0 and cur_state_i % self.dim_val != self.dim_val - 1 else None
        elif action == '1dwn1lft':
            res_i = (cur_state_i + self.dim_val) - 1 if cur_state_i + self.dim_val < len(
                self.maze_char_list) and cur_state_i % self.dim_val != 0 else None
        elif action == '1dwn1rt':
            res_i = (cur_state_i + self.dim_val) + 1 if cur_state_i + self.dim_val + 1 < len(
                self.maze_char_list) and cur_state_i % self.dim_val != self.dim_val - 1 else None
        elif action == '1rt1dwn':
            res_i = (cur_state_i + self.dim_val) + 1 if cur_state_i + self.dim_val + 1 < len(
                self.maze_char_list) and cur_state_i % self.dim_val != self.dim_val - 1 else None
        elif action == '1lft1dwn':
            res_i = (cur_state_i + self.dim_val) - 1 if cur_state_i + self.dim_val < len(
                self.maze_char_list) and cur_state_i % self.dim_val != 0 else None

        else:
            raise ValueError(
                f'incorrect action {action} passed to Maze Sequence.result() must be up, down, right, or left')

        return res_i

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        res_i = self._get_valid_index(state, action)
        return {'index':res_i, 'char_val': self.maze_char_list[res_i]}

    def goal_test(self, state):
        """Return True if the state is a goal."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        # if state char values are a,b,c the +1
        if state2['char_val'] == 'a' or 'b' or 'c':
            return c + 1

        else:
            c+2

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
    single_star_char_sub = {'a': 'b', 'b': 'c', 'c': 'a'}
    double_star_char_sub = {'a':'c', 'b':'a', 'c':'b'}
    def __init__(self, state, parent=None, action=None, path_cost=0, count_star_as=None, mid_val_state=None):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state # {index: <array_index>, char_val:<char_val>}
        self.count_star_as = count_star_as # kwarg for testing purposes
        self.parent = parent
        self.mid_val_state = mid_val_state  # to hold transitional value for 2 step moves
        self.action = action
        if self.state['char_val'] == '*':
            self.count_star_as = self._set_count_star_as()
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}, star counts as: {}>".format(self.state, self.count_star_as)

    def __lt__(self, node):
        return self.path_cost < node.path_cost

    def _set_count_star_as(self):
        """helper to set char method so as to not clutter the __init__ method"""
        star_map_type = self.double_star_char_sub if self.action in SequenceMaze.middle_map.keys() else self.single_star_char_sub
        if self.parent.count_star_as is None:
            count_star_as = star_map_type[self.parent.state['char_val']]
        else:
            count_star_as = star_map_type[self.parent.count_star_as]

        return count_star_as

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        tmp_state = self.state
        parent_i = self.parent.state['index'] if self.parent else -1
        if self.state['char_val'] == '*' and self.parent is not None:
            # set value to count star as (or simply what the star counted as when it was used)
            val_count_star_as = self.count_star_as
            # state used to check for valid moves - node state is not altered
            tmp_state = {'index':self.state['index'], 'char_val':val_count_star_as}
        valid_actions = problem.actions(tmp_state, parent_i)
        return [self.child_node(problem, action)
                for action in valid_actions]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        middle_val = None
        print('\n#########\naction', action)
        if action in problem.middle_map.keys():
            middle_val = problem.get_middle_val(self.state, action)
            print('mid_val in child node', middle_val)
            print('\n###############end')
        next_state = problem.result(self.state, action)

        next_node = Node(next_state, parent=self, action=action,
                         path_cost=problem.path_cost(self.path_cost, self.state,
                                           action, next_state), mid_val_state=middle_val)
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        # return [node.state for node in self.path()[1:]]
        # include middle tile in solution seq
        seq = []
        for node in self.path()[1:]:
            if node.mid_val_state is not None:
                print('mid val in solution', node.mid_val_state)
                seq.append(('TRANSITION NODE',node.mid_val_state))
            seq.append(node)

        return seq

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
        if not isinstance(other, Node):
            return False
        if self.parent is not None:
            if other.parent is None:
                return False
            # else compare states
            parents_equal = self.parent.state == other.parent.state and self.parent.count_star_as == other.parent.count_star_as
        else:
            parents_equal = False
        return ( isinstance(other, Node) and self.state == other.state
                 and self.count_star_as == other.count_star_as and parents_equal
                )

    def __hash__(self):
        # return hash(tuple([*self.state.values(), self.count_star_as]))
        parent_hash = hash(self.parent.state.values()) + hash(self.parent.count_star_as) if self.parent is not None else 0
        return hash(self.state.values()) + hash(self.count_star_as) + parent_hash