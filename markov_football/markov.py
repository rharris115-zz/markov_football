from typing import List
from collections import namedtuple, OrderedDict, defaultdict
import numpy as np

Tx = namedtuple('Tx', ['s_from', 's_to', 'weight'])


class DuplicateTransitionError(Exception):
    pass


class MarkovChain(object):
    def __init__(self, transitions: List[Tx]):

        # Let's preserve the order of states as they appear in the list of transitions.

        tx_dict = defaultdict(dict)
        for t in transitions:
            inner = tx_dict[t.s_from]
            if t.s_to in inner:
                raise DuplicateTransitionError("A transition from '%r' to '%r' already exists." % (t.s_from, t.s_to))
            if t.weight <= 0:
                raise ValueError("Non-positive weight. weight='%f'" % t.weight)
            inner[t.s_to] = t.weight

        states = OrderedDict.fromkeys((s for t in transitions for s in (t.s_from, t.s_to))).keys()

        absorbing_states = OrderedDict.fromkeys(
            (s for s in states if s not in tx_dict or sum(tx_dict[s].values()) <= 0)).keys()

        transient_states = OrderedDict.fromkeys(
            (s for s in states if s not in absorbing_states)).keys()

        self.transient_states = tuple(transient_states)
        self.absorbing_states = tuple(absorbing_states)

        self.states = self.transient_states + self.absorbing_states
        self.state_indices = {s: i for i, s in enumerate(self.states)}

        transition_matrix = np.matrix(
            [[tx_dict.get(s_from, {s_from: 1.0}).get(s_to, 0.0) for s_to in self.states]
             for s_from in self.states], dtype=np.float64)

        self.transition_matrix = transition_matrix / np.tile(transition_matrix.sum(axis=1),
                                                             (1, transition_matrix.shape[1]))

        n_transient_states = len(self.transient_states)
        self.Q = self.transition_matrix[:n_transient_states, :n_transient_states]
        self.R = self.transition_matrix[:n_transient_states, n_transient_states:]

        self.N = np.linalg.inv(np.identity(len(self.transient_states), dtype=np.float64) - self.Q)
        self.B = self.N * self.R

    def calculate_outcome_given_state(self, s):
        if s not in self.states:
            raise ValueError('No such state. s=%r' % s)

        if s in self.absorbing_states:
            return {s: 1.0}

        return {state: self.B[self.state_indices[s], i] for i, state in enumerate(self.absorbing_states)}

    def simulate_next(self, s):
        if s not in self.states:
            raise ValueError('No such state. s=%r' % s)
        n = len(self.states)
        tx_probs = [self.transition_matrix[self.state_indices[s], col] for col in range(n)]
        s_index = np.random.choice(n, p=tx_probs)
        return self.states[s_index]
