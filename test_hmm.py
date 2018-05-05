import numpy as np
from hmm import HMM

def test_init():
    hmm = HMM(n_items=15, n_states=3, alpha=100)

    # assertions
    # initial state probabilities must sum to 1
    np.testing.assert_almost_equal(sum(hmm.A), 1)

    # transition probabilities from a state i to all states in the next time step (including itself) must sum to 1
    for pi_i in hmm.pi:
        np.testing.assert_almost_equal(sum(pi_i), 1)

    # NBD params a and b should be initialised to random parameters (should exist)
    hmm.a
    hmm.b

    # theta_k: multinomial emission probability: *which* item is/will be selected by the user
    for theta_k in hmm.theta:
        np.testing.assert_almost_equal(sum(theta_k), 1)
