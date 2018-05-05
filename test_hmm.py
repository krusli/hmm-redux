import numpy as np
from numpy.testing import assert_almost_equal
from hmm import HMM

def test_init():
    hmm = HMM(n_items=15, n_states=3, alpha=100)

    # assertions
    # initial state probabilities must sum to 1
    assert_almost_equal(sum(hmm.pi), 1)

    # transition probabilities from a state i to all states in the next time step (including itself) must sum to 1
    for A_i in hmm.A:
        assert_almost_equal(sum(A_i), 1)

    # NBD params a and b should be initialised to random parameters (should exist)
    hmm.a
    hmm.b

    # theta_k: multinomial emission probability: *which* item is/will be selected by the user
    for theta_k in np.transpose(hmm.theta):
        assert_almost_equal(sum(theta_k), 1)

def test_NBD():
    p = HMM.nbinom(5, 1, 1)
    assert_almost_equal(p, 0.015625)
