import numpy as np
from numpy.testing import assert_almost_equal
from hmm import HMM

def test_init():
    hmm = HMM(n_items=15, n_states=3, ALPHA=100)

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


def test_expectation():
    hmm = HMM(15, 3, 100)

    observation_seqs = [
        [[0, 3, 2, 7], [1, 4], [2, 5, 6], [0, 0, 2, 3]],  # user 1, only tech
        [[0, 1, 2, 8, 9, 2], [3, 1, 4, 1, 5, 9], [8, 10, 12, 7], [1, 2, 1, 1]],
        # user 2, mixture of tech and fashion. Heavy user.
        [[0, 1], [2], [3], []],  # user 3, light user, mainly tech
        [[13], [14], [], [0, 1]],  # user 4, power tools, also browsed tech
        [[8, 9, 10], [9, 10, 11], [10, 11, 12], [8, 8, 9]]  # only fashion
    ]

    n_users = len(observation_seqs)
    T = len(observation_seqs[0])

    obsv_counts = np.zeros(shape=(T, hmm.n_items, n_users))
    total_counts = np.zeros(shape=(T, n_users))
    for t in range(T):
        for i in range(hmm.n_items):
            for u in range(n_users):
                obsv_counts[t][i][u] = sum(1 for item in observation_seqs[u][t] if item == i)
        for u in range(n_users):
            total_counts[t][u] = len(observation_seqs[u][t])

    alphas, betas, gammas, xis = hmm.expectation(observation_seqs)

    # TODO STUB