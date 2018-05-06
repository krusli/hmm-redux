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


def test_gamma():
    hmm = HMM(15, 3, 100)

    alphas = np.array([[ 0.26041011,  0.18494247,  0.55464742],
       [ 0.81832943,  0.09157791,  0.09009266],
       [ 0.62316769,  0.18980412,  0.18702818],
       [ 0.42329472,  0.28790271,  0.28880257]])

    betas = np.array([[ 2.46145618,  0.27263977,  0.26822218],
        [ 1.85320493,  0.57133421,  0.56296995],
        [ 1.26246349,  0.86630114,  0.86899945],
        [ 1.        ,  1.        ,  1.        ]])

    gamma = alphas * betas
    gamma_hmm = hmm.gamma((alphas, betas))

    assert (gamma == gamma_hmm).all()


def test_gamma_0_in_alpha_or_beta():
    hmm = HMM(15, 3, 100)

    alphas = np.array([[ 0,  0.18494247,  0.55464742],
       [ 0.81832943,  0.09157791,  0.09009266],
       [ 0.62316769,  0.18980412,  0.18702818],
       [ 0.42329472,  0.28790271,  0.28880257]])

    betas = np.array([[ 2.46145618,  0.27263977,  0.26822218],
        [ 1.85320493,  0.57133421,  0.56296995],
        [ 1.26246349,  0.86630114,  0.86899945],
        [ 1.        ,  1.        ,  1.        ]])

    gamma_hmm = hmm.gamma((alphas, betas))
    assert not np.isin(0, gamma_hmm)


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
