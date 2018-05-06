from pprint import pprint
from multiprocessing import Pool, cpu_count
from collections import defaultdict

import numpy as np
from numpy.testing import assert_almost_equal
from scipy.misc import comb
from scipy import stats, special

from tqdm import tqdm

class HMM:
    def __init__(self, n_items, n_states, ALPHA=100):
        """
        :param n_items: no of items (I) in the dataset
        :param n_states: no of hidden states (K)
        :param ALPHA: weight of evidence from each prior (used to initialise model)

        Init:
        - starting probability (pi) <- Dirichlet prior a1 to aK
        - transition probability (A) <- Dirichlet prior a1 to aK
        - emission/observation model from latent class memberships
        """
        self.pool = Pool(cpu_count())

        self.ALPHA = ALPHA  # ALPHA_K. Prior for states
        prior_params = ALPHA * np.ones(shape=n_states) / n_states

        self.n_states = n_states
        self.n_items = n_items

        if ALPHA / n_states < 1 or ALPHA / n_items < 1:
            raise ValueError('ALPHA must be greater than n_items.')

        # starting probabilities
        self.pi = np.random.dirichlet(prior_params)

        # transition probabilities
        self.A = np.zeros(shape=(n_states, n_states))
        for i in range(n_states):
            self.A[i] = np.random.dirichlet(prior_params)

        # emission probabilities
        # NBD (negative binomial distribution) params; NBD models the no of items the user selects
        # TODO 20
        self.a = np.random.random(n_states) * 20
        self.b = np.random.random(n_states) * 20

        # theta (multinomial) models which items the user selects
        self.theta = np.zeros(shape=(n_states, n_items))
        prior_params = ALPHA * np.ones(shape=n_items) / n_items
        for i in range(n_states):
            self.theta[i] = np.random.dirichlet(prior_params)
        self.theta = np.transpose(self.theta)  # shape=(n_items, n_states)

    @staticmethod
    def emission_prob(theta, a, b, k, i):
        """
        :param k: state
        :param i: observed items at this observation/time-step
        """
        # a[k], b[k]: Gamma shape/scale params => NBD params for the k-th state
        # Gamma dist is conjugate to Poisson; NBD is a Poisson with Gamma mean
        x = len(i)  # #(items) in the observation
        p_i = np.transpose(theta)[k]  # item probs
        multinomial = stats.multinomial(n=x, p=p_i)

        # count number of items observed
        item_counts = np.zeros(theta.shape[0])
        for item in i:
            item_counts[item] += 1

        # get the joint pdf
        if x > 0:
            multinomial_term =  multinomial.pmf(item_counts)
            # assert np.isfinite(multinomial_term)

            return HMM.nbinom(x, a[k], b[k]) * multinomial_term
        else:
            # NOTE TODO <- not compatible with NBD_MLE
            # if user does not look at any items, probability of looking at 0 items
            return HMM.nbinom(x, a[k], b[k])

    @staticmethod
    def nbinom(x, a, b):
        """NOTE wrong in old version (see original paper)"""
        return comb(a+x-1, x) * (b/(b+1))**x * (1 - b/(b+1))**a

    @staticmethod
    def forward(n_states, a, b, theta, pi, A, observation_seq):
        """
        observation_seq is a sequence of observations from time t=1 to t=T.
        :param n_states:
        :param observation_seq:
        :param pi:
        :param A:
        """
        T = len(observation_seq)
        scaling_factors = [1]  # NOTE used in backward pass

        # alphas: probability of states at a time step (given the observation sequence)
        # dynamic programming - carry results over from previous time steps
        alphas = np.zeros(shape=(T, n_states))

        # initialisation
        emissions = np.diag([
            HMM.emission_prob(theta, a, b, k, observation_seq[0])
            for k in range(n_states)
        ])
        alphas[0, :] = np.dot(pi, emissions)

        # scaling - make the alphas sum to 1 at each step
        scaling_factor = alphas[0, :].sum()
        if scaling_factor:
            alphas[0, :] /= scaling_factor
        scaling_factors.append(scaling_factor)

        # recursion - use previously computed values (dynamic programming)
        for t in range(1, T):
            emissions = np.diag([
                HMM.emission_prob(theta, a, b, k, observation_seq[t])
                for k in range(n_states)
            ])
            alphas[t, :] = np.dot(np.dot(emissions, np.transpose(A)), alphas[t-1, :])

            scaling_factor = alphas[t, :].sum()
            if scaling_factor:
                alphas[t, :] /= scaling_factor
            scaling_factors.append(scaling_factor)

        return alphas, scaling_factors

    @staticmethod
    def backward(n_states, a, b, theta, A, observation_seq, scaling_factors):
        T = len(observation_seq)
        betas = np.zeros(shape=(T+1, n_states))

        # initialisation
        betas[T, :] = np.ones(n_states)

        # recursion
        for t in range(T-1, 0, -1):
            emissions = np.diag([
                HMM.emission_prob(theta, a, b, k, observation_seq[t])
                for k in range(n_states)
            ])
            betas[t, :] = np.dot(np.dot(emissions, np.transpose(A)), betas[t+1, :])  # TODO check no transpose

            # scaling
            scaling_factor = scaling_factors[t+1]
            if scaling_factor != 0:
                betas[t, :] /= scaling_factor

        # remove irrelevant 1st beta (only here for ease of notation)
        return betas[1:]

    @staticmethod
    def forward_backward(params):
        """
        For a user.
        """
        n_states, a, b, theta, pi, A, seq = params
        alphas, scaling_factors = HMM.forward(n_states, a, b, theta, pi, A, seq)
        betas = HMM.backward(n_states, a, b, theta, A, seq, scaling_factors)

        return alphas, scaling_factors, betas

    @staticmethod
    def gamma(params):
        """
        Calculates gamma for a user.
        """
        alphas, betas = params
        gammas = alphas * betas
        EPSILON = 10**(-10)
        for t in range(len(gammas)):
            for k in range(len(gammas[t])):
                if gammas[t][k] == 0:
                    gammas[t][k] = EPSILON
        return gammas

    @staticmethod
    def xi(params):
        alpha, beta, A, scaling_factors, T, n_states, a, b, theta, observation_seq = params

        xi = np.zeros(shape=(T-1, n_states, n_states))
        for t in range(T-1):
            emissions = np.array([HMM.emission_prob(theta, a, b, k, observation_seq[t+1]) for k in range(n_states)])
            for i in range(n_states):
                xi[t][i] = np.multiply(
                    np.multiply(
                        np.multiply(
                            alpha[t][i], A[i]
                        ), emissions
                    ), beta[t+1]
                )

            # t+2: implementation detail
            if scaling_factors[t+2] != 0:
                xi[t] /= scaling_factors[t+2]

        return xi

    @staticmethod
    def maximise_theta(params):
        i, k, gammas, observation_counts, total_counts, alpha, n_states, T = params
        n_items = len(observation_counts[0])  # NOTE not n_states
        gammas_ = np.swapaxes(gammas, 0, 2)  # gammas[k][t][u] from gammas[u][t][k]

        numerator = 0
        denominator = 0
        for t in range(T):
            # gammas for all users (dot) counts for all users
            numerator += (gammas_[k][t] * observation_counts[t][i]).sum()
            denominator += (gammas_[k][t] * total_counts[t]).sum()
        numerator += alpha / n_items - 1
        denominator += alpha - n_items

        theta_ik = numerator / denominator
        # if (theta_ik < 0):
        #     pprint(gammas)
        #     pprint(observation_counts)
        #     pprint(total_counts)
        #     print(numerator, denominator)
        assert (theta_ik >= 0)

        return i, k, theta_ik


    @staticmethod
    def NBD_MLE(a, b, gammas, observation_seqs):
        """
        Maximises a negative binomial distribution.
        See Minka 2002, section 2.1.
        """
        counts = [[0 for _ in u] for u in observation_seqs]

        n_users = len(observation_seqs)

        log = np.log
        polygamma = special.polygamma
        digamma = special.digamma

        T = len(observation_seqs[0])
        for u in range(len(counts)):  # count number of items selected for each user at each time
            for t in range(T):
                counts[u][t] = len(observation_seqs[u][t])

        average_counts = []
        for k in range(len(a)):  # K states
            numerator = 0
            denominator = 0
            for u in range(len(counts)):  # for each user
                for t in range(T):
                    # gammas: "mixing weights"
                    numerator += gammas[u][t][k] * ((counts[u][t] + a[k]) * b[k] / (b[k] + 1))
                    denominator += gammas[u][t][k]  # "number of observations"
            average_counts.append(numerator / denominator)
            if denominator == 0:
                print('denominator == 0')
                return a, b

        average_log_counts = []
        for k in range(len(a)):
            numerator = 0
            denominator = 0
            for u in range(len(counts)):  # for each user
                # NOTE numerical errors when gammas == 0
                for t in range(T):
                    numerator += gammas[u][t][k] * (digamma(counts[u][t] + a[k]) + log(b[k] / (b[k] + 1)))
                    denominator += gammas[u][t][k]

            if denominator == 0:
                print('denominator == 0')
                b[k] = average_counts[k] / a[k]  # Minka 2002 (3)
                return a, b
            average_log_counts.append(numerator / denominator)

        for k in range(len(a)):
            b[k] = average_counts[k] / a[k]  # Minka 2002 (3)
            # fast starting point
            if average_counts[k] > 0:
                a[k] = 0.5 / (log(average_counts[k]) - average_log_counts[k])
            else:
                # TODO
                print('average_counts == 0')
                return a, b

            # print(a)
            for i in range(4):
                a_new_inv = 1/a[k] + (average_log_counts[k] - log(average_counts[k]) + log(a[k]) - digamma(a[k])) / \
                                     (a[k]**2 * (1/a[k] - polygamma(1, a[k])))
            assert (a[k] >= 0)

        return a, b

    def baum_welch(self, observation_seqs, n_iterations=20):
        # TODO cache emission probs
        # constant values (for M-stage)
        n_users = len(observation_seqs)
        T = len(observation_seqs[0])

        observation_counts = np.zeros(shape=(T, self.n_items, n_users))
        total_counts = np.zeros(shape=(T, n_users))

        deltas = []

        for t in range(T):
            for i in range(self.n_items):
                for u in range(n_users):
                    count = 0
                    for item in observation_seqs[u][t]:
                        if item == i:
                            count += 1
                    observation_counts[t][i][u] = count

        for t in range(T):
            for u in range(n_users):
                total_counts[t][u] = len(observation_seqs[u][t])

        for i in tqdm(range(n_iterations)):
            # gets expectations based on the current parameter values
            alphas, betas, gammas, xis = self.expectation(observation_seqs)

            # updates parameters (maximises likelihood)
            delta = self.maximisation(gammas, xis, n_users, T, observation_counts, total_counts, observation_seqs)
            deltas.append(delta)
            if delta < 0.001:
                break
        return deltas  # break

    def expectation(self, observation_seqs):
        T = len(observation_seqs[0])

        params = []
        for seq in observation_seqs:
            params.append((self.n_states, self.a, self.b, self.theta, self.pi, self.A, seq))
        results = self.pool.map(HMM.forward_backward, params)
        # results = map(HMM.forward_backward, params)

        alphas = []
        scaling_factors = []
        betas = []

        for alphas_, scaling_factors_, betas_ in results:
            alphas.append(alphas_)
            scaling_factors.append(scaling_factors_)
            betas.append(betas_)

        """
        Expecation
        ----------
        Calculate gammas and xis, using alphas, betas and scaling probs.
        """
        # calculate gammas
        params = []
        for u in range(len(observation_seqs)):
            params.append((alphas[u], betas[u]))
        # gammas = list(map(HMM.gamma, params))
        gammas = self.pool.map(HMM.gamma, params)

        # calculate xis
        params = []
        for u in range(len(observation_seqs)):
            params.append((alphas[u], betas[u], self.A, scaling_factors[u], T, self.n_states, self.a, self.b, self.theta, observation_seqs[u]))
        # xis = list(map(HMM.xi, params))
        xis = self.pool.map(HMM.xi, params)

        return alphas, betas, gammas, xis

    def delta(self, a, b, theta, pi, A):
        delta = 0

        for k in range(self.n_states):
            delta += abs(a[k] - self.a[k]) + abs(b[k] - self.b[k]) + abs(pi[k] - self.pi[k])
            for l in range(self.n_states):
                delta += abs(A[k][l] - self.A[k][l])
            for i in range(self.n_items):
                delta += abs(theta[i][k] - self.theta[i][k])

        return delta

    def maximisation(self, gammas, xis, n_users, T, observation_counts, total_counts, observation_seqs):
        # initial and transition probabilities (HMM model parameters)
        pi = np.copy(self.pi)
        A = np.copy(self.A)

        # emission probabilities (multinomial model parameters)
        theta = np.copy(self.theta)

        # negative binomial model parameters
        a = np.copy(self.a)
        b = np.copy(self.b)


        """
        Decomposition of the term in Sahoo et al's paper -> can maximise each likelihood term independently.
        Maximisation by MAP (maximum-a-posteriori): maximise the likelihood of the posterior.
        """
        # maximise likelihood for pi (init probabilities)
        for i in range(self.n_states):
            pi[i] = sum(gammas[u][0][i] for u in range(n_users)) + self.ALPHA / self.n_states - 1
            pi[i] /= (sum(sum(gammas[u][0][k] for k in range(self.n_states)) for u in range(n_users)) + self.ALPHA - \
                      self.n_states)
        pi[i] /= pi[i].sum()  # NOTE as below

        # maximise likelihood for A (transition probabilities)
        for i in range(self.n_states):
            for j in range(self.n_states):
                xis_ = np.swapaxes(np.swapaxes(xis, 0, 2), 1, 3)
                gammas_ = np.swapaxes(gammas, 0, 2)

                A[i][j] = xis_[i, j].sum() + self.ALPHA / self.n_states - 1
                A[i][j] /= gammas_[i][:T-1].sum() + self.ALPHA - self.n_states

            # sometimes will not sum to 1 due to floating point inaccuracies
            A[i] /= A[i].sum()

        # maximise likelihood for theta (multinomial emission parameters)
        params = []
        for i in range(self.n_items):
            for k in range(self.n_states):
                params.append((i, k, gammas, observation_counts, total_counts, self.ALPHA, self.n_states, T))
        # results = map(HMM.maximise_theta, params)
        results = self.pool.map(HMM.maximise_theta, params)
        for i, k, theta_ik in results:
            theta[i][k] = theta_ik

        a, b = HMM.NBD_MLE(a, b, gammas, observation_seqs)

        delta = self.delta(a, b, theta, pi, A)

        # update w/ new values
        self.a = a
        self.b = b
        self.theta = theta
        self.pi = pi
        self.A = A

        return delta

    def item_rank(self, alphas):
        """
        Returns a list of relevant items for a user in order of relevance.
        """
        # calculate distribution over the states for the user at time t+1
        p_t_plus_1 = []
        for k in range(self.n_states):
            total = 0
            for l in range(self.n_states):
                total += alphas[-1][k] * self.A[l][k]
            p_t_plus_1.append(total)

        item_rank = defaultdict(float)
        for i in range(len(self. theta)):  # for each item
            item_rank[i] = -sum(p_t_plus_1[k] * (1 + self.b[k] * self.theta[i][k]) ** (-self.a[k])
                                for k in range(self.n_states))

        return sorted(item_rank, key=item_rank.__getitem__, reverse=True)


# if __name__ == "__main__":
#     hmm = HMM(15, 3, 100)
#     observation_seqs = [
#         [[0, 3, 2, 7], [1, 4], [2, 5, 6], [0, 0, 2, 3]],  # user 1, only tech
#         [[0, 1, 2, 8, 9, 2], [3, 1, 4, 1, 5, 9], [8, 10, 12, 7], [1, 2, 1, 1]], # user 2, mixture of tech and fashion. Heavy user.
#         [[0, 1], [2], [3], []],  # user 3, light user, mainly tech
#         [[13], [14], [], [0, 1]],  # user 4, power tools, also browsed tech
#         [[8, 9, 10], [9, 10, 11], [10, 11, 12], [8, 8, 9]]  # only fashion
#     ]
#     hmm.baum_welch(observation_seqs)
