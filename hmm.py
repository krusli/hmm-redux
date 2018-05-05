from pprint import pprint
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.misc import comb
from scipy import stats

class HMM:
    def __init__(self, n_items, n_states, alpha=100):
        """
        :param n_items: no of items (I) in the dataset
        :param n_states: no of hidden states (K)
        :param alpha: weight of evidence from each prior (used to initialise model)

        Init:
        - starting probability (pi) <- Dirichlet prior a1 to aK
        - transition probability (A) <- Dirichlet prior a1 to aK
        - emission/observation model from latent class memberships
        """
        self.pool = Pool(cpu_count())

        prior_params = alpha * np.ones(shape=n_states) / n_states

        self.n_states = n_states
        self.n_items = n_items

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
        for i in range(n_states):
            self.theta[i] = np.random.random(n_items)
            self.theta[i] /= sum(self.theta[i])  # normalise to sum to 1
        self.theta = np.transpose(self.theta)

    @staticmethod
    def emission_prob(theta, a, b, k, i):
        """
        :param k: state
        :param i: observed items at this observation/time-step
        """
        # a[k], b[k]: Gamma shape/scale params => NBD params for the k-th state
        # Gamma dist is conjugate to Poisson; NBD is a Poisson with Gamma mean
        p_i = b[k] / (b[k] + 1)  # probability for NBD
        x = len(i)  # #(items) in the observation

        p_i = np.transpose(theta)[k]  # item probs
        multinomial = stats.multinomial(n=x, p=p_i)

        # count number of items observed
        item_counts = np.zeros(theta.shape[0])
        for item in i:
            item_counts[item] += 1

        # get the joint pdf
        if x > 0:
            return HMM.nbinom(x, a[k], b[k]) * multinomial.pmf(item_counts)
        else:
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
        alpha, beta = params
        return alpha * beta

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

    def baum_welch(self, observation_seqs):
        # TODO no iteration
        # TODO cache emission probs
        # N_ITERATIONS = 1

        # constant values (for M-stage)
        n_users = len(observation_seqs)
        T = len(observation_seqs[0])

        obsv_counts = np.zeros(shape=(T, self.n_items, n_users))
        total_counts = np.zeros(shape=(T, n_users))
        for t in range(T):
            for i in range(self.n_items):
                for u in range(n_users):
                    obsv_counts[t][i][u] = sum(1 for item in observation_seqs[u][t] if item == i)
            for u in range(n_users):
                total_counts[t][u] = len(observation_seqs[u][t])

        alphas, betas, gammas, xis = self.expectation(observation_seqs)
        # self.maximisation(alphas, scaling_factors, betas, obsv_counts, total_counts)

    def expectation(self, observation_seqs):
        T = len(observation_seqs[0])

        params = []
        for seq in observation_seqs:
            params.append((self.n_states, self.a, self.b, self.theta, self.pi, self.A, seq))
        # results = self.pool.map(HMM.forward_backward, params)
        results = map(HMM.forward_backward, params)

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
            params.append(alphas[u], betas[u])
        gammas = map(HMM.gamma, params)

        # calculate xis
        params = []
        for u in range(len(observation_seqs)):
            params.append((alphas[u], betas[u], self.A, scaling_factors[u], T, self.n_states, self.a, self.b, self.theta, observation_seqs[u]))
        xis = map(HMM.xi, params)

        return alphas, betas, gammas, xis

    # TODO non-final signature
    def maximisation(self, alphas, scaling_factors, betas, observation_counts, total_counts):
        # M

        pass

