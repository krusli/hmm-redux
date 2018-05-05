from multiprocessing import Pool, cpu_count
import numpy as np
from scipy import special, stats, misc

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
        prior_params = alpha * np.ones(shape=n_states) / n_states

        self.n_states = n_states
        self.n_items = n_items

        # starting probabilities
        self.A = np.random.dirichlet(prior_params)

        # transition probabilities
        self.pi = np.zeros(shape=(n_states, n_states))
        for i in range(n_states):
            self.pi[i] = np.random.dirichlet(prior_params)

        # emission probabilities
        # NBD (negative binomial distribution) params; NBD models the no of items the user selects
        self.a = np.random.random(n_states)
        self.b = np.random.random(n_states)

        # theta (multinomial) models which items the user selects
        self.theta = np.zeros(shape=(n_states, n_items))
        for i in range(n_states):
            self.theta[i] = np.random.random(n_items)
            self.theta[i] /= sum(self.theta[i])  # normalise to sum to 1

    def emission_prob(self, k, i):
        """
        :param k: state
        :param i: observed items at an observation/time-step
        """

        # a[k], b[k]: Gamma shape/scale params => NBD params for the k-th state
        # Gamma dist is conjugate to Poisson; NBD is a Poisson with Gamma mean

        p = b[k] / (b[k] + 1)  # probability for NBD
        x = len(i)  # #(items) in the observation

    @staticmethod
    def nbinom(x, a, b):
        """NOTE wrong in old version"""
        return misc.comb(a+x-1, x) * (b/(b+1))**x * (1 - b/(b+1))**a

    def forward(self, observation_seq):
        """
        observation_seq is a sequence of observations from time t=1 to t=T.
        """
        T = len(observation_seq)

        # alphas: probability of states at a time step (given the observation sequence)
        # dynamic programming - carry results over from previous time steps
        alphas = np.zeros(shape=(T, self.n_states))

        # initialisation
        # emissions = np.diag()
