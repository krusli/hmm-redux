from multiprocessing import Pool, cpu_count
import numpy as np

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

        if not prior_params:
            prior_params = alpha * np.ones(shape=n_states) / n_states

        # starting probabilities
        self.A = np.random.dirichlet(prior_params)

        # transition probabilities
        self.pi = np.zeros(shape=(n_states, n_states))
        for i in range(n_states):
            self.pi[i] = np.random.dirichlet(prior_params)

        # emission probabilities
        # NBD (negative binomial distribution) params
        self.a = np.random(n_states)
        self.b = np.random(n_states)

        self.theta = np.zeros(shape=(n_states, n_items))
        for i in range(n_states):
            self.theta[i] = np.random(n_items)
            self.theta[i] /= sum(self.theta[i])  # normalise to sum to 1
