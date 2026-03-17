import warnings

import numpy as np
import matplotlib.pyplot as plt

class Oracle:
    def __init__(self, threshold, p_lie=0, seed=None):
        assert 0 <= p_lie < 0.5, "`p_lie` should be in the interval [0, 0.5)."
        self._rng = np.random.default_rng(seed)
        self.p_lie = p_lie
        self.threshold = threshold

    def query(self, x):
        """Return (possibly incorrect) label of `x`."""
        is_lying = self._rng.uniform() < self.p_lie
        return (x >= self.threshold) ^ is_lying


class Experiment:
    """Class to run an experiment."""
    def __init__(self, learner, tol=1e-2):
        self.learner = learner
        threshold = np.random.default_rng().uniform(learner.domain_lb, learner.domain_ub)
        self.oracle = Oracle(threshold)
        self.tol = tol

    def estimate_threshold(self):
        """Current estimate of the threshold."""
        return (self.learner.version_space_lb + self.learner.version_space_ub) / 2

    def is_converged(self):
        """Determine if the uncertainty on the threshold is small enough."""
        d = abs(self.learner.version_space_ub - self.learner.version_space_lb)
        return d <= self.tol

    def run(self, verbose=True):
        """Run an experiment until convergence. At each iteration we perform the following steps:

            (1) Ask the learner to suggest a point
            (2) Ask the oracle to label the point
            (3) Add the point and its corresponding label to the data set
        """

        i = 0
        while not self.is_converged():
            i += 1

            x = self.learner.query()
            y = self.oracle.query(x)
            self.learner.add_data(x, y)

        if verbose:
            print(f"Converged using {i} samples")
            print(f"Estimated threshold   {self.estimate_threshold():0.3f}")
            print(f"True threshold        {self.oracle.threshold:0.3f}")

            fig, ax = plt.subplots(figsize=(12,4))
            x = np.array(self.learner.x)
            y = np.array(self.learner.y)
            ax.scatter(x[~y], np.ones(np.sum(~y)))
            ax.scatter(x[y], np.ones(np.sum(y)))
            ax.axvline(self.oracle.threshold, color="gray", linestyle="--")
            ax.legend([0, 1, "Threshold"])
            ax.set_xlim(self.learner.domain_lb, self.learner.domain_ub)
            ax.set_yticks([])
            ax.grid(alpha=0.25)

        return i


# This class defines a discretization of the domain, e.g., {0.00, 0.01, ..., 0.99, 1.00} if bounds=(0,1) and resolution=101.
class DiscreteDomain:
    def __init__(self, bounds=None, resolution=501):
        if bounds:
            self.bounds_lower, self.bounds_upper = bounds
            assert self.bounds_lower < self.bounds_upper
        else:
            self.bounds_lower, self.bounds_upper = (0, 1)
        self.n_points = resolution
        self.points = np.linspace(self.bounds_lower, self.bounds_upper, self.n_points)


class ProbabilisticExperiment:
    def __init__(self, learner, p_lie=0.2, seed=None):
        self.learner = learner
        threshold = np.random.default_rng(seed).uniform(learner.theta[0], learner.theta[-1])
        # threshold = np.random.choice(learner.theta)
        self.oracle = Oracle(threshold, p_lie, seed)
        # some data about the experiment
        self.data = dict(label_wrong=[], entropy_pred_prob=[], entropy_posterior=[])

    @staticmethod
    def compute_entropy(p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return -np.sum(p*np.nan_to_num(np.log2(p)))

    def run(self, plot='final', n=100):
        """
        plot : str
            'final' or 'all'.
            if plot == 'all' then n is ignored and forced to 20.
            if plot == 'final' then n samples are drawn.
        n : int
            number of samples to draw.
        """

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = dict(zip(('p_theta', 'uncertainty'), prop_cycle.by_key()['color'][:2]))

        if (plot_all := plot == 'all'):
            n = 20
            fig, axes = plt.subplots(n//5, 5, sharex=True, constrained_layout=True, figsize=(5*3,n//5*3))

        for i in range(n):
            # query for point
            if self.learner.name == 'Least Confident':
                self.learner.compute_uncertainty()
            elif self.learner.name == 'Expected Entropy':
                self.learner.compute_expected_entropy(self.oracle.p_lie)
            else:
                raise ValueError

            x = self.learner.query_uncertainty()

            # query for label
            y = self.oracle.query(x)

            if plot_all:
                ax = axes.flat[i]
                axr = ax.twinx()
                ax.set_yticks([])
                axr.set_yticks([])

                at = self.learner.plot_uncertainty(axr, colors['uncertainty'])[0]
                pt = self.learner.plot_posterior(ax, colors['p_theta'])[0]
                tr = ax.axvline(self.oracle.threshold, color="grey", linestyle="--")

                if i % 5 == 0:
                    ax.set_ylabel('p(theta)')
                if i % 5 == 4:
                    axr.set_ylabel(self.learner.name)
                if i == 0:
                    ax.legend([pt, at, tr], ['p(theta)', self.learner.name, "Threshold", 0, 1])
                if i >= n-5:
                    ax.set_xlabel('theta')

            # update p(theta)
            self.learner.update_posterior(x, y, self.oracle.p_lie, add_data=True)
            if plot_all:
                xy = ax.scatter(self.learner.x, np.repeat(ax.get_ylim()[1], len(self.learner.x)), c=self.learner.y, cmap='cool')

            self.data['label_wrong'].append(y == (x < self.oracle.threshold))
            self.data['entropy_pred_prob'].append(self.compute_entropy(self.learner.compute_pred_prob()))
            self.data['entropy_posterior'].append(self.compute_entropy(self.learner.p_theta))

        if plot == 'final':
            fig, ax = plt.subplots()
            axr = ax.twinx()

            at = self.learner.plot_uncertainty(axr, colors['uncertainty'])[0]
            pt = self.learner.plot_posterior(ax, colors['p_theta'])[0]
            tr = ax.axvline(self.oracle.threshold, color="grey", linestyle="--")
            xy = ax.scatter(self.learner.x, np.repeat(ax.get_ylim()[1], len(self.learner.x)), c=self.learner.y, cmap='cool')

            ax.set_ylabel('p(theta)')
            ax.set_xlabel('theta')
            axr.set_ylabel(self.learner.name)
            ax.legend([pt, at, tr], ['p(theta)', self.learner.name, "Threshold"])
