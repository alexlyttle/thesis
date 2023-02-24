from .distributions import Distribution
from jax.tree_util import tree_flatten, tree_unflatten


class Model:
    symbols = None  # PyTreeLike

    def __init__(self, prior: Distribution):
        self.prior = prior
        leaves, self.treedef = tree_flatten(self.prior.mean)
        self.ndim = len(leaves)
        
    def unflatten(self, leaves):
        return tree_unflatten(self.treedef, leaves)

    def log_likelihood(self, params):
        ...

    def prior_transform(self, uparams):
        ...
