import jax.numpy as jnp
from jax import random, tree_util, tree_map
from jax.scipy.stats import norm, uniform
from jax.typing import ArrayLike

from ..random import split_like_tree


class Distribution:
    periodic = False
    reflective = False
    
    def transform(self, u):
        pass

    def log_probability(self, y):
        pass
    
    def sample(self, key, shape=()):
        pass
    
    @property
    def mean(self):
        pass

    @property
    def variance(self):
        pass


class Normal(Distribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
    
    def transform(self, u):
        return norm.ppf(u, self.loc, self.scale)

    def log_probability(self, y):
        return norm.logpdf(y, self.loc, self.scale)
    
    def sample(self, key, shape=(), dtype=jnp.float_):
        return self.loc + self.scale * random.normal(key, shape=shape, dtype=dtype)

    @property
    def mean(self):
        return self.loc

    @property
    def variance(self):
        return self.scale**2


class Uniform(Distribution):
    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high
        self.loc = low
        self.scale = high - low

    def transform(self, u):
        return self.loc + self.scale * u
    
    def log_probability(self, y):
        uniform.logpdf(y, self.loc, self.scale)

    def sample(self, key, shape=(), dtype=jnp.float_):
        return random.uniform(key, shape=shape, dtype=dtype, minval=self.low, maxval=self.high)
    
    @property
    def mean(self):
        return (self.low + self.high) / 2.0
    
    @property
    def variance(self):
        return (self.high - self.low)**2 / 12.0


class CircularUniform(Uniform):
    periodic = True
    def __init__(self, low=0.0, high=2*jnp.pi):
        super().__init__(low, high)

    def transform(self, u):
        # prior_transform expects u from unconstrained space for periodic distributions
        return super().transform(jnp.mod(u, 1.0))


class JointDistribution(Distribution):
    def __init__(self, distributions):
        self.distributions = distributions  # pytree structure
        self.periodic = tree_map(lambda x: x.periodic, self.distributions)
        self.reflective = tree_map(lambda x: x.reflective, self.distributions)
        
    def transform(self, u):
        return tree_map(
            lambda x, v: x.transform(v), 
            self.distributions,
            u
        )

    def log_probability(self, y):
        logp = tree_map(
            lambda x, z: x.log_probability(z), 
            self.distributions, 
            y
        )
        return sum(tree_util.tree_leaves(logp))
        
    def sample(self, key, shape=(), dtype=jnp.float_):
        keys = split_like_tree(key, self.distributions)
        return tree_map(
            lambda k, x: x.sample(k, shape=shape, dtype=dtype), 
            keys,
            self.distributions, 
        )

    @property
    def mean(self):
        return tree_map(lambda x: x.mean, self.distributions)
    
    @property
    def variance(self):
        return tree_map(lambda x: x.variance, self.distributions)
