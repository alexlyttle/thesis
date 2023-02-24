import jax, dynesty
import dynesty.plotting as dyplot
import numpy as np
import jax.numpy as jnp
from jax.tree_util import tree_leaves
from collections import namedtuple


NestedResults = namedtuple(
    "NestedResults", 
    ['nlive',
     'niter',
     'ncall',
     'eff',
     'samples',
     'blob',
     'samples_id',
     'samples_it',
     'samples_u',
     'logwt',
     'logl',
     'logvol',
     'logz',
     'logzerr',
     'information',
     'bound',
     'bound_iter',
     'samples_bound',
     'scale'],
)


def importance_weights(results: NestedResults):
    weights = jnp.exp(results.logwt - results.logz[-1])
    return weights / weights.sum()


def resample(key, results: NestedResults):
    """TODO: Write own resampling using jax."""
    raise NotImplementedError()


def nested_sampler(model, xla=True, **kwargs):
    ndim = model.ndim
    assert ndim == kwargs.pop("ndim", ndim)  # check ndims match
    
    def log_likelihood(x):
        params = model.unflatten(x)
        return model.log_likelihood(params)

    def prior_transform(u):
        uparams = model.unflatten(u)
        params = model.prior_transform(uparams)
        return tree_leaves(params)

    _gradient = kwargs.pop("gradient", None)
    gradient = None
    if _gradient is not None:
        def gradient(u):
            uparams = model.unflatten(u)
            return _gradient(uparams)

    if xla:
        log_likelihood = jax.jit(log_likelihood)
        prior_transform = jax.jit(prior_transform)
        if gradient is not None:
            gradient = jax.jit(gradient)        
    
    _periodic, = np.where(tree_leaves(model.prior.periodic))
    _reflective, = np.where(tree_leaves(model.prior.reflective))

    if _periodic.size == 0:
        _periodic = None
    if _reflective.size == 0:
        _reflective = None
        
    periodic = kwargs.pop("periodic", _periodic)
    reflective = kwargs.pop("reflective", _reflective)
    
    return dynesty.NestedSampler(
        log_likelihood, prior_transform, model.ndim, gradient=_gradient,
        periodic=periodic, reflective=reflective, **kwargs
    )


class NestedSampler:
    """Wrapper for dynesty.NestedSampler"""
    def __init__(self, model, xla=True, **kwargs):
        self.model = model
        self._sampler = nested_sampler(model, xla=xla, **kwargs)

    def run_nested(self, **kwargs):
        self._sampler.run_nested(**kwargs)
        
    def resample(self, rstate=None):
        samples = self._sampler.results.samples_equal(rstate)
        return self.model.unflatten(list(samples.T))

    @property
    def results(self):
        res = self._sampler.results.asdict()
        res["samples"] = self.model.unflatten(list(res.samples.T))
        res["samples_i"] = self.model.unflatten(list(res.samples_u.T))
        return NestedResults(**res)

    # @property
    # def dynesty_results(self):
    #     return self._sampler.results
    
    def _flatten_dyplot_kwargs(self, kwargs: dict):
        kwargs["labels"] = tree_leaves(kwargs.get("labels", self.model.symbols))

        if "truths" in kwargs:
            kwargs["truths"] = tree_leaves(kwargs["truths"])
        
        def is_leaf(leaf):
            # Unlikely that model tree contains tuples since assignment not
            # supported. This should be fairly safe.
            if isinstance(leaf, tuple):
                return len(leaf) == 2
            return isinstance(leaf, (int, float))

        if "span" in kwargs:
            kwargs["span"] = tree_leaves(kwargs["span"], is_leaf=is_leaf)
            # maybe flatten and check treedef == model.treedef
            
        return kwargs

    def traceplot(self, **kwargs):
        return dyplot.traceplot(
            self._sampler.results, 
            **self._flatten_dyplot_kwargs(kwargs)
        )

    def cornerplot(self, **kwargs):
        return dyplot.cornerplot(
            self._sampler.results, 
            **self._flatten_dyplot_kwargs(kwargs)
        )
