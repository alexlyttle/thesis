import jax, dynesty
import numpy as np
from jax.tree_util import tree_leaves

from .results import NestedResults


def _nested_sampler(model, xla=True, **kwargs) -> dynesty.NestedSampler:
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
        self._sampler = _nested_sampler(model, xla=xla, **kwargs)

    def run_nested(self, **kwargs) -> NestedResults:
        self._sampler.run_nested(**kwargs)
        return self._get_results()
        
    # def resample(self, rstate=None):
    #     samples = self._sampler.results.samples_equal(rstate)
    #     return self.model.unflatten(list(samples.T))

    def _get_results(self) -> NestedResults:
        res = self._sampler.results.asdict()
        res["samples"] = self.model.unflatten(list(res["samples"].T))
        res["samples_u"] = self.model.unflatten(list(res["samples_u"].T))
        _ = res.pop("bound")
        _ = res.pop("samples_bound")
        return NestedResults(**res)
