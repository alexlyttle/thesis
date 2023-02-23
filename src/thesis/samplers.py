import jax, dynesty
import numpy as np
import jax.numpy as jnp
from jax import tree_util


def nested_sampler(model, xla=True, **kwargs):
    leaves, treedef = tree_util.tree_flatten(model.prior.mean)
    ndim = len(leaves)
#     print(ndim)
    assert ndim == kwargs.pop("ndim", ndim)  # check ndims match

    def log_likelihood(x):
        params = tree_util.tree_unflatten(treedef, x)
        return model.log_likelihood(params)

    def prior_transform(u):
        uparams = tree_util.tree_unflatten(treedef, u)
        return tree_util.tree_leaves(model.prior_transform(uparams))

    _gradient = kwargs.pop("gradient", None)
    gradient = None
    if _gradient is not None:
        def gradient(u):
            uparams = tree_util.tree_unflatten(treedef, u)
            return _gradient(uparams)

    if xla:
        log_likelihood = jax.jit(log_likelihood)
        prior_transform = jax.jit(prior_transform)
        if gradient is not None:
            gradient = jax.jit(gradient)        
    
    _periodic, = np.where(tree_util.tree_leaves(model.prior.periodic))
    _reflective, = np.where(tree_util.tree_leaves(model.prior.reflective))

    if _periodic.size == 0:
        _periodic = None
    if _reflective.size == 0:
        _reflective = None
        
    periodic = kwargs.pop("periodic", _periodic)
    reflective = kwargs.pop("reflective", _reflective)
    
    return dynesty.NestedSampler(
        log_likelihood, prior_transform, ndim, gradient=_gradient,
        periodic=periodic, reflective=reflective, **kwargs)
