import numpy as np
import jax.numpy as jnp
import xarray as xr
import datatree as dt

from collections import namedtuple
from dynesty.utils import Results as _Results
from jax.tree_util import tree_leaves, tree_map
from jax import random


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
    #  'bound',
     'bound_iter',
    #  'samples_bound',
     'scale'],
)

def importance_weights(results: NestedResults) -> jnp.ndarray:
    weights = jnp.exp(results.logwt - results.logz[-1])
    return weights / weights.sum()

def resample(key, results: NestedResults, shape=None):
    """Resample results such that they are equally weighted."""
    weights = importance_weights(results)
    if shape is None:
        shape = weights.shape  # same shape as results
    return tree_map(
        lambda x: random.choice(key, x, shape=shape, p=weights),
        results.samples
    )

def to_datatree(results: NestedResults) -> dt.DataTree:
    num_samples = results.logz.shape[0]
    coords = {"index": np.arange(num_samples)}
    dims = "index"

    samples_dict = {}
    uniform_dict = {}

    for k in results.samples.keys():
        samples_dict[k] = xr.DataArray(results.samples[k], coords=coords, dims=dims)
        uniform_dict[k] = xr.DataArray(results.samples_u[k], coords=coords, dims=dims)    

    stats_dict = {}
    results_dict = results._asdict()
    for k in ["logl", "logwt", "logvol", "logz", "logzerr", 
            "ncall", "blob", "samples_id", "samples_it", 
            "information", "bound_iter", "scale"]:
        stats_dict[k] = xr.DataArray(results_dict.pop(k), dims="index")

    stats_attrs = {
        "nlive": results.nlive,
        "niter": results.niter,
        "eff": results.eff,
    }   

    ds = {
        "samples": xr.Dataset(samples_dict),
        "uniform_samples": xr.Dataset(uniform_dict),
        "sampler_stats": xr.Dataset(stats_dict, attrs=stats_attrs)
    }
    return dt.DataTree.from_dict(ds)

def save_results(results: NestedResults, filepath) -> None:
    data_tree = to_datatree(results)
    return data_tree.to_netcdf(filepath, engine="h5netcdf")

def from_datatree(results: dt.DataTree) -> NestedResults:
    results_dict = {}
    results_dict["samples"] = {
        k: a.values for k, a in results["/samples"].data_vars.items()
    }
    results_dict["samples_u"] = {
        k: a.values for k, a in results["/uniform_samples"].data_vars.items()
    }
    results_dict.update({
        k: a.values for k, a in results["/sampler_stats"].data_vars.items()
    })
    results_dict.update(results["/sampler_stats"].attrs)
    return NestedResults(**results_dict)

def load_results(filepath) -> NestedResults:
    results = dt.open_datatree(filepath, engine="h5netcdf")
    return from_datatree(results)

def to_dynesty(results: NestedResults) -> _Results:
    results_dict = results._asdict()
    results_dict["samples"] = np.stack(tree_leaves(results.samples), -1)
    results_dict["samples_u"] = np.stack(tree_leaves(results.samples_u), -1)
    return _Results(list(results_dict.items()))
