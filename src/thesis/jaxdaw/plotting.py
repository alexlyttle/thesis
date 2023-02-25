# import matplotlib.pyplot as plt
import dynesty.plotting as dyplot
from .sampler import NestedResults
from .results import to_dynesty
from jax.tree_util import tree_leaves

def _flatten_dyplot_kwargs(kwargs: dict):
    """Flatten PyTree kwargs for dyplot."""
    if "labels" in kwargs:
        kwargs["labels"] = tree_leaves(kwargs["labels"])

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

def traceplot(results: NestedResults, **kwargs):
    """Wrapper for dynesty.plotting.traceplot"""
    dynesty_results = to_dynesty(results)
    return dyplot.traceplot(
        dynesty_results, 
        **_flatten_dyplot_kwargs(kwargs)
    )

def cornerplot(results: NestedResults, **kwargs):
    """Wrapper for dynesty.plotting.cornerplot"""
    dynesty_results = to_dynesty(results)
    return dyplot.cornerplot(
        dynesty_results, 
        **_flatten_dyplot_kwargs(kwargs)
    )
