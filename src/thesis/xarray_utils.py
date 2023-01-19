import numpy as np
import xarray as xr
from scipy.integrate import trapezoid, cumulative_trapezoid

def gradient(y, *varargs, axis=None, edge_order=1):
    z = np.gradient(y, *varargs, axis=axis, edge_order=edge_order)    
    return xr.DataArray(z, dims=y.dims)

def integrate(y, x=None, dx=1.0, axis=-1):
    z = trapezoid(y, x=x, dx=dx, axis=axis)
    dims = tuple(d for d in y.dims if d != y.dims[axis])
    return xr.DataArray(z, dims=dims)
    
def cumulative_integrate(y, x=None, dx=1.0, axis=-1, initial=None):
    z = cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial)
    return xr.DataArray(z, dims=y.dims)

def complement(y, x=None, dx=1.0, axis=-1, initial=None):
    return integrate(y, x=x, dx=dx, axis=axis) - cumulative_integrate(y, x=x, dx=dx, axis=axis, initial=initial)
