import numpy as np
import xarray as xr

from scipy.integrate import trapezoid, cumulative_trapezoid

from .utils import is_data_array

def differentiate(y, *varargs, axis=None, edge_order=1):
    """Returns the gradient of y given varargs using np.gradient.
    
    Args:
        y (array_like): Values of some function to be differentiated.
        varargs (list of scalar or array, optional):
        edge_order (int{1, 2}, optional):
        axis (None or int or tuple of ints, optional):
    
    Returns:
        np.ndarray or xr.DataArray:
    """
    z = np.gradient(y, *varargs, axis=axis, edge_order=edge_order)    
    if is_data_array(y):
        return xr.DataArray(z, dims=y.dims)
    return z

def integrate(y, x=None, dx=1.0, axis=-1):
    """Integrates y using the trapezium rule.
    
    Args:
        y (array_like): Values of some function to be integrated.
        x (None or array_like, optional): Values over which to integrate y.
        dx (float or array_like, optional): Spacing between points in y. 
        axis (int, optional): Axis over which to integrate.
    
    Returns:
        np.ndarray or xr.DataArray:
    """
    z = trapezoid(y, x=x, dx=dx, axis=axis)
    if is_data_array(y): 
        dims = tuple(d for d in y.dims if d != y.dims[axis])
        return xr.DataArray(z, dims=dims)
    return z
    
def cumulative_integrate(y, x=None, dx=1.0, axis=-1, initial=None):
    """Cumulatively integrates y using the trapezium rule.
    
    Args:
        y (array_like): Values of some function to be integrated.
        x (None or array_like, optional): Values over which to integrate y.
        dx (float or array_like, optional): Spacing between points in y. 
        axis (int, optional): Axis over which to integrate.
        initial (None or float, optional): Value to append to the start of the
            result.
    
    Returns:
        np.ndarray or xr.DataArray:
    """
    z = cumulative_trapezoid(y, x=x, dx=dx, axis=axis, initial=initial)
    if is_data_array(y):
        return xr.DataArray(z, dims=y.dims)
    return z

def complement(y, x=None, dx=1.0, axis=-1, initial=None):
    """Returns the integral of y minus the cumulative integral of y.
    
    Args:
        y (array_like): Values of some function to be integrated.
        x (None or array_like, optional): Values over which to integrate y.
        dx (float or array_like, optional): Spacing between points in y. 
        axis (int, optional): Axis over which to integrate.
        initial (None or float, optional): Value to append to the start of the
            result.
    
    Returns:
        np.ndarray or xr.DataArray:
    """
    return (
        integrate(y, x=x, dx=dx, axis=axis)
        - cumulative_integrate(y, x=x, dx=dx, axis=axis, initial=initial)
    )
