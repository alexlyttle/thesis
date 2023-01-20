import xarray as xr

def is_data_array(x):
    return isinstance(x, xr.DataArray)
