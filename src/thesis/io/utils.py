import xarray as xr

def structured_array_to_attrs(array):
    return {key: array[key] for key in array.dtype.names}

def structured_array_to_vars(array, coords=None, dims=None):
    data_vars = {}
    for key in array.dtype.names:
        data_vars[key] = xr.DataArray(array[key], coords=coords, dims=dims)
    return data_vars

def tomso_to_dataset(tomso_log, dim=None):
    attrs = structured_array_to_attrs(tomso_log.header)
    coords, dims = (None, None)
    if dim is not None:
        coords = {dim: tomso_log.data[dim]}
        dims = (dim,)
    data_vars = structured_array_to_vars(tomso_log.data, coords=coords, dims=dims)
    return xr.Dataset(data_vars, attrs=attrs)
