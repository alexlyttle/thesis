import h5py
import numpy as np
import xarray as xr
import pandas as pd

from .utils import decode

def clean_array(value):
    if value.dtype.names == ("re", "im"):
        return value["re"] + 1j * value["im"]
    return value

def load_summary(filename, ncol="n_pg", lcol="l"):
    """Load summary file as Dataset"""
    icol = "index"
    # Could prob do all this a better way with Pandas MultiIndex to create
    # A combined index then use Dataset.unstack() to put them into n_pg
    with h5py.File(filename) as file:
        attrs = {k: decode(v) for k, v in file.attrs.items()}
        index = pd.MultiIndex.from_arrays(
            [file[ncol][()], file[lcol][()]],
            names=(ncol, lcol)
        )
        coords = {icol: index}
        dims = (icol,)
        
        data_vars = {
            k: xr.DataArray(clean_array(v[()]), coords, dims) for k, v in file.items()
        }

    ds = xr.Dataset(data_vars, attrs=attrs)
    return ds.unstack(icol)

def load_detail(filename, xcol="x"):
    with h5py.File(filename) as file:
        attrs = {k: decode(v) for k, v in file.attrs.items()}
        dims = (xcol,)
        coords = {xcol: file[xcol][()]}

        data_vars = {
            k: xr.DataArray(clean_array(v[()]), coords, dims) for k, v in file.items()
        }
    return xr.Dataset(data_vars, attrs=attrs)

def load_details(filenames, xcol="x", ncol="n_pg", lcol="l"):
    icol = "index"
    details = []
    for filename in filenames:
        d = load_detail(filename, xcol=xcol)
        d = d.assign_coords({
            icol: pd.MultiIndex.from_tuples(
                [(d.attrs[ncol], d.attrs[lcol])], names=(ncol, lcol)
            )
        })
        details.append(d)

    return xr.concat(details, icol, combine_attrs="drop_conflicts").unstack(icol)

def load_output(summary_filename, detail_filenames, xcol="x", ncol="n_pg", lcol="l"):
    summary = load_summary(summary_filename, ncol=ncol, lcol=lcol)
    details = load_details(detail_filenames, xcol=xcol, ncol=ncol, lcol=lcol)
    return summary.merge(details)
