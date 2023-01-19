from tomso import mesa, gyre
from .utils import tomso_to_dataset

def load_dataset(filename, kind="profile"):
    if kind == "profile":
        log = mesa.load_profile(filename)
        dim = "zone"
    elif kind == "gyre":
        log = gyre.load_gyre(filename)
        dim = "k"
    else:
        raise NotImplementedError(f"Kind '{kind}' not implemented.")
    return tomso_to_dataset(log, dim=dim)
