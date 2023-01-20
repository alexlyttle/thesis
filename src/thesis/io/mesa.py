from tomso import mesa
from .utils import tomso_to_dataset

def load_mesa(filename, kind="profile"):
    if kind == "profile":
        log = mesa.load_profile(filename)
        dim = "zone"
    else:
        raise NotImplementedError(f"Kind '{kind}' not implemented.")
    return tomso_to_dataset(log, dim=dim)
