import xarray as xr

from tomso import fgong as _fgong

GLOBAL_KEYS = [
    "M",
    "R",
    "L",
    "initial_Z",
    "initial_X",
    "alpha_MLT",
    "phi_MLT",
    "xi_MLT",
    "beta_surf",
    "lambda_surf",
    "d2lnPc_dlnr2",
    "d2lnrhoc_dlnr2",
    "star_age",
    "Teff",
    "G"
]

VARIABLE_KEYS = [
    "r",  # from surface to centre
    "ln_q",
    "T",
    "P",
    "rho",
    "X",
    "L_r",
    "kappa",
    "eps",
    "Gamma_1",
    "nabla_ad",
    "delta",
    "cp",
    "inverse_mu_e",
    "AA",
    "nuc_X_rate",
    "Z",
    "R_minus_r",  # from centre to surface
    "eps_g",
    "L_g",
    "X_He3",
    "X_C12",
    "X_C13",
    "X_N14",
    "X_O16",
    "dGamma_1_rho",
    "dGamma_1_P",
    "dGamma_1_Y",
    "X_H2",
    "X_He4",
    "X_Li7",
    "X_Be7",
    "X_N15",
    "X_O17",
    "X_O18",
    "X_Ne20",
]

def load_fgong(filename):
    gong = _fgong.load_fgong(filename)
    attrs = {k: gong.glob[i] for i, k in enumerate(GLOBAL_KEYS)}
    
    r = gong.var[::-1, 0]
    coords = {"x": r/attrs["R"]}
    dims = ("x",)
    data_vars = {}
    for i, k in enumerate(VARIABLE_KEYS):
        data_vars[k] = xr.DataArray(gong.var[::-1, i], coords, dims)
    
    return xr.Dataset(data_vars, attrs=attrs)
