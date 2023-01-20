import numpy as np

from .constants import G

from ..calculus import (
    differentiate, integrate, cumulative_integrate, complement
)

# move this to a gyre utils module
def to_rad_per_sec(freq, freq_units, mass, radius):
    factor = {
        "NONE": np.sqrt(G * mass / radius**3),
        "HZ": 2.0 * np.pi,
        "UHZ": 2e-6 * np.pi,
        "RAD_PER_SEC": 1.0,
        "CYC_PER_DAY": 2.0 * np.pi / 86400.0
    }
    return factor[freq_units] * freq 

def structure_kernels(model):
    """Returns a dict of stellar structural kernels.
    """
    # Constants
    M = model.M
    R = model.R

    # Profile data
    r = model.r             # radial co-ordinate
    m = np.exp(model.ln_q)*M             # mass co-ordinate
    P = model.P             # pressure
    rho = model.rho         # density
    Gamma1 = model.Gamma_1  # first adiabatic index
    c2 = Gamma1*P/rho          # square of the sound speed
    
    u = 1/r
    u[0] = 0.0  # need better solution to this!

    # Pulsation data
    xi_r = model.xi_r.real  # radial component of eigenfunction
    xi_h = model.xi_h.real  # horiz. component of eigenfunction
    
#     omega = 2.*np.pi*pulse["freq"].real*1e-6  # convert to angular frequency
    omega = to_rad_per_sec(model.freq.real, model.freq_units, M, R)

    ell = model["l"]
    L2 = ell * (ell + 1)
    
    drho_dr = differentiate(rho, r)
    dxi_r_dr = differentiate(xi_r, r, axis=0)
    
    chi = dxi_r_dr + 2*xi_r * u - L2*xi_h * u
    
    S = integrate(r**2 * rho * (xi_r**2 + L2 * xi_h**2), r, axis=0)
    
    K_c2_rho = 0.5 * r**2 * rho * c2 * chi**2 / S / omega**2
    
    alpha = rho * (chi + 0.5 * xi_r * drho_dr / rho) * xi_r
    beta = (rho * chi + xi_r * drho_dr)

    K_rho_c2 = (
        - 0.5 * (xi_r**2 + L2*xi_h**2) * rho * omega**2 * r**2
        + 0.5 * rho * c2 * chi**2 * r**2 - G * m * alpha
        - 4 * np.pi * G * rho * r**2 * complement(alpha, r, axis=0, initial=0.0)
        + G * m * rho * xi_r * dxi_r_dr
        + 0.5 * G * (m * drho_dr + 4 * np.pi * rho**2 * r**2) * xi_r**2
        # These last terms have negligable effect! Why?
        - 4 * np.pi * G / (2 * ell + 1) * rho * (
            (ell + 1) * u**ell * (xi_r - ell * xi_h)
            * cumulative_integrate(beta * r**(ell + 2), r, axis=0, initial=0.0)
            - ell * r**(ell + 1) * (xi_r + (ell + 1) * xi_h)
            * complement(beta * r * u**ell, r, axis=0, initial=0.0)
        )
    ) / S / omega**2
    
    # No idea how to verify this next stuff
    K_G1_rho = K_c2_rho

    alpha = rho * u**2 * cumulative_integrate(K_c2_rho / P, r, axis=0, initial=0.0)

    K_rho_G1 = (
        K_rho_c2 - K_c2_rho + G * m * alpha
        + 4 * np.pi * G * rho * r**2 * complement(alpha, r, axis=0, initial=0.0)
    )

    # Next do helium, where we need some Gamma derivatives from EOS
    dG1_Y = model.dGamma_1_Y
    # dG1_rho = model.dGamma_1_rho
    # dG1_P = model.dGamma_1_P
    
    # Where u = p / rho
    K_Y_u = dG1_Y * K_G1_rho
    K_u_Y = None

#     F = (dG1_P + dG1_rho) * K_G1_rho + K_rho_G1
    
#     rho_r = interp1d(r, rho)
#     P_r = interp1d(r, P)

#     def solve_phi(_F):
#         F_r = interp1d(r, _F)

#         def bvp_fun(t, y):
#             v = 1/t**2
#             v[0] = 0.0
#             dy1 = F_r(t) + t**2 * rho_r(t) * y[1]
#             dy2 = - 4 * np.pi * G * rho_r(t) * v**2 / P_r(t) * y[0]
#             return np.vstack([dy1, dy2])

#         sol = solve_bvp(bvp_fun, lambda ya, yb: np.array([ya[0], yb[0]]),
#                         x=r, y=np.zeros((2, len(r))))

#         if not sol.success:
#             raise ValueError("No solution found")

#         return interp1d(sol.x, sol.y[0, :])(r)
    
#     psi = xr.apply_ufunc(solve_phi, F.stack(index=["n_pg", "l"]), input_core_dims=[["index"]])
#     psi = psi.unstack("index")
    
#     K_rho_Y = dG1_P * K_G1_rho - P * differentiate(psi / P, r)
    
    return {
        "c2_rho": (K_c2_rho, K_rho_c2),
        "G1_rho": (K_G1_rho, K_rho_G1),
        "Y_u": (K_Y_u, K_u_Y),
    }
