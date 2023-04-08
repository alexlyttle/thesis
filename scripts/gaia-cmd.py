#!/usr/bin/env python3

# intended to mimic Gaia DR2 HR diagrams (A&A 616, A10, 2018),
# specifically, Fig. 6c because I can't be bothered with this extinction business
# https://www.aanda.org/articles/aa/abs/2018/08/aa32843-18/aa32843-18.html
import os
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from cmcrameri import cm
from matplotlib.colors import PowerNorm
# from astropy.table import Table
# from isochrones.mist import MIST_EvolutionTrack
# from astroquery.vizier import Vizier
# from scipy.interpolate import LinearNDInterpolator

GAIA_FILENAME = 'data/gaia_200pc.npy'

if os.path.exists(GAIA_FILENAME):
    print("Reading Gaia data.")
    r = np.load(GAIA_FILENAME)
else:
    dirname = os.path.dirname(GAIA_FILENAME)
    if not os.path.exists(dirname):
        print(f"Making directory '{dirname}'.")
        os.makedirs(dirname)
    print("Downloading Gaia data.")
    from astroquery.gaia import Gaia
    job = Gaia.launch_job_async("""
SELECT
phot_g_mean_mag AS g,
parallax,
e_bp_min_rp_val AS e_bp_rp,
phot_bp_mean_mag AS bp,
phot_rp_mean_mag AS rp
FROM gaiadr2.gaia_source
WHERE parallax_over_error > 10
AND parallax > 5
AND phot_g_mean_flux_over_error > 50
AND phot_rp_mean_flux_over_error > 20
AND phot_bp_mean_flux_over_error > 20
AND phot_bp_rp_excess_factor < 1.3+0.06*power(phot_bp_mean_mag-phot_rp_mean_mag,2)
AND phot_bp_rp_excess_factor > 1.0+0.015*power(phot_bp_mean_mag-phot_rp_mean_mag,2)
AND visibility_periods_used > 8
AND astrometric_chi2_al/(astrometric_n_good_obs_al-5)<1.44*greatest(1,exp(-0.4*(phot_g_mean_mag-19.5)))""")
    r = job.get_results().as_array().data
    print("Saving Gaia data.")
    np.save(GAIA_FILENAME, r)

G = r['g'] + 5 * np.log10(r['parallax']) - 10

# tracks = MIST_EvolutionTrack()
# num_points = 201
# eep = np.linspace(202., 808., num_points)
# params = [
#     eep,                        # EEP
#     np.zeros(num_points),       # [Fe/H]
#     10.0*np.ones(num_points),   # distance (pc)
#     np.zeros(num_points)        # Av (mag)
# ]
# bands = ["G", "BP", "RP"]

print("Making plot.")
cmap = cm.oslo

# set background color and theme to match colormap
facecolor = cmap.colors[0]
brightness = np.sum(
    np.array([0.2126, 0.7152, 0.0722]) * facecolor
)
if brightness < 0.5:
    plt.style.use("dark_background")

fig = plt.figure(figsize=(6, 6), dpi=150, facecolor=cmap.colors[0], tight_layout=True)
ax = fig.add_subplot()

ax.hist2d(r['bp']-r['rp'], G, cmap=cmap, bins=200, norm=PowerNorm(gamma=1/3), rasterized=True, zorder=0)

# for mass in np.logspace(-1, 2, 9):
#     _, _, _, mags = tracks.interp_mag([mass] + params, bands)
#     ax.plot(mags[:, 1]-mags[:, 2], mags[:, 0], "-", color="k", zorder=1)

# SUN
ax.scatter(0.772, 4.542, marker="o", s=64, color="k", zorder=2, facecolor="none")
ax.scatter(0.772, 4.542, marker=".", s=16, color="k", zorder=2)

ax.set_xlabel(r'$\mathrm{G_{BP}}-\mathrm{G_{RP}}$')
ax.set_ylabel(r'$M_\mathrm{G}$')
ax.invert_yaxis()

plt.show()

# print("Saving plot.")
# fig.tight_layout()
# fig.savefig("../figures/hr-diagram.pdf", format="pdf", dpi=300)

print("Done.")
