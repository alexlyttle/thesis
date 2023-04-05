#!/usr/bin/env python3

# intended to mimic Gaia DR2 HR diagrams (A&A 616, A10, 2018),
# specifically, Fig. 6c because I can't be bothered with this extinction business
# https://www.aanda.org/articles/aa/abs/2018/08/aa32843-18/aa32843-18.html
import os, requests
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import cmcrameri
from matplotlib.colors import PowerNorm
from astropy.table import Table
# from scipy.interpolate import LinearNDInterpolator

GAIA_FILENAME = 'data/gaia_HR.npy'
KPLR_FILENAME = 'data/gaia_kplr_1arcsec.fits'
# BC_FILENAME = "data/bc.csv"
# GRID_FILENAME = "/var/local/Scratch/shared/data/mesa_grids/grid2p5a/grid.h5"

# print("Reading grid.")
# grid = []
# for mass in np.arange(0.8, 1.24, 0.04):
#     df = pd.read_hdf(GRID_FILENAME, key=f"m{mass:.2f}")
#     grid.append(df)
# grid = pd.concat(grid, ignore_index=True).reset_index(drop=True)
# grid = grid.loc[np.isclose(grid.initial_feh, 0.0) & np.isclose(grid.initial_Yinit, 0.28) & np.isclose(grid.initial_MLT, 1.9)]

# if os.path.exists(BC_FILENAME):
#     bc = pd.read_csv(BC_FILENAME)
# else:
#     print("Downloading BCs.")
#     import io, tarfile
#     r = requests.get("https://waps.cfa.harvard.edu/MIST/BC_tables/UBVRIplus.txz")
#     fileobj = io.BytesIO(r.content)
#     tf = tarfile.open(fileobj=fileobj)
#     member = tf.getmember("fehp000.UBVRIplus")
#     file = tf.extractfile(member)
#     for _ in range(6):
#         header = file.readline()
#     names = header.decode("utf-8").lstrip("#").strip().split()
#     bc = pd.read_table(file, names=names, delimiter="\s+", on_bad_lines="skip")
#     bc = bc.loc[np.isclose(bc["Av"], 0.0)]
#     bc.to_csv(BC_FILENAME, index=False)

# bc_func = LinearNDInterpolator(bc[["Teff", "logg"]], bc[["Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"]])

if os.path.exists(GAIA_FILENAME):
    print("Reading Gaia data.")
    r = np.load(GAIA_FILENAME)
else:
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

if not os.path.exists(KPLR_FILENAME):
    print("Downloading Kepler-Gaia cross match.")
    request = requests.get(
        "https://www.dropbox.com/s/xo1n12fxzgzybny/kepler_dr2_1arcsec.fits?dl=1", 
        allow_redirects=True
    )
    with open(KPLR_FILENAME, 'wb') as f:
        f.write(request.content)

print("Reading Kepler-Gaia cross match.")
df = Table.read(KPLR_FILENAME).to_pandas()
df = df.sort_values("kepler_gaia_ang_dist")
df = df.drop_duplicates("kepid", keep="first")
df = df.dropna(subset=["parallax", "phot_g_mean_mag", "bp_rp"])
df = df.loc[
    (df["parallax"] > 1e-3) & (df["parallax_over_error"] > 10) \
    & (df["phot_g_mean_flux_over_error"] > 50) \
    & (df["phot_rp_mean_flux_over_error"] > 20) \
    & (df["phot_bp_mean_flux_over_error"] > 20)
]

G = r['g'] + 5 * np.log10(r['parallax']) - 10
kG = df["phot_g_mean_mag"] + 5 * np.log10(df['parallax']) - 10

print("Making plot.")
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()

ax.plot(r['bp']-r['rp'], G, 'k.', ms=1, alpha=0.2, rasterized=True, zorder=0)
ax.hist2d(r['bp']-r['rp'], G, cmap='cmc.grayC', bins=200, cmin=10, norm=PowerNorm(gamma=1/3))

# ax.plot(df['bp_rp'], kG, 'k.', ms=1, alpha=0.2, rasterized=True, zorder=2)
ax.hist2d(df['bp_rp'], kG, cmap='cmc.lajolla', bins=200, cmin=10, norm=PowerNorm(gamma=1/3))

# for mass, group in grid.groupby("dirname"):
#     mags = bc_func(group[["effective_T", "log_g"]])
#     ax.plot(mags[:, 1]-mags[:, 2], mags[:, 0], "k--")

ax.invert_yaxis()
ax.set_xlabel(r'$B_\mathrm{p}-R_\mathrm{p}$')
ax.set_ylabel(r'$G$')

print("Saving plot.")
fig.tight_layout()
fig.savefig("../figures/hr-diagram.pdf", format="pdf", dpi=300)

print("Done.")
