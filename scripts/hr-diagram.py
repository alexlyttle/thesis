#!/usr/bin/env python3

# intended to mimic Gaia DR2 HR diagrams (A&A 616, A10, 2018),
# specifically, Fig. 6c because I can't be bothered with this extinction business
# https://www.aanda.org/articles/aa/abs/2018/08/aa32843-18/aa32843-18.html
import os, requests
import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from cmcrameri import cm
from matplotlib.colors import PowerNorm
from astropy.table import Table
from isochrones.mist import MIST_EvolutionTrack
from astroquery.vizier import Vizier
# from scipy.interpolate import LinearNDInterpolator

GAIA_FILENAME = 'data/gaia_200pc.npy'
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

if not os.path.exists(KPLR_FILENAME):
    dirname = os.path.dirname(KPLR_FILENAME)
    if not os.path.exists(dirname):
        print(f"Making directory '{dirname}'.")
        os.makedirs(dirname)
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

print("Query Serenelli et al. (2017)")
tlist = Vizier(
    catalog=["J/ApJS/233/23/table3","J/MNRAS/452/2127/table3","J/ApJ/835/173/table3"], 
    columns=["KIC"], 
    row_limit=-1
).query_constraints()
s17, d16, l17 = [t["KIC"].value.data for t in tlist]

G = r['g'] + 5 * np.log10(r['parallax']) - 10
kG = df["phot_g_mean_mag"] + 5 * np.log10(df['parallax']) - 10

tracks = MIST_EvolutionTrack()
num_points = 201
eep = np.linspace(202., 605., num_points)
params = [
    eep,                        # EEP
    np.zeros(num_points),       # [Fe/H]
    10.0*np.ones(num_points),   # distance (pc)
    np.zeros(num_points)        # Av (mag)
]
bands = ["G", "BP", "RP"]

print("Making plot.")
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot()

cmap = cm.grayC_r
ax.plot(r['bp']-r['rp'], G, '.', c=cmap.colors[0], ms=1, alpha=0.2, rasterized=True, zorder=0)
ax.hist2d(r['bp']-r['rp'], G, cmap=cmap, bins=200, cmin=10, norm=PowerNorm(gamma=1/3), rasterized=True, zorder=1)

cmap = cm.devon
ax.plot(df['bp_rp'], kG, '.', c=cmap.colors[0], ms=1, alpha=0.2, rasterized=True, zorder=1)
ax.hist2d(df['bp_rp'], kG, cmap=cmap, bins=200, cmin=10, norm=PowerNorm(gamma=1/3), rasterized=True, zorder=2)

# mask = df["kepid"].isin(s17) & ~df["kepid"].isin(d16) & ~df["kepid"].isin(l17)
# ax.plot(df.loc[mask, 'bp_rp'], kG.loc[mask], "o", c="k", alpha=0.5, markerfacecolor="none")
# mask = df["kepid"].isin(d16)
# ax.plot(df.loc[mask, 'bp_rp'], kG.loc[mask], "^", c="C1", markerfacecolor="none")
# mask = df["kepid"].isin(l17)
# ax.plot(df.loc[mask, 'bp_rp'], kG.loc[mask], "s", c="C2", markerfacecolor="none")

# Instead, draw a inset plot and do tracks here
ax.set_xlabel(r'$\mathrm{G_{BP}}-\mathrm{G_{RP}}$')
ax.set_ylabel(r'$M_\mathrm{G}$')
ax.invert_yaxis()

axins = ax.inset_axes([0.45, 0.45, 0.53, 0.53])

cmap = cm.grayC_r
axins.plot(r['bp']-r['rp'], G, '.', c=cmap.colors[0], ms=1, alpha=0.2, rasterized=True, zorder=0)
axins.hist2d(r['bp']-r['rp'], G, cmap=cmap, bins=200, cmin=10, norm=PowerNorm(gamma=1/3), rasterized=True, zorder=1)
cmap = cm.devon
axins.plot(df['bp_rp'], kG, '.', c=cmap.colors[0], ms=1, alpha=0.2, rasterized=True, zorder=0)
axins.hist2d(df['bp_rp'], kG, cmap=cmap, bins=200, cmin=10, norm=PowerNorm(gamma=1/3), rasterized=True, zorder=1)
for mass in [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4]:
    _, _, _, mags = tracks.interp_mag([mass] + params, bands)
    axins.plot(mags[:, 1]-mags[:, 2], mags[:, 0], "-", color="k")

mask = df["kepid"].isin(s17) & ~df["kepid"].isin(d16) & ~df["kepid"].isin(l17)
ls, = axins.plot(df.loc[mask, 'bp_rp'], kG.loc[mask], "o", ms=4, c="k", markerfacecolor="none", label="Serenelli et al. (2017)")

c = cm.buda.resampled(2).colors
mask = df["kepid"].isin(l17)
ll, = axins.plot(df.loc[mask, 'bp_rp'], kG.loc[mask], "s", ms=4, c=c[0], markerfacecolor="none", label="Lund et al. (2017)")
mask = df["kepid"].isin(d16)
ld, = axins.plot(df.loc[mask, 'bp_rp'], kG.loc[mask], "^", ms=4, c=c[1], markerfacecolor="none", label="Davies et al. (2016)")

# SUN
axins.scatter(0.772, 4.542, marker="o", s=64, color=c[1], zorder=4, facecolor="none")
axins.scatter(0.772, 4.542, marker=".", s=16, color=c[1], zorder=4)

axins.set_xlim(0.4, 1.4)
axins.set_ylim(1.5, 6.0)
axins.set_xticks([])
axins.set_yticks([])

ax.indicate_inset_zoom(axins, edgecolor="black")
axins.invert_yaxis()

c = cmap.colors[len(cmap.colors)//2]
ax.legend(handles=[ld, ll, ls], loc="lower right", facecolor=c, edgecolor=c)
plt.show()
print("Saving plot.")
fig.tight_layout()
fig.savefig("../figures/hr-diagram.pdf", format="pdf", dpi=300)

print("Done.")
