#!/usr/bin/env python3

# intended to mimic Gaia DR2 HR diagrams (A&A 616, A10, 2018),
# specifically, Fig. 6c because I can't be bothered with this extinction business
# https://www.aanda.org/articles/aa/abs/2018/08/aa32843-18/aa32843-18.html
import os
import numpy as np
import matplotlib.pyplot as plt
import cmcrameri
from matplotlib.colors import PowerNorm
from astropy.table import Table

GAIA_FILENAME = 'data/gaia_HR.npy'
KPLR_FILENAME = 'data/gaia_kplr_1arcsec.fits'

if os.path.exists(GAIA_FILENAME):
    r = np.load(GAIA_FILENAME)
else:
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
    np.save(GAIA_FILENAME, r)

if not os.path.exists(KPLR_FILENAME):
    import requests
    request = requests.get(
        "https://www.dropbox.com/s/xo1n12fxzgzybny/kepler_dr2_1arcsec.fits?dl=1", 
        allow_redirects=True
    )
    with open(KPLR_FILENAME, 'wb') as f:
        f.write(request.content)

df = Table.read(KPLR_FILENAME).to_pandas()
# print(df["parallax"])

# print(list(df.columns))

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

plt.plot(r['bp']-r['rp'], G, 'k.', ms=1, alpha=0.2, rasterized=True, zorder=0)
plt.hist2d(r['bp']-r['rp'], G, cmap='cmc.grayC', bins=200, cmin=10, norm=PowerNorm(gamma=1/3), zorder=1)

# plt.plot(df['bp_rp'], kG, 'k.', ms=1, alpha=0.2, rasterized=True, zorder=2)
plt.hist2d(df['bp_rp'], kG, cmap='cmc.lajolla', bins=200, cmin=10, norm=PowerNorm(gamma=1/3), zorder=3)

plt.gca().invert_yaxis()
plt.xlabel(r'$B_\mathrm{p}-R_\mathrm{p}$')
plt.ylabel(r'$G$')
plt.show()
