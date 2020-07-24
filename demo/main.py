# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3.8.3 64-bit
#     language: python
#     name: python38364bit425a2724ae224223ab43e7c3d3663fd5
# ---

# %% [markdown]
# # The Main Demo Script
# import
# ======
# public packages
# ---------------

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# custom packages
# ---------------

# %%
from grb.config import path
from grb.lat.analysis import PHTable

# %% [markdown]
# Load Information
# ================
# Load basic information
# ----------------------
# The folling table is supposed to be already prepared

# %%
grbs = pd.read_hdf(path.TABLE / "query-results-latest.h5", key='table')
grbs

# %% [markdown]
# \begin{equation}
# \kappa = s \frac{E_\text{high} - E_\text{low}}{H_0} \frac{1}{(1+z)} \int_0^z \frac{(1 + z') \mathrm{d} z'}{\sqrt{\Omega_m (1 + z')^3 + \Omega_\Lambda}}
# \end{equation}
# is the Lorentz violaaion factor. $z$ is the redshift of the GRB source. $E_\text{low}$ is lower than 260 keV and can be omitted. $H_0 = 67.3 \pm 1.2 kms^{-1} Mpc^{-1}$ is the Hubble expansion rate, and $[\Omega_m, \Omega_\Lambda] = [0.315^{+0.016}_{-0.017}, 0.685^{+0.017}_{-0.016}]$ are cosmological constants.
#
# The intrinsic time lag at the GRB source between two photons is
# \begin{equation}
# \Delta t_\text{in} = \frac{\Delta t_\text{obs}}{1 + z} - \frac{\kappa}{E_\text{LV}}
# \end{equation}
# $s = \pm 1$ indicates whether the high energy photon travels slower $(s = +1)$ or faster $(s = −1)$ than the low energy photon.
#
# Read data from FITS data files
# ------------------------------

# %%
indir = path.FITS
ph = PHTable(grbs, indir)

# %% [markdown]
# Plotting
# ========
# Assign output directoryAssign output directory

# %%
outdir = path.IMAGE

# %% [markdown]
# Assign plot style

# %%
plt.style.use(['seaborn-darkgrid', 'seaborn-talk'])

# %% [markdown]
# Histogram of Data Set I
# -----------------------
# This will also calculate the power law for Data Set I, and will be applied to other Data Sets.

# %% tags=[]
# predefined binmethod: ‘auto’, ‘fd’, ‘doane’, ‘scott’, ‘stone’, ‘rice’, ‘sturges’, ‘sqrt’
binmethod = np.logspace(0, 1, num=24)
figbin, popt, pconv = ph.regPlot(binmethod)
popt, pconv
plt.ylabel(r'counts $/\Delta E$ (1/GeV)')
plt.legend()
plt.tight_layout()

# plt.savefig(outdir / 'data-set-I-counts-bin.pdf')

# %% [markdown]
# Set Data Set IV' (classIV_less)
# -------------------------------

# %%
ph.classIV

# %%
ph.classIV_less.remove_rows([1, 3])
ph.classIV_less

# %% [markdown]
# Scatter plots for all Data Sets
# -------------------------------

# %%
figs = ph.scatterPlots()

# %% [markdown]
# **Note**: As a demo, we assign `repeat` to 100 only below. However, in the reference and in the thesis, `repeat` was 100000.

# %%
repeat = 100    # number of random sets for each Data Set
rho = 5         # \rho in the test function

# %% [markdown]
# Array of test function values for each test E_LV of each Data Set
# -----------------------------------------------------------------
# This will also choose the preferred value of $E_\text{LV}$

# %%
T1 = ph.T(ph.classI, rho=rho)[0]
T2 = ph.T(ph.classII, rho=rho)[0]
T3 = ph.T(ph.classIII, rho=rho)[0]
T4 = ph.T(ph.classIV, rho=rho)[0]
T5 = ph.T(ph.classIV_less, rho=rho)[0]

# %%
i = T4.argmax()
E_LV = ph.E_LVs[i]
E_LV, 1/E_LV

# %% [markdown]
# $T$-$E_\text{LV}$ plot with $n$-$\sigma$ regions
# ------------------------------------------------

# %%
res2 = ph.randPlotII(repeat=repeat, rho=rho, return_extr=True)

# %%
res3 = ph.randPlotIII(repeat=repeat, rho=rho, return_extr=True)

# %%
res4 = ph.randPlotIV(repeat=repeat, rho=rho, return_extr=True)

# %%
label = "Data Set IV'"
res4_less = ph.randPlotIV(repeat=repeat, rho=rho, less=True, return_extr=True, label=label)

# %%
res1 = ph.randPlotI(repeat=repeat, rho=rho, return_extr=True)

# %% [markdown]
# Save plots
# ----------
# Uncomment to save the above plots.

# %%
# figname = ['data-set-' + i + '-ET.pdf' for i in ['I', 'II', 'III', 'IV', "IV_less"]]
# res = [res1[0], res2[0], res3[0], res4[0], res4_less[0]]
# ressave = lambda n: res[n].savefig(outdir / figname[n], facecolor=res[n].get_facecolor())

# for i in range(5):
#     ressave(i)

# %%
# scattername = ['data-set-' + i + '-kt.pdf' for i in ['I', 'II', 'III', 'IV', 'IV_less']]
# figsave = lambda n: figs[n].savefig(outdir / scattername[n], facecolor=figs[n].get_facecolor())

# for i in range(5):
#     figsave(i)

# %% [markdown]
# Other plots
# ------------
# ### Scatter plots with fit line

# %% tags=[]
from scipy.optimize import curve_fit

f = lambda x, b: 1/E_LV.value * x + b
x = ph.classIV_less['kappa']
y = ph.classIV_less['DTtsf']
popt, pcov = curve_fit(f, x, y)
b = popt[0]
fig = plt.subplot()

plt.scatter(x, y, label="Data Set IV'", alpha=0.75)
xlim = np.array(plt.xlim())
plt.plot(xlim, f(xlim, *popt), '--', label=r'fit line: ${:.4e}x {:.4f}$'.format(1/E_LV.value, b))
plt.xlabel(r'$\kappa$ (s $\cdot$ GeV)')
plt.ylabel(r'$\Delta \tau_z$ (s)')
plt.legend()
plt.tight_layout()

print(r'$\sigma = {}$'.format(np.sqrt(np.diag(pcov))))
# plt.savefig(outdir / 'data-set-IV_less-kt.pdf')

# %% [markdown]
# ### $T$-$E_\text{LV}$ for all five Data Sets

# %%
labels = ['Data Set ' + i for i in ['I', 'II', 'III', 'IV', "IV'"]]
plt.plot(ph.E_LVs, T1, '--')
plt.plot(ph.E_LVs, T2, '--')
plt.plot(ph.E_LVs, T3)
plt.plot(ph.E_LVs, T4)
plt.plot(ph.E_LVs, T5)
ax = plt.gca().get_ylim()
plt.vlines(E_LV.value, ax[0], ax[1], alpha=0.25, linestyles='dashdot')
plt.xscale('log')
plt.xlabel(r'$E_\mathrm{LV}$ (GeV)')
plt.ylabel(r'$T$')
plt.legend(labels)
plt.show()

# plt.tight_layout()
# plt.savefig(outdir / 'TE-curves.pdf')

# %% [markdown]
# \begin{align}
# \mathcal{T}_\rho(E_\text{LV}) = \frac{\sum_{i=1}^{N-\rho} \log \left[\bar{t}_\rho / \left(t_{i+\rho} - t_i \right) \right]}{N - \rho}, \\
# \bar{t}_\rho = \frac{\sum_{i=1}^{N-\rho} \left(t_{i+\rho} - t_i \right)}{N - \rho}.
# \end{align}
# $\rho = 5$

# %% [markdown]
# https://en.wikipedia.org/wiki/Standard_deviation#Rules_for_normally_distributed_data
#
# Confidence | Proportion within | Proportion | without
# -- | -- | -- | --
# interval | Percentage |	Percentage |	Fraction
# 1σ |	68.2689492% |	31.7310508% |	1 / 3.1514872
# 3σ |	99.7300204% |	0.2699796% |	1 / 370.398
# 5σ |	99.9999426697% |	0.0000573303% |	1 / 1744278
