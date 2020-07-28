# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3.8.3 64-bit
#     language: python
#     name: python38364bit425a2724ae224223ab43e7c3d3663fd5
# ---

# General references:
# - [Learn Astropy](http://learn.astropy.org/)
#     - [FITS File Handling](https://docs.astropy.org/en/stable/io/fits/index.html)
#     - [Astropy Table](https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table)
# - [FITS Standard Page](https://fits.gsfc.nasa.gov/fits_standard.html)

# +
# Import Astropy Modules for file operation and converting to table
from astropy.io import fits
from astropy.table import Table

# Import custom module for pre-defined data path
from grb.config import path
# -

dir(fits)

dir(Table)

# One Way to Load the FITS Data
# =============================

# Open the file
# -------------

filename = path.FITS / "160625/L200408111538F357373F92_PH00.fits"
f = fits.open(filename)

# View File Information
# ---------------------

# + tags=[]
f.info()
# -

# Select Needed Data
# ------------------
# `PRIMARY` contains general information about the GRB.

header = f[0].header
header

# See [Time in *Fermi* Data Analysis](https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_Data/Time_in_ScienceTools.html#:~:text=The%20Fermitools%20use%20mission%20elapsed%20time%20%28MET%29%2C%20the,the%20UTC%20system.%20Time%20Systems%20in%20a%20Nutshell) for help in `MET`.

METREF = header['MJDREFI'] + header['MJDREFF']
METREF

# `EVENTS` contains the (photons or spacecraft) events information

data = f[1].data
data

# Load Data to Astropy Table
# --------------------------

t = Table(data)
t

# Close The Opened File
# ---------------------
# Always remember to close the opened file.

f.close()

# An Alternative Way to Open A File
# =================================
# [`with` statement](https://docs.python.org/3/reference/compound_stmts.html#with) automatically handles open and close of files.

# + tags=[]
with fits.open(filename) as f:
    f.info()
    data2 = f[1].data
# -

# Check if the two ways get the same data

import numpy as np

np.all(data==data2)

# An Alternative Way to Load FITS data
# ====================================

t2 = Table.read(filename, hdu=1)

t2

# Check if the two ways get the same data.

np.all(t==t2)

# Convert Astropy Table to Pandas DataFrame
# =========================================
# Sometimes it would be better to deal with Pandas DataFrame instead of Astropy Table.
#
# **Note**: Some information, for example, units, will get lost, after the conversion.

t3 = t[['ENERGY', 'TIME']]
df = t3.to_pandas()
df

# + tags=[]
print(f"{type(df) = },\n{type(t3) = }")
# -

# Application to GBM TTE FITS Files
# =================================

# + tags=[]
ttefile = path.FITS / "160625/TTE/glg_tte_b0_bn160625945_v00.fit"
with fits.open(ttefile) as f:
    f.info()
    header = f[0].header
header
# -

# Trigger time relative to MJDREF (in second)

TRIGTIME = header['TRIGTIME']
TRIGTIME

# +
from grb.lat.timeutils import UTCMET

trigtime = UTCMET.met2utc(TRIGTIME)
trigtime
# -

trigtime.value

tobs = header['DATE-OBS']
tobs_met = UTCMET.utc2met(tobs)
tobs_met

tobs_met.value

dt = trigtime - tobs_met
dt

dt.value


