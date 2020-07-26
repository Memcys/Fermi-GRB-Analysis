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
#     name: python38364bitb790c22c1a684c1eac40ecab49941293
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

f.info()

# Select Needed Data
# ------------------
# `EVENTS` contains the (photons or spacecraft) events information

data = f[1].data
data

# Load Data to Astropy Table
# --------------------------

t = Table(data)
t

# Close Any Opened File
# ---------------------
# Always remember to close the opened file.

f.close()

# An Alternative Way to Open A File
# =================================
# [`with` statement](https://docs.python.org/3/reference/compound_stmts.html#with) automatically handles open and close of files.

with fits.open(filename) as f:
    f.info()
    data2 = f[1].data

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

print(f"{type(df) = },\n{type(t3) = }")


