# ---
# jupyter:
#   jupytext:
#     formats: py:light,ipynb
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Reproduce plots in XHW2018
# # Import
# Scientific Library

import pandas as pd

# pkg modules

from grb.config import path
from grb.lat.query import Query

# Logging

import logging
logging.basicConfig(level=logging.WARNING)

fits_dir = path.FITS
table_dir = path.TABLE

fits_dir, table_dir

dir(path)

dir(Query)

# ## Load data
# ### Load data in Table 1

# +
data_csv = table_dir / 'GRB_Tb1.csv'
data1 = pd.read_csv(data_csv, converters={0: str, 3: str})
name = pd.concat([data1.iloc[:, 0], data1.iloc[:, 3]], ignore_index=True)
z = pd.concat([data1.iloc[:, 1], data1.iloc[:, 4]], ignore_index=True)
t = pd.concat([data1.iloc[:, 2], data1.iloc[:, 5]], ignore_index=True)

table1 = pd.DataFrame({
        "GRB": name,
        "z": z,
        "tpeak": t
        })
table1 = table1.dropna()
# -

table1

# +
trigtime_csv = table_dir / 'trigtime-lat-gbm-met.csv'
trigtime_df = pd.read_csv(trigtime_csv)

trigtime_df['urls'] = trigtime_df['urls'].astype('str', copy=False)
trigtime_df['fits'] = trigtime_df['fits'].astype('str', copy=False)
trigtime_df['info'] = trigtime_df['info'].astype('str', copy=False)
trigtime_df
# -

trigtime_df.dtypes

# The relation between observed energy $E_\mathrm{obs}$, "intrinsic energy" $E_\mathrm{in}$ and redshift $z$ might be
# \begin{equation}
# E_\mathrm{in} = (1 + z) E_\mathrm{obs}
# \end{equation}
# The highest redshift of GRBs is 8.2 (as of 2013, N.. (2013). The highest redshift gamma-ray bursts.)


# ### Begin Downloads
# #### Clear download cache as needed

# +
# from astropy.utils.data import clear_download_cache
# clear_download_cache()
# -

# The first GRB, **test only**

query_df = trigtime_df.head(1)
query_df

out_dir = fits_dir
init = True
timeout = 90
query = Query(query_df, out_dir, init=init)

query

query.Requery()

query.grbs.urls.values
# query.grbs['urls'].values

# #### Export

# +
# import datetime

# fname = "query_results-" + str(datetime.date.today()) + ".h5"
# query.grbs.to_hdf(table_dir / fname, 'f')
# -


