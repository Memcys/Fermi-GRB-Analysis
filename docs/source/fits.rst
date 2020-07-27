FITS File Handling
==================

*Fermi* provides events (photons and spacecrafts) files in `FITS
format <https://fits.gsfc.nasa.gov/fits_standard.html>`__. Astropy
provides tools to deal with FITS files.

With ``astropy.io.fits``, one can - retrieve information about a GRB,
and - load recorded data.

FITS is a binary format. One may get weird results to *directly* convert
data in FITS to popular data structures such as Numpy Array or Pandas
DataFrame. Fortunately, ``astropy.table.Table`` makes it easy to load
data and then convert to Pandas DataFrame.

``demo/fits.py`` demonstrates how to make good use of Astropy to handle
FITS files. One may prefer to first convert it to ``.ipynb`` via
``jupytext``, see `Jupytext <jupytext>`__.

General references
------------------

-  `Learn Astropy <http://learn.astropy.org/>`__

   -  `FITS File
      Handling <https://docs.astropy.org/en/stable/io/fits/index.html>`__
   -  `Astropy
      Table <https://docs.astropy.org/en/stable/api/astropy.table.Table.html#astropy.table.Table>`__

-  `FITS Standard
   Page <https://fits.gsfc.nasa.gov/fits_standard.html>`__
