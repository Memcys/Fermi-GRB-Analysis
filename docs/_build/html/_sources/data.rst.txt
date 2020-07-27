Data Source
===========

Overview
--------

======================================= ================================== ===========================
Type                                    Usage                              Source
======================================= ================================== ===========================
GRB/GCN name                            Request and downloading FITS files See `below <#lat-data>`__
GRB red shift                           Calculation                        See `below <#red-shifts>`__
First low peak time of GBM TTE          Reference time                     GBM TTE FITS header
Observed time and energy of LAT photons Calculation                        LAT FITS header
======================================= ================================== ===========================

LAT Data
--------

This project requires information of photon events detected by *Large
Area Telegram*. The `data
server <https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi>`__
currently provides Pass 8 (P8R3) data. There are at least two ways to
query the required LAT FITS files: 1. Manually input information about a
LAT GRB in the `data
server <https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi>`__
2. Use the package ``fermi`` provided by ``astroquery`` to query the
information.

``demo/query-LAT.py`` is the demo script to query and download LAT FITS
files making use of ``astroquery``.

`This
webpage <https://fermi.gsfc.nasa.gov/ssc/observations/types/grbs/lat_grbs/table.php>`__
lists all Fermi LAT Gamma-Ray Bursts (GRBs) from 080825C to 180720B.
Note that the field “Object name or coordinates” in the `data
server <https://fermi.gsfc.nasa.gov/cgi-bin/ssc/LAT/LATDataQuery.cgi>`__
corresponds to the ``GCN Name`` in the table.

Downloaded FITS files are assumed to be saved under the directory
assigned by ``grb.config.path.FITS``, which is by default ``data/fits/``
relative to the root directory.

LAT FITS files that I used have been released
`here <https://github.com/Memcys/LAT-GRB-data/releases>`__. Please
download a release file and extraced to ``data/fits/`` such that it
looks like: - data/fits/ - 080916/ - criteria.csv - info.csv -
L200408135242F357373F23_PH00.fits - L200408135242F357373F23_SC00.fits -
090323/ - … - …

GBM TTE Data
------------

This project does not perform a GBM analysis. There are, however, some
*dirty* scripts that try to do such an anlysis, which assumes that GBM
TTE FITS files for a certain GRB be located under its GRB directory. For
example, - data/fits/ - 080916/ - criteria.csv - info.csv -
L200408135242F357373F23_PH00.fits - L200408135242F357373F23_SC00.fits -
TTE/ - glg_tte_b0_bn080916009_v01.fit - glg_tte_b1_bn080916009_v01.fit -
glg_tte_n0_bn080916009_v01.fit - … - glg_tte_nb_bn080916009_v01.fit - …

``.fit`` files can be equally treated as ``.fits``.

To query or download the GBM TTE files, there are at least two ways: 1.
Search in the `Fermi GBM Trigger Catalog
(fermi) <https://heasarc.gsfc.nasa.gov/db-perl/W3Browse/w3table.pl?tablehead=name%3Dfermigtrig&Action=More+Options>`__.
2. Suppose the trigger name of a GRB is known. Then the FITS file may be
downloaded in:
https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/triggers/yyyy/bnyymmddfff/current/glg_tte_xN_bnyymmddfff_vzz.fit,
where ``yyyy`` is the year, ``yymmdd`` the Year-Month-Date, ``fff`` the
fraction of a day, ``x`` in ``b`` (for BGO) or ``n`` (for NaI), ``N``
the hexadecimal (``b``: 0-1, ``n``: 0-b), ``zz`` the version (typically
``00`` or ``01``). Note that ``yymmddfff`` corresponds to the column
“GRB”. See
`here <https://fermi.gsfc.nasa.gov/ssc/library/support/Science_DP_FFD_RevA.pdf>`__
for details.

Red shifts
----------

Red shifts can be obtained in the `GCN Circular
Archive <http://gcn.gsfc.nasa.gov/gcn3_archive.html>`__ or
`here <http://www.mpe.mpg.de/~jcg/grbgen.html>`__.
