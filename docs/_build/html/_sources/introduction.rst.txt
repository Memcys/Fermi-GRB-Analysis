Introduction
============

The repo
`Fermi-GRB-Analysis <https://github.com/Memcys/Fermi-GRB-Analysis>`__
aims to reproduce the results in the paper [Xu2018][1].

Goals
-----

The main goals are:

1. Retrive the raw data (Fermi LAT/GRB photon events)
2. Do the calculus and statistics
3. Make plots

File Tree
---------

-  ``data``: parent directory for all kinds of data; can be altered by
   modifying ``grb/config/path.py`` or assigned by running ``setup.py``
-  ``demo``: demo scripts that demonstrate the functionalities of the
   package ``grb`` (**entrance for usage**). It recommended to convert
   all ``.py`` to ``.ipynb`` before viewing and running. (See
   `Jupytext <jupytext#Jupytext>`__ for help.)

   -  ``main.py``: use ``pkg/lat/analysis.py`` for data analysis and
      data visualization (goals 2 and 3)
   -  ``query-LAT.py``: use ``pkg/lat/query.py`` to request *Fermi* LAT
      data (goal 1)

-  ``docs``: documentation for the repo
-  ``grb``: Python namespace package

   -  ``config``: submodule for configuration

      -  ’path.py`: for unified paths for root, data, tables and images
      -  ``__init__.py``: necessary file to make ``grb`` a `namespace
         package <https://docs.python.org/3/tutorial/modules.html#packages>`__

   -  ``lat``: submodule for *Fermi* LAT analysis

      -  ``analysis.py``: for data analysis and data visualization
      -  ``query.py``: for requests of *Fermi* LAT
      -  ``__init__.py``

-  ``setup.py``: setup script to install the packag ``grb``

[1]: Xu, H., & Ma, B.-Q. (2018). Regularity of high energy photon events
from gamma ray bursts. Journal of Cosmology and Astroparticle Physics,
2018(01), 050–050. https://doi.org/10.1088/1475-7516/2018/01/050,
http://arxiv.org/abs/1801.08084
