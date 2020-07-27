Guide Usage
===========

The detailed code of this guide can be seen in ``demo/main.py``.

Suppose you have the required FITS data under the directory assigned by
``grb.config.path.FITS``. Other required information

-  GRB/GCN name
-  GBM MET (reference time)
-  first low peak time (relative to the reference time)
-  red shift

are in ``grb.config.path.TABLE / "query-results-latest.h5"`` and named
“table”.

Follow these steps to perform the analysis.

1. Instantiate the ``PHTable``.
2. Run the method ``regPlot`` to obtain the power parameter in the power
   law.
3. Manually assign the primed class IV (classIV_less). However, this
   primed class is actually ommitable in my analysis.
4. Run one of the method ``randPlotI``, ``randPlotII``, ``randPlotIII``,
   or ``randPlotIV`` to generate required number (via ``repeat``) of
   samples and plot the results.

Note that you may use the Python function ``dir`` to list all available
attributes of any Python objects. For example,

.. code:: python

   from grb.lat.analysis import PHTable

   dir(PHTable)
