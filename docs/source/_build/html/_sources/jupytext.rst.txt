Jupytext
========

Scripts under the ``demo`` directory are uploaded as ``.py`` files for
better version control. However, it is recommended to view (and run)
these demo scripts as ``.ipynb`` files.
`Jupytext <https://jupytext.readthedocs.io/en/latest/>`__ does the job
well. Example usage:

::

   # Turn notebook.ipynb into a paired ipynb/py notebook
   jupytext --set-formats py,ipynb query-LAT.py
   # Update whichever of notebook.ipynb/notebook.py is outdated
   jupytext --sync query-LAT.py

Please see the
`documentation <https://jupytext.readthedocs.io/en/latest/using-cli.html>`__
for details.
