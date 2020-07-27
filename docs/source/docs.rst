Build the Docs
==============

Requirements (can be installed via ``pip`` or ``conda``):

::

   sphinx
   sphinx-rtd-theme

First, generate package API via (at ROOT directory)

::

   sphinx-apidoc -f --implicit-namespaces -o docs/source grb

Then build (html format) via

::

   sphinx-build -b html docs/source docs/_build/html

And all done.

To view the output documentation, please open the file
``docs/_build/html/index.html`` with any web browser you like, e.g.,

::

   firefox docs/_build/html/index.html
