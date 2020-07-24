Setup
=====

Please ensure ``Python`` version **at least 3.8**. Note that the repo is
developed and tested on Linux, not Windows.

.. code:: bash

   python3 --version   # or python --version, if python=python3

Python `virtual
environment <https://docs.python.org/3/library/venv.html>`__ is
recommended. The following assumes use of ```venv``
module <https://docs.python.org/3/library/venv.html>`__.

.. code:: python

   # create a virtual environment (need only once)
   python3 -m venv /path/to/new/virtual/environment
   # activate a virtual environment (every time log in)
   source /path/to/new/virtual/environment/bin/activate
   # update pip itself
   pip3 install -U pip
   # install dependencies
   pip3 install -r /path/to/requirements.txt

You may prefer to assign a near mirror before download/install, for
example, `TUNA <https://mirrors.tuna.tsinghua.edu.cn/help/pypi/>`__.

::

   pip install pip -U
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

If you prefer to use ``conda``, please refer to
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html.

To install the custom package ``pkg``, please run under the **root**
directory of the local repo:

::

   python setup.py install

-  ``ROOT`` (absolute path) for the root directory, default to current
   working directory
-  ``FITS`` (relative to ``ROOT``) for data files formated in
   `FITS <https://fits.gsfc.nasa.gov/fits_standard.html>`__
-  ``TABLE`` (relative to ``ROOT``) for generated dataframes or
   downloaded tables
-  ``IMAGE`` (relative to ``ROOT``) for generated images

Once installed, you could import it the same way as other modules, for
example:

.. code:: python

   from pkg import config

You could also uninstall the package via

::

   pip uninstall pkg

To find out outdated packages, please run

.. code:: python

   pip list --outdated

To update outdated package, you may run

::

   pip install -U package1, package2, ...

or follow https://github.com/pypa/pip/issues/3819.

**NOTE**: one should avoid update all packages unless using virtual
environments, especially in \*nix systems.
