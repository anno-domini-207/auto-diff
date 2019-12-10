Software Organization
=======================================

Directory Structure
-------------------
::

  AnnoDomini/
    AutoDiff.py
    BFGS.py
    DFP.py
    hamilton_mc.py
    newtons_method.py
    steepest_descent.py
  docs/
    source/
      .index.rst.swp
      conf.py
      index.rst (documentation file)
    Makefile
    make.bat
    milestone1.md
  tests/
    test_AutoDiff.py
    test_BFGS.py
    test_DFP.py
    test_hmc.py
    test_newton.py
    test_steepest_descent.py
  .gitignore
  .travis.yml
  LICENSE
  README.md

Basic Modules
-------------
- AutoDiff.py

  - Contains implementation of the master class and its methods for calculating derivatives of elementary functions (list of methods shown in **Core Classes** section below).

- newtons_method.py

  - Contains implementation of the root finding algorithm, Newton's Method

- steepest_descent.py

  - Contains implementation of the optimization method, Steepest Descent

- BFGS.py

  - Contains implementation of the optimization method, BFGS

- DFP.py

  - Contains implementation of the optimization method, BFGS

- hamilton_mc.py

  - Contains the simulation method, Hamiltonian Monte Carlo

Testing
-------

Our tests are contained in ``tests/`` directory.
- ``test_AutoDiff`` test_AutoDiff.py is used to test the functions in the AutoDiff Class. It includes tests for both scaler and vector inputs and outputs to ensure our core implementation is correct for general cases.
- ``test_BFGS`` tests BFGS converges on the Rosenbock function
- ``test_BFGS`` tests DFP converges
- ``test_hmc`` tests the Hamiltonian Monte Carlo method
- ``test_newton`` tests Newton's root-finding method
- ``test_steepestDescent`` tests that steepest descent optimization converges


Our test suites are hosted through TravisCI and CodeCov. We run TravisCI first to test the accuracy and CodeCov to test the test coverage. The results can be inferred via the README section.

Our tests are integrated via the TravisCI. that is, call ask TravisCI to CodeCov after completion.

.. figure:: TravisCI.png
    :width: 2000px
    :align: center
    :height: 300px
    :alt: alternate text
    :figclass: align-center

Packaging
---------
Details on how to install our package are included in the section, `How to use Anno Domini <https://cs207-finalproject-group15.readthedocs.io/en/latest/how_to_use.html>`_.

We use Git to develop the package; we follow instructions `:here <https://python\-packaging.readthedocs.io/en/latest/>`_ to package our code and distribute it on PyPi. Instead of using a framework such as PyScaffold, we will adhere to the proposed directory structure. We provide necessary documentation via .rst files (rendered through Sphinx) to provide a clean, readable format on Github.
