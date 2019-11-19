Anno Domini's Documentation
=======================================

Introduction
------------

Calculating the derivative and gradients of functions is essential to many computational and mathematical fields. In particular, this is useful in machine learning because these ML algorithms are centered around minimizing an objective loss function. Traditionally, scientists have used numerical differentiation methods to compute these derivatives and gradients, which potentially accumulates floating point errors in calculations and penalizes accuracy.

Automatic differentiation is an algorithm that can solve complex derivatives in a way that reduces these compounding floating point errors. The algorithm achieves this by breaking down functions into their elementary components and then calculates and evaluates derivatives at these elementary components. This allows the computer to solve derivatives and gradients more efficiently and precisely. This is a huge contribution to machine learning, as it allows scientists to achieve results with more precision.

Background
----------

In automatic differentiation, we can visualize a function as a graph structure of calculations, where the input variables are represented as nodes, and each separate calculation is represented as an arrow directed to another node. These separate calculations (the arrows in the graph) are the function's elementary components.

We then are able to compute the derivatives through a process called the forward mode. In this process, after breaking down a function to its elementary components, we take the symbolic derivative of these components via the chain rule. For example, if we were to take the derivative of :math:`\sin(x)`, we would have that :math:`\frac{d}{dx}\sin(x) = \sin^{\prime}(x)x^{\prime}`, where we treat “x” as a variable, and x prime is the symbolic derivative that serves as a placeholder for the actual value evaluated here. We then calculate the derivative (or gradient) by evaluating the partial derivatives of elementary functions with respect to each variable at the actual value.

For further visualization automatic differentiation, consider the function, :math:`x^2+2x+1`. The computational graph for this function looks like:

.. figure:: ad_graph.jpg
    :width: 2000px
    :align: center
    :height: 500px
    :alt: alternate text
    :figclass: align-center


The corresponding evaluation trace looks like:

===========   ===================  ======================  ===================================   ============
Trace         Elementary Function  Current Function value  Elementary Function Derivative        :math:`f(1)`
===========   ===================  ======================  ===================================   ============
:math:`x_1`   :math:`x_1`          x                       1                                     1

:math:`x_2`   :math:`x_2`          x                       1                                     1

:math:`x_3`   :math:`x_3`          x                       1                                     1

:math:`x_4`    :math:`x_1x_2`       :math:`x^2`            :math:`\dot{x_1}x_2 + x_1\dot{x_2}`   2

:math:`x_5`   :math:`x_4 + 2x_3`   :math:`x^2 + 2x`        :math:`\dot{x_4} + 2\dot{x_3}`        4

:math:`x_6`   :math:`x_5 + 1`      :math:`x^2 + 2x + 1`    :math:`\dot{x_5}`                     4
===========   ===================  ======================  ===================================   ============

For the single output case, what the forward model is calculating the product of gradient and the initializaed vector p, represented mathematically as :math:`D_px = \Delta x \cdot p`. For the multiple output case, the forward model calculates the product of Jacobian and the initialized vector p: :math:`D_px = J\cdot p`. We can obtain the gradient or Jacobian matrix of the function through different seeds of the vector p.


How to use Anno Domini
----------------------

How to Install
^^^^^^^^^^^^^^

**Install via Pip:**

.. code-block:: bash

    pip install -i https://test.pypi.org/simple/ AnnoDomini

**Install in a Virtual Environment:**

.. code-block:: bash


    $ pip install virtualenv # If Necessary
    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install numpy
    $ pip install -i https://test.pypi.org/simple/ AnnoDomini
    >> import AnnoDomini.AutoDiff as AD
    >> AD.AutoDiff(3)
    Function Value: 3 | Derivative Value: 1.0
    >> quit()
    $ deactivate


*Note*: Numpy and Pytest are also required. If they are missing an error message will indicate as much.


How to Use
^^^^^^^^^^

Consider the following example in scalar input case:s
Suppose we want to to find the derivative of :math:`x^2+2x+1`. We can the utilize the AnnoDomini package as follows:

.. code-block:: python

    f = lambda x: x**2 + 2*x + 1
    temp = AD.AutoDiff(1.5)
    print(temp)
    >> Function Value: 1.5 | Derivative Value: 1.0
    df = f(temp)
    >> Function Value: 6.25 | Derivative Value: 5.0

Say we only want to access only the value or derivative component. We can do this as follows:

.. code-block:: python

    val, der = df.val, df.der
    print(der)
    >> 5.0
    print(val)
    >> 6.25

Software Organization
---------------------

Directory Structure
^^^^^^^^^^^^^^^^^^^
::

  AnnoDomini/
    AutoDiff.py
  docs/
    source/
      .index.rst.swp
      conf.py
      index.rst (documentation file)
    Makefile
    make.bat
    milestone1.md
  tests/
    initial_test.py
    test_AutoDiff.py
  .gitignore
  .travis.yml
  LICENSE
  README.md

Basic Modules
^^^^^^^^^^^^^
- AutoDiff.py

  - Contains implementation of the master class and its methods for calculating derivatives of elementary functions (list of methods shown in **Core Classes** section below).


Testing
^^^^^^^

- Where do the tests live?
- How are they run?
- How are they integrated?

Our tests are contained in tests/test_AutoDiff.py. Our test suites are hosted through TravisCI and CodeCov. We run TravisCI first to test the accuracy and CodeCov to test the test coverage.

Packaging
^^^^^^^^^

Details on how to install our package are included in the section, **How to use Anno Domini**.

We use Git to develop the package; after we notice that the package is mature, we follow instructions `here <https://python\-packaging.readthedocs.io/en/latest/index.html/>`_ to package our code and distribute it on the PyPi. Instead of using a framework such as PyScaffold, we will adhere to the proposed directory structure. We provide necessary documentation via both MD and .rst files (rendered through Sphinx) to provide a clean, readable format on Github.

Implementation Details
----------------------

Core Data Structures
^^^^^^^^^^^^^^^^^^^^
Given that the computation table we will be constructing is inherently ordered, it makes sense to use arrays to represent the necessary data. As we build and improve our AutoDiff implementation, we will look to optimize these structures via pre-allocation and leverage numpy arrays when possible. We will also be creating pandas dataframes in order to create a nice, well-structured table with good printing functionality.

Core Classes
^^^^^^^^^^^^

**AutoDiff Class**:

Methods Included in AutoDiff:

::

    def __init__
    def __repr__
    def __eq__
    def __ne__
    def __add__
    def __radd__
    def __sub__
    def __rsub__
    def __mul__
    def __rmul__
    def __truediv__
    def __rtruediv__
    def __pow__
    def __rpow__
    def __neg__
    def sqrt
    def sin
    def cos
    def tan
    def arcsin
    def arccos
    def arctan
    def sinh
    def cosh
    def tanh
    def log
    def exp
    def logistic

The structure of the AutoDiff class looks as follows:

.. code-block:: python

    class AutoDiff:
    	def __init__(self, val=0, der=1):
    		self.val = val
    		self.der = der

The following code shows a deeper example of how our AutoDiff class is implemented and useful:

Consider again the function, :math:`x^2+2x+1`. Suppose we want to use Newton's Method to find the root, using our package. Then we have:

.. code-block:: python

    import AnnoDomini.AutoDiff as AD
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy import linalg as la

    f = lambda x: x**2 + 2*x + 1
    x0 = 1.5

    def newtons_method(f, x0, iters = 100, tol = 1e-6, alpha = 1):
        """Use Newton's method to approximate a root.

        Inputs:
            f (function): A function to handle.
            x0 (float): Initial guess.
            iters (int): Maximum number of iterations before the function
                returns. Defaults to 100.
            tol (float): The function returns when the difference between
                successive approximations is less than tol.
            alpha (float): Defaults to 1.  Allows backstepping.

        Returns:
            A float that is the root that Newton's method finds
        """
        # Newton's Method on Scalar Input
        xold = x0
        for i in range(iters):
            # compute derivative via AutoDiff
            temp = AD.AutoDiff(xold)
            df = f(temp)

            #solve for x_k1
            xnew = xold - alpha * f(xold)/df.der
            if la.norm(xnew - xold) < tol:
                return xnew
            else:
                xold = xnew

        return xnew

    ans = newtons_method(f,x0)

    # plot solution
    xs = np.linspace(-7,5,100)
    plt.plot(xs, f(xs), label="f")
    plt.scatter(ans, f(ans),label="Root")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Visual of Newton's Method on $x^2 + 2x + 1$")
    plt.legend()
    plt.show()

.. figure:: example_plot1.jpg
    :width: 2000px
    :align: center
    :height: 500px
    :alt: alternate text
    :figclass: align-center


Important Attributes
^^^^^^^^^^^^^^^^^^^^
The AutoDiff class contains the following attributes:

- self.var:  the value of the calculated function (can be a vector or scalar)

- self.der: the derivative of the calculated function (can be a vector or scalar)

External Dependencies
^^^^^^^^^^^^^^^^^^^^^
- **numpy** :

  - scientific calculations

- **scipy**:

  - accomodating several statistical probability functions

- **pandas**:

  - Visualization

- **functools** (and other built in python dependencies):

  - wrapping functions

  - manipulating built in data structures

- **pytest**

  - testing

Elementary Functions
^^^^^^^^^^^^^^^^^^^^

To handle these functions, we will overload these functions in the AD class, and define the updated derivative and value for the class. For instance, we may define the logarithmic function as follows:

.. code-block:: python

    class AutoDiff()
    	...
    	def log(self):
    		# this would be called for numpy.log(AD)
    		val = np.log(self.val)
    		der = self.der * 1/self.val
    		return AutoDiff(val,der)

A full list of methods are included in the **AutoDiff Methods** section above.

Additional Implementation
^^^^^^^^^^^^^^^^^^^^^^^^^
**Multivariable Inputs/Outputs**

We will handle multiple inputs and multiple outputs in the following ways:

.. code-block:: python

    # multiple input
    def f(x, y, z):
      return 2*x*y + z**4
    ad = grad(f)
    print(ad(1, 2, 3))
    >>>>>[ 4. 2. 108.]

.. code-block:: python

    # multiple input and multiple output
    def f(x, y, z):
    	return [2*x*y + z**4, x*y*z]
    ad = grad(f)
    print(ad(1, 2, 3))
    >>>>>[[4. 2. 108.]
    	  [6. 3. 2]]

**Additional Module**

AutoDiffExtended.py

  - Potentially contains additional functions to leverage the AD module for the optimization problem (May include finding roots where the derivative/gradient equals zero) and other extensions like sampling problems (May includes methods like hamiltonian monte carlo).

  - If we have thought of other extensions and this file to be too long, we can split the model to several submodules.

**Demo Class**:

Run some demos (include optimization demo, etc). For now we have thought of the following three demos (could be updated afterwards):

- Comparison between ad and numeric methods

- Use newton's methods to calculate the root of a given function

- Use hamiltonian monte carlo to sample from a given function

The structure would resemble:

.. code-block:: python

    class Demo():
    	def compare_ad_numeric(self):
    		# demo of the automatic differentiation
    	def newton_method(self,func = lambda x**2 - 2*x + 1):
    		# demo of the newton's method to solve the roots
    	def hamiltonian_monte_carlo(self,func = lambda x: np.exp(x ** 2))
    		# demo of the hamiltonian monte carlo

Future Features
---------------
.. toctree::
   :maxdepth: 2
