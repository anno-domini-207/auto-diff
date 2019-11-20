Implementation Details
=======================================

The `AutoDiff` class takes as input the value to evaluate the function at. It contains two important attributes, val and der, that respectively store the evaluated function and derivative values at each stage of evaluation.

For instance, consider the case where we want to evaluate the derivative of :math:`x^2+2x+1` evaluated at :math:`x = 5.0`.  This is achieved by first setting ``x = AD.AutoDiff(5.0)``, which automatically stores function and derivative values of :math:`x` evaluated at :math:`x = 5.0` (i.e. ``x.val`` = 5.0 and ``x.der`` = 1.0).  When we pass the desired expression (i.e. :math:`x^2+2x+1`) to Python, :math:`x` is raised to the power of 2, which is carried through `AutoDiff`'s ``__pow__`` method that returns a new `AutoDiff` object with the updated ``val`` and ``der`` values.  Specifically, the new returned object in this case has ``val`` = 25.0 and ``der`` = 10.0, which are the function and derivative values of :math:`x^2` evaluated at :math:`x = 5.0`.  A similar process occurs for the other operation steps (i.e. multiplication and addition), and we eventually obtain the `AutoDiff` object with the desired ``val`` and ``der`` values (i.e. function and derivative values of :math:`x^2+2x+1` evaluated at :math:`x = 5.0`).

Central to this entire process is the `AutoDiff` class which is initiated by:

.. code-block:: python

    class AutoDiff:
        def __init__(self, val=0.0, der=1.0):
            self.val = val
            self.der = der

As mentioned above, the `AutoDiff` class has its own methods that define its behavior for common elementary functions such as addition and multiplication.  Specifically, we currently have the following methods implemented:

::

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
    def log
    def exp
    def logistic

As a simple illustration, here is the way the `__add__` method is implemented:

.. code-block:: python

    def __add__(self, other):
        try:
            val = self.val + other.val
            der = self.der + other.der
        except AttributeError:
            val = self.val + other
            der = self.der
        return AutoDiff(val, der)

We can see that the method returns a new `AutoDiff` object with new updated `val` and `der`.

Note that many methods in the `AutoDiff` class, such as `cos` and `exp`, rely on their counterparts in NumPy (e.g., `numpy.cos` and `numpy.exp`).  NumPy will play even more important role in our future development to support multiple functions of multiple inputs as NumPy arrays support fast and effective vectorized operations.

The following code shows a deeper example of how our `AutoDiff` class is implemented and useful. Consider again the function, :math:`x^2+2x+1`. Suppose we want to use Newton's Method to find the root, using our package. Then we have:

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

External Dependencies
---------------------

- **numpy** :

  - scientific calculations

- **pytest**

  - testing

Additional Implementation
--------------------------

**Multivariable Inputs/Outputs**

Currently, our package handles the single input, single output case. We will also extend our package to handle the following cases:

1. Multiple Input, Single Output

Consider the case where the user would like to input the function,
:math:`f = xy`. Then, the derivative of this would be represented in a Jacobian matrix,
:math:`J = [\frac{df_1}{dx}, \frac{df_1}{dy}] = [y,x]`.

.. code-block:: python

    # Potential Implementation: Multiple Input, Single Output
    x = AD.AutoDiff(3., [1., 0.])
    y = AD.AutoDiff(2., [0., 1.])
    z = x*y
    print(z.val)
    >>> 6.0
    print(z.der)
    >>> [2.0, 3.0]

2. Single Input, Multiple Output

Consider the case where the user would like to input the two functions,
:math:`F = [x^2, 2x]`. Then, the derivative of this would be represented in a Jacobian matrix,
:math:`J = [\frac{df_1}{dx}, \frac{df_1}{dy}] = [2x,2]`.

.. code-block:: python

    # Potential Implementation: Single Input, Multiple Output
    x = AD.AutoDiff(3., 1.)
    z = [x**2, 2*x]
    print(z.val)
    >>> [9.0, 6.0]
    print(z.der)
    >>> [6.0, 2.0]

3. Multiple Input, Multiple Output

Consider the case where the user would like to input the two functions,
:math:`F = [x+y, xy]`. Then, the derivative of this would be represented in a Jacobian matrix,
:math:`J = [[\frac{df_1}{dx}, \frac{df_1}{dy}],[\frac{df_2}{dx}, \frac{df_2}{dy}]] = [[1, 1], [y, x]]`.

.. code-block:: python

    # Potential Implementation: Multiple Input, Multiple Output
    x = AD.AutoDiff(3., [1., 0.])
    y = AD.AutoDiff(2., [0., 1.])
    z = [x+y, x*y]
    print(z.val)
    >>> [5.0, 6.0]
    print(z.der)
    >>> [[1.0, 1.0], [2.0, 3.0]]

For the case where the input variables are arrays (instead of scalars), we are considering to store the values in the Jacobian matrix as a dictionary for its intuitive structure and flexibility to handle different lengths/sizes.

**Additional Module**

AutoDiffExtended.py

  - This module will contain additional functions to leverage the `AutoDiff` module for optimization problems (e.g., root-finding methods) and other extensions (e.g., Hamiltonian Monte Carlo).

  - We will possibly split this module to submodules.

**Demo Class**:

Currently, we have implemented a demo of Newton's Method in this documentation. In the future, we will create a Demo class that runs demos on the following methods in addition to Newton's Method:

- Comparison between ad and numeric methods

- Hamiltonian Monte Carlo to sample from a given function

The structure would resemble:

.. code-block:: python

    class Demo():
    	def compare_ad_numeric(self):
    		# demo of the automatic differentiation
    	def newton_method(self,func = lambda x**2 + 2*x + 1):
    		# demo of the newton's method to solve the roots
    	def hamiltonian_monte_carlo(self,func = lambda x: np.exp(x ** 2))
    		# demo of the hamiltonian monte carlo
