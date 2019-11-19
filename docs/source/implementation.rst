Implementation Details
=======================================

The AutoDiff class takes as input the value to evaluate the function at. It contains two important attributes, val and der, that respectively store the evaluated function and derivative values at each stage of evaluation.

For instance, consider the case where we want to evaluate the derivative of `2x^3+10` evaluated at `x = 5.0`.  This is achieved by first setting ``x = AD.AutoDiff(5.0)``, which automatically stores function and derivative values of `x` evaluated at `x = 5.0 (i.e. ``x.val`` = 5.0 and ``x.der`` = 1.0).  When we pass the desired expression (i.e. `2x^3+10`) to Python, `x` is raised to the power of 3, which is carried through `AutoDiff`'s `__pow__` method that returns a new `AutoDiff` object with the updated `val` and `der` values.  Specifically, the new returned object in this case has ``val = 125.0`` and ``der = 75.0``, which are the function and derivative values of `x^3` evaluated at `x = 5.0`.  The same process occurs for the subsequent operation steps (i.e. multiplication by 2 and addition by 10), and we eventually obtain the ``AutoDiff`` object with the desired `val` and `der` values (i.e. function and derivative values of `2x^3+10` evaluated at `x = 5.0`).

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

The following code shows a deeper example of how our AutoDiff class is implemented and useful. Consider again the function, :math:`x^2+2x+1`. Suppose we want to use Newton's Method to find the root, using our package. Then we have:

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

Currently, our package handles the single input, single output case. To extend further, we will also extend our package to handle the following cases:

For the cases provided below (where the input variables are scalars), we would require the input to be an array, and we will handle the output as an array.

- Multiple input, Single output

Mathematically, consider the case where the user would like to input the function,
:math:`f = xy`. Then, the derivative of this would be represented in a Jacobian matrix,
:math:`J = [\frac{df_1}{dx}, \frac{df_1}{dy}] = [y,x]`.

.. code-block:: python

    # Possible Implementation: multiple inputs, single output
    def f(x, y):
      return x*y


- Single input, Multiple output

Mathematically, consider the case where the user would like to input the two functions,
:math:`F = [x^2, 2x]`. Then, the derivative of this would be represented in a Jacobian matrix,
:math:`J = [\frac{df_1}{dx}, \frac{df_1}{dy}] = [2x,2]`.

.. code-block:: python

    # Possible Implementation: single input, multiple outputs
    def f(x):
      return [x**2, 2*x]


- Multiple input, Multiple output

Mathematically, consider the case where the user would like to input the two functions,
:math:`F = [x+y, xy]`. Then, the derivative of this would be represented in a Jacobian matrix,
:math:`J = [[\frac{df_1}{dx}, \frac{df_1}{dy}],[\frac{df_2}{dx}, \frac{df_3}{dy}]] = [[1, 1], [y, x]]`.

.. code-block:: python

    # Possible Implementation: multiple inputs, multiple outputs
    def f(x, y):
      return [x+y, xy]

For the case where the input variables are arrays, we would store the values in the Jacobian matrix as a dictionary (for its intuitive structure) or an array.

**Additional Module**

AutoDiffExtended.py

  - Contain additional functions to leverage the AutoDiff module for optimization problems (i.e. root-finding methods) and other extensions (i.e. hamiltonian monte carlo).

  - We will possibly split this model to submodules.

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
