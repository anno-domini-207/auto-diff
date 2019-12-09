Additional features
=======================================

Our package comes with a library package containing the following methods:

- Hamiltonian Monte Carlo
- Newton's Root Finding
- Steepest Descent
- BFGS
- DFP

All of these methods' implementations use our AnnoDomini package to solve for the gradients within the methods. We include demos
of each method in the directory, demos.

A more detailed description of each method is provided below:

Hamiltonian Monte Carlo
~~~~~~~~~~~~~~~~~~~~~~~

Newton's Root Finding
~~~~~~~~~~~~~~~~~~~~~~~
Newton's root finding algorithm is a useful approach to finding the root(s) of functions that are difficult to solve for analytically, i.e. :math:`log(2x) + arctan(3 * x + 5)`.
Mathematically, the algorithm is given by

:math:`x_{n+1} = x_n - \frac{f(x_n)}{\prime{f(x_n)}}`

where we begin with an initial guess :math:`x_0` (scalar), and iterate until a maximum number of iters has been reached (default: 50 iters), or when the estimated root converges (:math:`x_{n+1} - x_n < T`, for some tolerance T).
A useful resource is found `here <http://tutorial.math.lamar.edu/Classes/CalcI/NewtonsMethod.aspx>`_.

Our implementation works for scalar input, :math:`x_0` and single functions. A full demo of this method is provided in the directory, demos.


Steepest Descent
~~~~~~~~~~~~~~~~

Steepest Descent is an unconstrained optimization algorithm used to find the minima/maxima for a specified function. It achieves this by iteratively following the direction of the negative gradient at every step. We determine the optimal step size by following the line-search approach: evaluate the function over a range of possible stepsizes and choosing the minimum value.
Mathematically, the algorithm is give by

:math:`x_{k+1} = x_k + \eta_k s_k`,

where :math:`\eta_k` is the optimal step size, and :math:`s_k = -\nabla f(x_k)`.

Our method works for both single and multivariable inputs, and single output functions. The user must input a function that accounts for specified number of variables desired. A full demo of this method is included in the directory, demos.


BFGS
~~~~

DFP
~~~
