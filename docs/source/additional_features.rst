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

Our implementation works for scalar input, :math:`x_0` and single functions. This method can be implemented as follows:

.. code-block:: bash

    $ pip install virtualenv # If Necessary
    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install numpy
    $ pip install scipy
    $ pip install AnnoDomini
    $ python
    >>> from AnnoDomini.newtons_method import *
    >>> import numpy as np
    >>> from scipy import linalg as la
    >>> f = lambda x: np.sin(x) + x * np.cos(x)
    >>> x0 = -3
    >>> demo = Newton(f,x0)
    >>> root = demo.find_root()
    >>> print(root)
    -2.028757838110434
    >>> quit()
    $ deactivate

A full demo of this method is available in the demos subdirectory.


Steepest Descent
~~~~~~~~~~~~~~~~

Steepest Descent is an unconstrained optimization algorithm used to find the minima/maxima for a specified function. It achieves this by iteratively following the direction of the negative gradient at every step. We determine the optimal step size by following the line-search approach: evaluate the function over a range of possible stepsizes and choosing the minimum value.
Mathematically, the algorithm is give by

1. Initialize initial guess, :math:`x_0`
2. Compute :math:`s_k = -\nabla f(x_k)`
3. Iteratively update the estimated value, :math:`x_{k+1} = x_k + \eta_k s_k`, where :math:`\eta_k` is the optimal step size
4. Terminate if maximum number of iterations is reached, or :math:`x_{n+1} - x_n < T`, for some tolerance T

Our method works for both single and multivariable inputs, and single output functions. The user must input a function that accounts for specified number of variables desired. This method can be implemented as follows:

.. code-block:: bash

    $ pip install virtualenv # If Necessary
    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install numpy
    $ pip install scipy
    $ pip install AnnoDomini
    $ python
    >>> from AnnoDomini.steepest_descent import *
    >>> import numpy as np
    >>> from scipy import linalg as la
    >>> def f(args):
    >>>     [x,y] = args
    >>>     ans = 100*(y-x**2)**2 + (1-x)**2
    >>>     return ans
    >>> x0 = [2,1]
    >>> sd = SteepestDescent(f, x0)
    >>> root = sd.find_root()
    >>> print(root)
    [1. 1.]
    >>> quit()
    $ deactivate

A full demo of this method is available in the demos subdirectory.


BFGS
~~~~
BFGS, or the Broyden–Fletcher–Goldfarb–Shanno algorithm, is a
first-order quasi-Newton optimization method, which approximates the Hessian matrix with the gradient and direction of a function.
The algorithm is as follows, in terms of the Approximate Hessian, :math:`B_k`, the step  :math:`s_k`, and :math:`y_k`

1. Solve for :math:`s_k` by solving the linear system :math:`B_k s_k = -y_k`
2. :math:`x_{k+1} = s_k + x_k`
3. :math:`y_k = \nabla x_{k+1} -  \nabla x_{k}`
4. :math:`B_{k+1} =  B_k + \frac{y_k y_k^T}{y_k^T s_k} + \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}`
5. Terminate when :math:`s_k <` Tolerance

This method can be implemented as follows:

.. code-block:: bash

    $ pip install virtualenv # If Necessary
    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install numpy
    $ pip install scipy
    $ pip install AnnoDomini
    $ python
    >>> from AnnoDomini.BFGS import *
    >>> import numpy as np
    >>> from scipy import linalg as la
    >>> def f(args):
    >>>     [x,y] = args
    >>>     ans = 100*(y-x**2)**2 + (1-x)**2
    >>>     return ans
    >>> x0 = [2,1]
    >>> sd = BFGS(f, x0)
    >>> root = sd.find_root()
    >>> print(root)
    [1. 1.]
    >>> quit()
    $ deactivate


A full demo of this method is available in the demos subdirectory.


DFP
~~~
DFP, or the Davidon–Fletcher–Powell formula, is another
first-order quasi-Newton optimization method, which also approximates the Hessian matrix with the gradient and direction of a function.
The algorithm is as follows, in terms of the Approximate Hessian, :math:`B_k`, the step  :math:`s_k`, :math:`\gamma_k  = \frac{1}{y_k^T s_k}`

1. Solve for :math:`s_k` by solving the linear system :math:`B_k s_k = -y_k`
2. :math:`x_{k+1} = s_k + x_k`
3. :math:`y_k = \nabla x_{k+1} -  \nabla x_{k}`
4. :math:`B_{k+1} =  (I - \gamma_k y_k s_k^T)B_k(I - \gamma_k s_k y_k^T) + \gamma_k y_k y_k^T`
5. Terminate when :math:`s_k <` Tolerance

This method can be implemented as follows:

.. code-block:: bash

    $ pip install virtualenv # If Necessary
    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install numpy
    $ pip install scipy
    $ pip install AnnoDomini
    $ python
    >>> from AnnoDomini.DFP import *
    >>> import numpy as np
    >>> from scipy import linalg as la
    >>> def f(args):
    >>>     [x,y] = args
    >>>     return np.e**(x+1) + np.e**(-y+1) + (x-y)**2
    >>> x0 = [2,0]
    >>> sd = DFP(f, x0)
    >>> root = sd.find_root()
    >>> print(root)
    [-0.43837842  0.43837842]
    >>> quit()
    $ deactivate


A full demo of this method is available in the demos subdirectory.


=======
**Note:**  DFP is empirically significantly less performant than BFPS. For instance, it may take up to 1 million iterations to converge on the Rosenbrock function.
>>>>>>> 7f228a40f8410687d6da6d9538c22729c76b62b7
