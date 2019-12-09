Additional features
=======================================

Our package comes with a library package containing the following methods:

- Newton's Root Finding
- Steepest Descent
- Broyden–Fletcher–Goldfarb–Shanno (BFGS)
- Davidon–Fletcher–Powell formula (DFP)
- Hamiltonian Monte Carlo(HMC)

All of these methods' implementations use our AnnoDomini package to solve for the gradients within the methods. We include demos of each method in the directory, demos.

A more detailed description of each method is provided below:

Newton's Root Finding
----------------------

Background
~~~~~~~~~~

Newton's root finding algorithm is a useful approach to finding the root(s) of functions that are difficult to solve for analytically, i.e. :math:`log(2x) + arctan(3x + 5)`.
Mathematically, the algorithm is given by

:math:`x_{n+1} = x_n - \frac{f(x_n)}{\prime{f(x_n)}}`

For the multivariate case:

:math:`x_{n+1} = x_n - (J(f)(x_n))^{-1} f(x_n)`

where we begin with an initial guess :math:`x_0` (scalar), and iterate until a maximum number of iters has been reached (default: 50 iters), or when the estimated root converges (:math:`x_{n+1} - x_n < T`, for some tolerance T).
A useful resource is found `here <http://tutorial.math.lamar.edu/Classes/CalcI/NewtonsMethod.aspx>`_.

API
~~~
``Newton Class``:

	- ``f``: function of interest. If f is a scaler function, we can define f as follows:
	
		.. code-block:: python
		
			# option one
			def f(x):
			   return np.sin(x) + x * np.cos(x)
			
			# option two
			f = lambda x: np.sin(x) + x * np.cos(x)
		
		if f is a multivariate function:
		
		.. code-block:: python
			
			def f(x,y):
				return x ** 2 + y ** 2 - 3 * x * y - 4 # (x-y)^2 = 9
			
			# or
			f = lambda x, y: x ** 2 + y ** 2 - 3 * x * y - 4
	
	
	- ``x0``: initial point to start. Support both scaler and vector expressions.
	
		.. code-block:: python
			
			x0 = [1,2,3] # supported
			x0 = 1 # supported
	
	- ``maxiter``: max iteration number if convergence not method
	
	- ``tol``: stop condition parameter. :math: `||x_n - x_{n-1}|| < tol`
	
	- ``alpha``: constant in the newtons_method. The smaller the smaller the step.
	
``find_root``: return a root of a function.

Demo
~~~~

.. code-block:: python

    >>> from AnnoDomini.newtons_method import Newton
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

And then we can do the visulization:

.. code-block:: python
	
	from matplotlib import pyplot as plt
	xs = np.linspace(-7,5,100)
	plt.plot(xs, f(xs), label="f")
	plt.scatter(root, f(root),label="Root", color = 'black')
	plt.scatter(x0, f(x0),label="initial", color = 'red')
	plt.xlabel("x")
	plt.ylabel("y")
	plt.title("Visual of Newton's Method on $sin(x) + x * cos(x)$")
	plt.axhline(y = 0, color = 'red')
	plt.legend()
	plt.show()

.. figure:: newtons_method.png
    :width: 2000px
    :align: center
    :height: 500px
    :alt: alternate text
    :figclass: align-center

.. code-block:: python

	>>> def f(x,y):
		return x ** 2 + y ** 2 - 3 * x * y - 4 # (x-y)^2 = 9
	>>> x0 = 1.0
	>>> y0 = -2.0
	>>> init_vars = [x0, y0]
	>>> demo = Newton(f,init_vars)
	>>> ans = demo.find_root()
	>>> print(ans)
	[0.45925007 -1.37598283]

And we can do the visulization by the following ways:

.. code-block:: python

    delta = 0.025 
    lam1 = np.arange(-3, 3, delta) 
    lam2 = np.arange(-5, 3, delta) 
    Lam1, Lam2 = np.meshgrid(lam1, lam2) 
    value = Lam1 ** 2 + Lam2 ** 2 - 3 * Lam1 * Lam2 -4
    CS = plt.contour(Lam1, Lam2, value,levels = 30) 
    plt.scatter(x0,y0,color = "red",label = "Initialization")
    plt.scatter(ans[0],ans[1],color = "green",label = "root found") 
    plt.clabel(CS, inline=1, fontsize=10) 
    plt.xlabel('x') 
    plt.ylabel('y') 
    plt.legend() 
    plt.title('Level Curve of $x^2 + y^2 - 3*x*y - 4$ wrt x and y')
    plt.savefig('newton_multivar.png')


	
.. figure:: newton_multivar.png
    :width: 2000px
    :align: center
    :height: 500px
    :alt: alternate text
    :figclass: align-center


A full demo of this method is available in the demos subdirectory.

Steepest Descent
----------------

Background
~~~~~~~~~~

Steepest Descent is an unconstrained optimization algorithm used to find the minima/maxima for a specified function. It achieves this by iteratively following the direction of the negative gradient at every step. We determine the optimal step size by following the line-search approach: evaluate the function over a range of possible stepsizes and choosing the minimum value.

Mathematically, the algorithm is give by

1. Initialize initial guess, :math:`x_0`
2. Compute :math:`s_k = -\nabla f(x_k)`
3. Iteratively update the estimated value, :math:`x_{k+1} = x_k + \eta_k s_k`, where :math:`\eta_k` is the optimal step size
4. Terminate if maximum number of iterations is reached, or :math:`x_{n+1} - x_n < T`, for some tolerance T

Our method works for both single and multivariable inputs, and single output functions. The user must input a function that accounts for specified number of variables desired. This method can be implemented as follows:

API
~~~

``SteepestDescent Class``:

	- ``f``: function of interest. If f is a scaler function, we can define f as follows:
	
		.. code-block:: python
		
			# option one
			def f(x):
			   return np.sin(x) + x * np.cos(x)
			
			# option two
			f = lambda x: np.sin(x) + x * np.cos(x)
		
		if f is a multivariate function:
		
		.. code-block:: python
			
			def f(args):
				[x,y] = args
				ans = 100*(y-x**2)**2 + (1-x)**2
				return ans
	
	- ``x0``: initial point to start. Support both scaler and vector expressions.
	
		.. code-block:: python
			
			x0 = [1,2,3] # supported
			x0 = 1 # supported
	
	- ``maxiter``: max iteration number if convergence not method
	
	- ``tol``: stop condition parameter. :math: `||x_n - x_{n-1}|| < tol`
	
	- ``step``: constant in the steepest descent. The smaller the smaller the step.
	
``find_root``: return a root of a function.

Demos
~~~~~

.. code-block:: python

    >>> from AnnoDomini.steepest_descent import SteepestDescent
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

And also we can do the visualization:

.. code-block:: python

	X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 8, 100))
	Z = f(np.array([X,Y]))
	fig = plt.subplots(1,1, figsize = (10,7))
	plt.contour(X, Y, Z)
	plt.plot(ans[:,0], ans[:,1], "-.", label="Trajectory")
	plt.scatter(root[0],root[1], label="Root", c="red")
	plt.scatter(-1,1, label="Initial Guess", c ="orange")
	plt.title("Convergence of Steepest Descent on Rosenbrock Function")
	plt.xlim(-3, 3)
	plt.ylim(-2, 8)
	plt.legend()
	plt.show()

.. figure:: steepestDescent.png
    :width: 2000px
    :align: center
    :height: 500px
    :alt: alternate text
    :figclass: align-center

A full demo of this method is available in the demos subdirectory.


Broyden–Fletcher–Goldfarb–Shanno (BFGS)
---------------------------------------

Background
~~~~~~~~~~

BFGS, or the Broyden–Fletcher–Goldfarb–Shanno algorithm, is a
first-order quasi-Newton optimization method, which approximates the Hessian matrix with the gradient and direction of a function.
The algorithm is as follows, in terms of the Approximate Hessian, :math:`B_k`, the step  :math:`s_k`, and :math:`y_k`

1. Solve for :math:`s_k` by solving the linear system :math:`B_k s_k = -y_k`
2. :math:`x_{k+1} = s_k + x_k`
3. :math:`y_k = \nabla x_{k+1} -  \nabla x_{k}`
4. :math:`B_{k+1} =  B_k + \frac{y_k y_k^T}{y_k^T s_k} + \frac{B_k s_k s_k^T B_k}{s_k^T B_k s_k}`
5. Terminate when :math:`s_k <` Tolerance

API
~~~

``BFGS Class``:

	- ``f``: function of interest. If f is a scaler function, we can define f as follows:
	
		.. code-block:: python
		
			# option one
			def f(x):
			   return np.sin(x) + x * np.cos(x)
			
			# option two
			f = lambda x: np.sin(x) + x * np.cos(x)
		
		if f is a multivariate function:
		
		.. code-block:: python
			
			def f(args):
				[x,y] = args
				ans = 100*(y-x**2)**2 + (1-x)**2
				return ans
	
	- ``x0``: initial point to start. Support both scaler and vector expressions.
	
		.. code-block:: python
			
			x0 = [1,2,3] # supported
			x0 = 1 # supported
	
	- ``maxiter``: max iteration number if convergence not method
	
	- ``tol``: stop condition parameter. :math: `||x_n - x_{n-1}|| < tol`
	
``find_root``: return a root of a function.

Demos
~~~~~

This method can be implemented as follows:


.. code-block:: python

    >>> from AnnoDomini.BFGS import BFGS
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
	
And the visualization could be done:

.. code-block:: python

    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 8, 100))
    Z = f(np.array([X, Y]))
    xmesh, ymesh = np.mgrid[-4:4:80j, -4:4:80j]
    fmesh = f(np.array([xmesh, ymesh]))
    fig = plt.subplots(1,1, figsize = (10,7))
    plt.title('BFGS Path for Rosenbrock’s Function, Starting at [2,1]')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.contour(xmesh, ymesh, fmesh, 50)
    it_array = np.array(ans)
    plt.plot(it_array.T[0], it_array.T[1], "x-", label="Path")
    plt.plot(it_array.T[0][0], it_array.T[1][0], 'xr', label='Initial Guess', markersize=12)
    plt.plot(it_array.T[0][-1], it_array.T[1][-1], 'xg', label='Solution', markersize=12)
    plt.legend()

.. figure:: BFGS.png
    :width: 2000px
    :align: center
    :height: 500px
    :alt: alternate text
    :figclass: align-center

A full demo of this method is available in the demos subdirectory.


Davidon–Fletcher–Powell formula (DFP)
-------------------------------------

Background
~~~~~~~~~~

DFP, or the Davidon–Fletcher–Powell formula, is another
first-order quasi-Newton optimization method, which also approximates the Hessian matrix with the gradient and direction of a function.
The algorithm is as follows, in terms of the Approximate Hessian, :math:`B_k`, the step  :math:`s_k`, :math:`\gamma_k  = \frac{1}{y_k^T s_k}`

1. Solve for :math:`s_k` by solving the linear system :math:`B_k s_k = -y_k`
2. :math:`x_{k+1} = s_k + x_k`
3. :math:`y_k = \nabla x_{k+1} -  \nabla x_{k}`
4. :math:`B_{k+1} =  (I - \gamma_k y_k s_k^T)B_k(I - \gamma_k s_k y_k^T) + \gamma_k y_k y_k^T`
5. Terminate when :math:`s_k <` Tolerance

API
~~~

``DFP Class``:

	- ``f``: function of interest. If f is a scaler function, we can define f as follows:
	
		.. code-block:: python
		
			# option one
			def f(x):
			   return np.sin(x) + x * np.cos(x)
			
			# option two
			f = lambda x: np.sin(x) + x * np.cos(x)
		
		if f is a multivariate function:
		
		.. code-block:: python
			
			def f(args):
				[x, y] = args
				return np.e**(x+1) + np.e**(-y+1) + (x-y)**2
	
	- ``x0``: initial point to start. Support both scaler and vector expressions.
	
		.. code-block:: python
			
			x0 = [1,2,3] # supported
			x0 = 1 # supported
	
	- ``maxiter``: max iteration number if convergence not method
	
	- ``tol``: stop condition parameter. :math: `||x_n - x_{n-1}|| < tol`
	
``find_root``: return a root of a function.


Demos
~~~~~

This method can be implemented as follows:

.. code-block:: python

    >>> from AnnoDomini.DFP import DFP
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
	
.. code-block:: python

    X, Y = np.meshgrid(np.linspace(-3, 3, 100), np.linspace(-2, 8, 100))
    Z = f(np.array([X, Y]))
    xmesh, ymesh = np.mgrid[-4:4:80j, -4:4:80j]
    fmesh = f(np.array([xmesh, ymesh]))
    fig = plt.subplots(1,1, figsize = (10,7))
    plt.title('DPF Path for $f(x,y) = e^{x+1} + e^{1-y} + {(x-y)}^2$ Starting at [2,0]')
    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.contour(xmesh, ymesh, fmesh, 50)
    it_array = np.array(ans)
    plt.plot(it_array.T[0], it_array.T[1], "x-", label="Path")
    plt.plot(it_array.T[0][0], it_array.T[1][0], 'xr', label='Initial Guess', markersize=12)
    plt.plot(it_array.T[0][-1], it_array.T[1][-1], 'xg', label='Solution', markersize=12)
    plt.legend()

.. figure:: DFP.png
    :width: 2000px
    :align: center
    :height: 500px
    :alt: alternate text
    :figclass: align-center

A full demo of this method is available in the demos subdirectory.

.. note::  DFP is empirically significantly less performant than BFPS. For instance, it may take up to 1 million iterations to converge on the Rosenbrock function.

Hamiltonian Monte Carlo
-----------------------


