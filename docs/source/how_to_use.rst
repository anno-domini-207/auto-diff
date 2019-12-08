How to use Anno Domini
=======================================

How to Install
--------------

**Internal Note: How to Publish to Pip**

.. code-block:: bash


    $ python setup.py sdist
    $ twine upload dist/*

**Install via Pip:**

.. code-block:: bash

    pip install AnnoDomini

**Install in a Virtual Environment:**

.. code-block:: bash

    $ pip install virtualenv # If Necessary
    $ virtualenv venv
    $ source venv/bin/activate
    $ pip install numpy
    $ pip install AnnoDomini
    $ python
    >>> import AnnoDomini.AutoDiff as AD
    >>> AD.AutoDiff(3.0)
    ====== Function Value(s) ======
    3.0
    ===== Derivative Value(s) =====
    1.0
    >>> quit()
    $ deactivate

*Note*: Numpy and Pytest are also required. If they are missing an error message will indicate as much.

How to Use
----------

1. Single Variable, Single Function

Suppose we want to to find the derivative of :math:`x^2+2x+1`. We can the utilize the AnnoDomini package as follows:

.. code-block:: python

    >>> x = AD.AutoDiff(1.5)
    >>> print(x)
    ====== Function Value(s) ======
    1.5
    ===== Derivative Value(s) =====
    1.0
    >>> z = x**2 + 2*x + 1
    >>> print(z)
    ====== Function Value(s) ======
    6.25
    ===== Derivative Value(s) =====
    5.0

We can access only the value or derivative component as follows:

.. code-block:: python

    >>> print(z.val)
    6.25
    >>> print(z.der)
    5.0

Other elementary functions can be used in the same way.  For instance, we may evaluate the derivative of :math:`log_{2}(x)+arctan(3x+5)` at :math:`x = 10.0` as follows:

.. code-block:: python
    >>> x = AutoDiff(10.0)
    >>> z = x.log(2) + np.arctan(3 * x + 5)
    >>> print(z)
    ====== Function Value(s) ======
    4.864160763843499
    ===== Derivative Value(s) =====
    0.14671648614436125

2. Multiple Variables, Single Function

Consider the case where the user would like to input the function,
:math:`f = xy`. Then, the derivative of this would be represented in a Jacobian matrix,
:math:`J = [\frac{df_1}{dx}, \frac{df_1}{dy}] = [y,x]`.

.. code-block:: python

    >>> x = AD.AutoDiff(3., [1., 0.])
    >>> y = AD.AutoDiff(2., [0., 1.])
    >>> z = x*y
    >>> print(z)
    ====== Function Value(s) ======
    6.0
    ===== Derivative Value(s) =====
    [2. 3.]

3. Single Variable, Multiple Functions

Consider the case where the user would like to input the two functions,
:math:`F = [x^2, 2x]`. Then, the derivative of this would be represented in a Jacobian matrix,
:math:`J = [\frac{df_1}{dx}, \frac{df_1}{dy}] = [2x,2]`.

.. code-block:: python

    >>> x = AD.AutoDiff(3., 1.)
    >>> z = AD.AutoDiff([x**2, 2*x])
    >>> print(z)
    ====== Function Value(s) ======
    [9. 6.]
    ===== Derivative Value(s) =====
    [6. 2.]

4. Multiple Variables, Multiple Functions

Consider the case where the user would like to input the two functions,
:math:`F = [x+y, xy]`. Then, the derivative of this would be represented in a Jacobian matrix,
:math:`J = [[\frac{df_1}{dx}, \frac{df_1}{dy}],[\frac{df_2}{dx}, \frac{df_2}{dy}]] = [[1, 1], [y, x]]`.

.. code-block:: python

    >>> x = AD.AutoDiff(3., [1., 0.])
    >>> y = AD.AutoDiff(2., [0., 1.])
    >>> z = AD.AutoDiff([x+y, x*y])
    >>> print(z)
    ====== Function Value(s) ======
    [5. 6.]
    ===== Derivative Value(s) =====
    [[1. 1.]
     [2. 3.]]
