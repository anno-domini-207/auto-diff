Implementation Details
=======================================

The `AutoDiff` class, which is the core class for the `AnnoDomini` package, takes as input the value to evaluate the function at. It contains two important attributes, ``val`` and ``der``, that respectively store the evaluated function and derivative values at each stage of evaluation.

For instance, consider the case where we want to evaluate the derivative of :math:`x^2+2x+1` evaluated at :math:`x = 5.0`.  This is achieved by first setting ``x = AD.AutoDiff(5.0)``, which automatically stores function and derivative values of :math:`x` evaluated at :math:`x = 5.0` (i.e. ``x.val`` = 5.0 and ``x.der`` = 1.0).  When we pass the desired expression (i.e. :math:`x^2+2x+1`) to Python, :math:`x` is raised to the power of 2, which is carried through `AutoDiff`'s ``__pow__`` method that returns a new `AutoDiff` object with the updated ``val`` and ``der`` values.  Specifically, the new returned object in this case has ``val`` = 25.0 and ``der`` = 10.0, which are the function and derivative values of :math:`x^2` evaluated at :math:`x = 5.0`.  A similar process occurs for the other operation steps (i.e. multiplication and addition), and we eventually obtain the `AutoDiff` object with the desired ``val`` and ``der`` values (i.e. function and derivative values of :math:`x^2+2x+1` evaluated at :math:`x = 5.0`).

The `AutoDiff` class has its own methods that define its behavior for common elementary functions such as addition and multiplication.  Specifically, the class has the following methods implemented:

::

    __add__
    __radd__
    __sub__
    __rsub__
    __mul__
    __rmul__
    __truediv__
    __rtruediv__
    __pow__
    __rpow__
    __neg__
    sqrt
    sin
    cos
    tan
    arcsin
    arccos
    arctan
    sinh
    cosh
    log
    exp
    logistic

Note that many methods in the `AutoDiff` class, such as `cos` and `exp`, utilize their counterparts in NumPy (e.g., `numpy.cos` and `numpy.exp`).  Furthermore, the `AutoDiff` class heavily relies on NumPy arrays and their effective vectorized operations to handle cases with multiple variables and/or multiple functions.  Hence, NumPy is an important external dependency that provides a core data structure for the `AnnoDomini` package.
