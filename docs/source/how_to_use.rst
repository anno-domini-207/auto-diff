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
    >> import AnnoDomini.AutoDiff as AD
    >> AD.AutoDiff(3)
    Function Value: 3 | Derivative Value: 1.0
    >> quit()
    $ deactivate


*Note*: Numpy and Pytest are also required. If they are missing an error message will indicate as much.

How to Use
----------

Consider the following example in scalar input case:s
Suppose we want to to find the derivative of :math:`x^2+2x+1`. We can the utilize the AnnoDomini package as follows:

.. code-block:: python

    import AnnoDomini.AutoDiff as AD
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
