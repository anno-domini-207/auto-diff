# Anno Domini [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Build Status](https://travis-ci.org/anno-domini-207/cs207-FinalProject.svg?branch=master)](https://travis-ci.org/anno-domini-207/cs207-FinalProject.svg?branch=master) [![Coverage Status](https://codecov.io/gh/anno-domini-207/cs207-FinalProject/branch/master/graph/badge.svg)](https://codecov.io/gh/anno-domini-207/cs207-FinalProject) [![Documentation Status](https://readthedocs.org/projects/cs207-finalproject-group15/badge/?version=latest)](https://cs207-finalproject-group15.readthedocs.io/en/latest/?badge=latest)


Anno Domini is equivalent to Automatic Differentiation because they have the same abbreviation (AD).

## Quick Start

### Installation
```
$ pip install virtualenv  # If Necessary
$ virtualenv venv
$ source venv/bin/activate
$ pip install numpy
$ pip install AnnoDomini
$ python
>>> import AnnoDomini.AutoDiff as AD
>>> x = AD.AutoDiff(3.0)  # Automatically evaluate dx/dx at x=3.0
>>> print(x)
====== Function Value(s) ======
3.0
===== Derivative Value(s) =====
1.0
>>> quit()
$ deactivate
```

### Single Variable, Single Function

```python
>>> x = AD.AutoDiff(1.5)
>>> f = x**2 + 2*x + 1  # Automatically evaluate df/dx at x=3.0
>>> print(f)
====== Function Value(s) ======
6.25
===== Derivative Value(s) =====
5.0
```

### Multiple Variables, Single Function

```python
>>> x = AD.AutoDiff(3., [1., 0.])
>>> y = AD.AutoDiff(2., [0., 1.])
>>> f = x*y  # Evaluate J=[df/dx, df/dy] at x=3.0 and y=2.0
>>> print(f)
====== Function Value(s) ======
6.0
===== Derivative Value(s) =====
[2. 3.]
```

### Single Variable, Multiple Functions

```python
>>> x = AD.AutoDiff(3., 1.)
>>> f1 = x**2
>>> f2 = 2*x
>>> print(AD.AutoDiff([f1, f2]))  # Evaluate J=[df1/dx, df2/dx] at x=3.0
====== Function Value(s) ======
[9. 6.]
===== Derivative Value(s) =====
[6. 2.]
```

### Multiple Variables, Multiple Functions

```python
>>> x = AD.AutoDiff(3., [1., 0.])
>>> y = AD.AutoDiff(2., [0., 1.])
>>> f1 = x+y
>>> f2 = x*y
>>> print(AD.AutoDiff([f1, f2]))  # Evaluate J=[[df1/dx, df1/dy], [df2/dx, df2/dy]] at x=3.0 and y=2.0
====== Function Value(s) ======
[5. 6.]
===== Derivative Value(s) =====
[[1. 1.]
 [2. 3.]]
```

## More Resources

**Documentation: [https://cs207-finalproject-group15.readthedocs.io/en/latest/](https://cs207-finalproject-group15.readthedocs.io/en/latest/)**

**PyPI: [https://pypi.org/project/AnnoDomini/](https://pypi.org/project/AnnoDomini/)**


## Authors (CS207 Group 15):

- Simon Warchol
- Kaela Nelson
- Qiuyang Yin
- Sangyoon Park
